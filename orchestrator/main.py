"""CloudGPUFit orchestrator: a job queue in front of the xMem CPU-profiling
subprocess. Concurrency is fixed at 1 — xMem loads a full FP32 copy of the
model into host RAM to profile it, so two runs at once tend to OOM each
other. A single asyncio worker task enforces that; there is no external
queue system.

After a VRAM prediction succeeds, process_job() also (1) filters AWS
single-GPU instances down to ones that fit the predicted VRAM/region/budget
AND have their own eager_runtime target (7 today: g4dn.xlarge, g5.xlarge,
g6.xlarge, g6e.xlarge, g7.2xlarge, g7e.2xlarge, gr6.4xlarge), then (2) asks
eager_runtime for FP16 Median/P90 on each of those candidates, in the job's
engine_mode ("eager" default, or "compiled"). eager_runtime is imported
directly in-process (no dedicated Runtime HTTP server).
"""

import asyncio
import functools
import json
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

import aws_pricing
import db
import setup_manager
from cache_key import pipeline_cache_key_for
from instance_catalog import catalog_version, load_instance_catalog
from instance_filter import filter_instances
from runtime_adapter import predict_candidate_runtimes
from workload_params import validate_params
from xmem_runner import build_xmem_args, run_xmem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

app = FastAPI(title="CloudGPUFit Orchestrator")

job_queue: "asyncio.Queue[str]" = asyncio.Queue()
setup_lock = asyncio.Lock()

# Workload types the Runtime side predicts. audio_encoder (Wav2Vec2/HuBERT
# encoder-only) is supported by eager_runtime; its VRAM-side support
# (predict_vram.py) and web form are owned by the VRAM/web side.
SUPPORTED_TYPES = {"decoder_llm", "vision_cls", "text_encdec", "audio_encoder"}
CONTAINER_NAME = "xmem-cpu"
EAGER_RUNTIME_ROOT = Path(__file__).parent / "eager_runtime"


class DeploymentIn(BaseModel):
    aws_region: str = Field(min_length=1)
    # allow_inf_nan=False because Python's json.loads accepts bare NaN/Infinity
    # tokens, so a non-browser client can otherwise put one straight into the
    # budget filter's comparisons.
    budget_usd_per_hour: float = Field(gt=0, allow_inf_nan=False)


class JobCreate(BaseModel):
    model: str = Field(min_length=1)
    # Full config.json the web app already fetched when the user picked this
    # model. Named *_snapshot, not model_config -- "model_config" is a
    # reserved Pydantic v2 attribute (model configuration, not a field) and
    # using it here raises PydanticUserError.
    model_config_snapshot: dict[str, Any]
    # Left as a plain str, not a Literal: create_job's SUPPORTED_TYPES check
    # already rejects unrunnable types with an error code the UI handles, and
    # a Literal here would turn that into a generic validation failure instead.
    type: str
    # predict_vram.py's argparse has choices=[...] for --dtype, but the --json
    # path setattr()s straight past it, so an unknown dtype would only surface
    # as a KeyError deep inside DTYPE_SCALE.
    dtype: Literal["fp32", "fp16", "bf16"]
    batch_size: int = Field(gt=0)
    params: dict[str, Any]
    deployment: DeploymentIn
    warnings: list[str] = []
    # engine_mode is eager_runtime's own concept (predict_runtime.py's
    # engine_mode dispatch, added alongside the Compiled/Inductor predictor),
    # not part of the original CLAUDE_HANDOFF payload shape. Defaulting to
    # "eager" here means any client that hasn't been updated to send this
    # field yet keeps getting exactly the pre-Compiled behavior.
    engine_mode: Literal["eager", "compiled"] = "eager"

    @model_validator(mode="after")
    def _check_params(self) -> "JobCreate":
        validate_params(self.type, self.params)
        return self


class SloIn(BaseModel):
    """The user's acceptance bar, not a workload/arrival-rate input — it never
    feeds VRAM/Runtime prediction and submitting it never starts a new job."""

    max_latency_ms: float = Field(gt=0, allow_inf_nan=False)
    min_throughput_rps: float = Field(gt=0, allow_inf_nan=False)


def _validation_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "입력값이 올바르지 않습니다."
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    message = str(first.get("msg", "")).removeprefix("Value error, ")
    return f"{location}: {message}" if location else message


@app.exception_handler(RequestValidationError)
async def on_invalid_payload(_request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's stock 422 body is a nested pydantic error list the UI can't
    # render; every other endpoint here answers with {"error_code": ...}.
    return JSONResponse(
        status_code=400,
        content={"error_code": "invalid_payload", "message": _validation_message(exc)},
    )


@app.on_event("startup")
async def on_startup() -> None:
    db.init_db()
    asyncio.create_task(worker_loop())


async def worker_loop() -> None:
    while True:
        job_id = await job_queue.get()
        try:
            await process_job(job_id)
        except Exception as exc:  # noqa: BLE001 - the loop must survive any single job's failure
            logger.exception("job %s crashed the worker step", job_id)
            db.mark_finished(job_id, "failed", error_code="internal", error_message=str(exc))
        finally:
            job_queue.task_done()


async def process_job(job_id: str) -> None:
    row = db.get_job(job_id)
    if row is None:
        return

    payload = json.loads(row["payload_json"])
    db.mark_running(job_id)

    xmem_args = build_xmem_args(payload)
    on_progress = functools.partial(db.set_progress, job_id)
    vram_outcome = await asyncio.to_thread(run_xmem, xmem_args, on_progress)

    if vram_outcome.status != "succeeded":
        db.mark_finished(
            job_id,
            vram_outcome.status,
            error_code=vram_outcome.error_code,
            error_message=vram_outcome.error_message,
        )
        return

    # VRAM succeeded -> filter AWS candidates -> predict Runtime for them.
    # A candidate list can legitimately come back empty (no catalog data yet,
    # nothing fits budget/region, ...) -- that is a normal result, not a
    # reason to mark the job failed (CLAUDE_HANDOFF §10, §17 test #7).
    # Each set_progress() call here marks a real code transition, not a
    # timer-driven guess -- there is no step between these two calls slow
    # enough to deserve its own phase (filter_instances is in-memory).
    # engine_mode picks which eager_runtime predictor (Eager or Compiled)
    # answers this job; defaults to "eager" for any payload written before
    # this field existed (old queued rows, or a client that hasn't been
    # updated yet) so behavior for those is unchanged.
    engine_mode = str(payload.get("engine_mode") or "eager")

    db.set_progress(job_id, "AWS GPU 인스턴스 정보 조회 및 후보 필터링 중…")
    deployment = payload.get("deployment") or {}
    aws_region = deployment.get("aws_region")
    catalog_error: Optional[str] = None
    try:
        instance_catalog = await asyncio.to_thread(load_instance_catalog, aws_region)
    except aws_pricing.PriceFetchError as exc:
        # A live-fetch failure is not "no candidates fit" -- keep them
        # distinguishable in the result instead of collapsing both into an
        # empty candidates list with no explanation.
        instance_catalog = []
        catalog_error = str(exc)
    candidates = filter_instances(
        required_vram_gb=vram_outcome.result["required_vram_gb"],
        gpu_requirements=vram_outcome.result.get("gpu_requirements"),
        aws_region=aws_region,
        budget_usd_per_hour=deployment.get("budget_usd_per_hour"),
        instance_catalog=instance_catalog,
        # Candidates are only instances with their own target for the
        # job's engine_mode -- never a proxy for an unmeasured size.
        engine_mode=engine_mode,
    )

    db.set_progress(job_id, "후보 인스턴스별 성능 예측 중…")
    runtime_result = await asyncio.to_thread(
        predict_candidate_runtimes,
        payload,
        [candidate["instance_type"] for candidate in candidates],
        engine_mode,
    )

    # generate is the only mode the UI sends now, so predict_vram.py's
    # "mode=generate is EXPERIMENTAL" note would show on every LLM job.
    vram_result = dict(vram_outcome.result)
    vram_result["warnings"] = [
        w for w in vram_result.get("warnings", []) if not w.startswith("mode=generate is EXPERIMENTAL")
    ]

    result = {
        "vram": vram_result,
        "candidates": candidates,
        "catalog_error": catalog_error,
        "engine_mode": engine_mode,
        "runtime": runtime_result,
    }
    db.mark_finished(job_id, "succeeded", result=result)


@app.post("/jobs", status_code=202)
async def create_job(body: JobCreate):
    if body.type not in SUPPORTED_TYPES:
        return JSONResponse(status_code=400, content={"error_code": "unsupported_type"})

    payload = body.model_dump()
    xmem_args = build_xmem_args(payload)

    # Warms the region's price cache (if it's cold) *before* the cache key is
    # computed, so catalog_version() below reflects the data process_job()
    # will actually read later instead of a "not fetched yet" placeholder
    # from this instant -- otherwise two identical requests straddling that
    # fetch would get different cache keys for the same underlying catalog.
    aws_region = body.deployment.aws_region
    try:
        await asyncio.to_thread(load_instance_catalog, aws_region)
    except aws_pricing.PriceFetchError as exc:
        logger.warning("AWS 가격 사전 조회 실패 (%s): %s", aws_region, exc)

    cache_key = pipeline_cache_key_for(
        xmem_args,
        payload,
        eager_runtime_root=EAGER_RUNTIME_ROOT,
        catalog_version=catalog_version(aws_region),
    )

    job_id = str(uuid.uuid4())

    # Cache-hit reuse disabled on purpose (was: db.find_succeeded_by_cache_key()
    # short-circuiting straight to a stored old result). During active
    # development of runtime_adapter.py/instance_filter.py, a match on
    # cache_key silently served a result computed by an OLDER version of that
    # code, which is indistinguishable from a bug in the running code unless
    # you already suspect it. cache_key is still computed and stored on every
    # row (still useful for debugging/inspection), it's just never looked up
    # to skip real work anymore. Every submission always runs the full
    # pipeline.
    db.insert_job(job_id, "queued", payload, cache_key, cached=False)
    await job_queue.put(job_id)
    return {"job_id": job_id, "status": "queued", "cached": False}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    row = db.get_job(job_id)
    if row is None:
        return JSONResponse(status_code=404, content={"error_code": "not_found"})
    return db.job_to_dict(row)


@app.patch("/jobs/{job_id}/slo")
async def set_job_slo(job_id: str, body: SloIn):
    # Deliberately does not touch job_queue/cache_key/status: the SLO is
    # evaluated against whatever this job already computes (or is
    # computing), never a reason to compute anything again.
    row = db.get_job(job_id)
    if row is None:
        return JSONResponse(status_code=404, content={"error_code": "not_found"})
    db.set_slo(job_id, body.model_dump())
    return db.job_to_dict(db.get_job(job_id))


@app.get("/health")
async def health():
    try:
        proc = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        ok = proc.returncode == 0 and proc.stdout.strip() == "true"
    except Exception:  # noqa: BLE001 - docker missing/unreachable just means "not healthy"
        ok = False
    return {"ok": ok}


@app.post("/setup")
async def setup_environment():
    """Ensures xmem-cpu exists and is running. Called from the UI right after
    a model is picked (Step 1 -> Step 2), so the container is usually already
    warm by the time the user finishes the params form. The very first call
    on a fresh machine does the full pull+clone+install and can take minutes;
    every call after that is a fast docker inspect."""
    if setup_lock.locked():
        return JSONResponse(status_code=409, content={"status": "in_progress"})

    async with setup_lock:
        try:
            return await asyncio.to_thread(setup_manager.ensure_ready)
        except setup_manager.SetupError as exc:
            return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})
