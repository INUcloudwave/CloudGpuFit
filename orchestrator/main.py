"""CloudGPUFit orchestrator: a job queue in front of the xMem CPU-profiling
subprocess. Concurrency is fixed at 1 — xMem loads a full FP32 copy of the
model into host RAM to profile it, so two runs at once tend to OOM each
other. A single asyncio worker task enforces that; there is no external
queue system.

Since CLAUDE_HANDOFF_CLOUDGPUFIT_EAGER_RUNTIME.md's integration: after a
VRAM prediction succeeds, process_job() also (1) filters AWS single-GPU
G-series instances down to ones that fit the predicted VRAM/region/budget
and have an installed eager_runtime target, then (2) asks eager_runtime for
Eager FP16 Median/P90 on each of those candidates. eager_runtime is
imported directly in-process (no dedicated Runtime HTTP server, per §1).
"""

import asyncio
import functools
import json
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import db
import setup_manager
from cache_key import pipeline_cache_key_for
from eager_runtime import available_targets
from instance_catalog import catalog_version, load_instance_catalog
from instance_filter import filter_instances
from runtime_adapter import predict_candidate_runtimes
from xmem_runner import build_xmem_args, run_xmem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

app = FastAPI(title="CloudGPUFit Orchestrator")

job_queue: "asyncio.Queue[str]" = asyncio.Queue()
setup_lock = asyncio.Lock()

# The other 3 workload types aren't wired into predict_vram.py yet (no
# encoder-decoder / audio dataset+loop support) — reject them here instead of
# handing the subprocess a shape it can't run. NOTE: this stays {"decoder_llm"}
# even though eager_runtime itself also supports vision_cls/text_encdec —
# the VRAM step is still the gate, and only the intersection of what every
# stage actually supports counts as pipeline-supported (CLAUDE_HANDOFF §14).
SUPPORTED_TYPES = {"decoder_llm"}
CONTAINER_NAME = "xmem-cpu"
EAGER_RUNTIME_ROOT = Path(__file__).parent / "eager_runtime"


class DeploymentIn(BaseModel):
    aws_region: str
    budget_usd_per_hour: float


class JobCreate(BaseModel):
    model: str
    type: str
    dtype: str
    batch_size: int
    params: dict[str, Any]
    deployment: DeploymentIn
    warnings: list[str] = []


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
    deployment = payload.get("deployment") or {}
    instance_catalog = load_instance_catalog()
    candidates = filter_instances(
        required_vram_gb=vram_outcome.result["required_vram_gb"],
        aws_region=deployment.get("aws_region"),
        budget_usd_per_hour=deployment.get("budget_usd_per_hour"),
        instance_catalog=instance_catalog,
        installed_targets=available_targets(),
    )

    runtime_result = await asyncio.to_thread(
        predict_candidate_runtimes,
        payload,
        [candidate["instance_type"] for candidate in candidates],
    )

    result = {
        "vram": vram_outcome.result,
        "candidates": candidates,
        "runtime": runtime_result,
    }
    db.mark_finished(job_id, "succeeded", result=result)


@app.post("/jobs", status_code=202)
async def create_job(body: JobCreate):
    if body.type not in SUPPORTED_TYPES:
        return JSONResponse(status_code=400, content={"error_code": "unsupported_type"})

    payload = body.model_dump()
    xmem_args = build_xmem_args(payload)
    cache_key = pipeline_cache_key_for(
        xmem_args,
        payload,
        eager_runtime_root=EAGER_RUNTIME_ROOT,
        catalog_version=catalog_version(),
    )

    job_id = str(uuid.uuid4())

    cached_row = db.find_succeeded_by_cache_key(cache_key)
    if cached_row is not None:
        result = json.loads(cached_row["result_json"])
        ts = db.now_iso()
        db.insert_job(
            job_id, "succeeded", payload, cache_key,
            cached=True, result=result, started_at=ts, finished_at=ts,
        )
        return {"job_id": job_id, "status": "succeeded", "cached": True}

    db.insert_job(job_id, "queued", payload, cache_key, cached=False)
    await job_queue.put(job_id)
    return {"job_id": job_id, "status": "queued", "cached": False}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    row = db.get_job(job_id)
    if row is None:
        return JSONResponse(status_code=404, content={"error_code": "not_found"})
    return db.job_to_dict(row)


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
