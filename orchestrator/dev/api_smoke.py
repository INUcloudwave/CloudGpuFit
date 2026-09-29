#!/usr/bin/env python3
"""DEV TOOL: API smoke test through the real FastAPI app (main.py).

POST /jobs -> worker -> process_job -> instance_filter -> runtime_adapter ->
eager_runtime, then GET /jobs/{id}, for Eager and Compiled and four workload
types. Only the parts that need Docker or the network are replaced:

  * run_xmem            -> a fixed VRAM result (TEST value, not a prediction)
  * load_instance_catalog -> every recommendable instance plus g6.2xlarge,
                            g6.4xlarge, g5.2xlarge (no target yet), all with
                            price None ("available, price unknown")
  * the job DB          -> a temporary SQLite file (jobs.db is untouched)

Run from orchestrator/ (needs fastapi, httpx, and requirements-runtime.txt):

    python dev/api_smoke.py            # 4 workloads x 2 modes (~5 min)
    python dev/api_smoke.py --quick    # decoder_llm only
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ORCHESTRATOR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ORCHESTRATOR))

import db  # noqa: E402

db.DB_PATH = Path(tempfile.mkdtemp()) / "api_smoke_jobs.db"

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from instance_filter import recommendable_instance_types  # noqa: E402
from xmem_runner import RunOutcome  # noqa: E402

REGION = "ap-northeast-2"
NOT_MEASURED = ["g6.2xlarge", "g6.4xlarge", "g5.2xlarge"]
failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(("  PASS " if ok else "  FAIL ") + label + ("" if ok else f": {detail}"))
    if not ok:
        failures.append(f"{label}: {detail}")


def fake_catalog(aws_region=None):
    return [
        {"instance_type": name, "aws_region": REGION, "price_usd_per_hour": None}
        for name in recommendable_instance_types() + NOT_MEASURED
    ]


VRAM_GB = {"value": 2.0}
main.run_xmem = lambda args, on_progress=None: RunOutcome(
    status="succeeded", result={"status": "ok", "required_vram_gb": VRAM_GB["value"]}
)
main.load_instance_catalog = fake_catalog


def jobs() -> dict[str, dict]:
    sys.path.insert(0, str(ORCHESTRATOR / "eager_runtime"))
    import compiled_models
    from transformers import ResNetConfig

    def cfg(name):
        return compiled_models.build_transformer_config(name).to_dict()

    resnet = ResNetConfig(depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 256, 512], layer_type="basic").to_dict()
    resnet["architectures"] = ["ResNetForImageClassification"]
    base = {"dtype": "fp16", "batch_size": 8, "deployment": {"aws_region": REGION, "budget_usd_per_hour": 100.0}}
    return {
        "decoder_llm": {**base, "model": "facebook/opt-125m", "type": "decoder_llm",
                        "model_config_snapshot": cfg("opt"), "params": {"seq_len": 128, "mode": "prefill"}},
        "audio_encoder": {**base, "model": "facebook/wav2vec2-base", "type": "audio_encoder",
                          "model_config_snapshot": cfg("wav2vec2"), "params": {"audio_samples": 32000}},
        "vision_cls": {**base, "model": "microsoft/resnet-18", "type": "vision_cls",
                       "model_config_snapshot": resnet, "params": {"resolution": 224}},
        "text_encdec": {**base, "model": "t5-small", "type": "text_encdec", "model_config_snapshot": cfg("t5"),
                        "params": {"encoder_seq_len": 128, "decoder_seq_len": 1, "mode": "prefill"}},
    }


def wait(client: TestClient, job_id: str, timeout_s: float = 600) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body.get("status") not in ("queued", "running"):
            return body
        time.sleep(0.5)
    return {"status": "timeout"}


def main_() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    all_jobs = jobs()
    if args.quick:
        all_jobs = {"decoder_llm": all_jobs["decoder_llm"]}
    recommendable = recommendable_instance_types()

    with TestClient(main.app) as client:
        for name, payload in all_jobs.items():
            medians = {}
            for mode in ("eager", "compiled"):
                print(f"\n[POST /jobs {name} engine_mode={mode}]")
                resp = client.post("/jobs", json={**payload, "engine_mode": mode})
                check(resp.status_code == 202, "accepted (202)", f"{resp.status_code} {resp.text[:200]}")
                if resp.status_code != 202:
                    continue
                job = wait(client, resp.json()["job_id"])
                check(job.get("status") == "succeeded", "job succeeded", str(job)[:300])
                result = job.get("result") or {}
                cands = sorted(c["instance_type"] for c in result.get("candidates", []))
                check(cands == sorted(recommendable), "candidates = instances with their own target", str(cands))
                check("g7.2xlarge" in cands, "g7.2xlarge recommended")
                check(not set(cands) & set(NOT_MEASURED), "unmeasured sizes not recommended")
                runtime = result.get("runtime") or {}
                preds = [p for p in runtime.get("predictions", []) if p.get("status") == "ok"]
                check(runtime.get("status") == "ok" and len(preds) == len(cands), "all runtime predictions ok",
                      f"{runtime.get('status')} {runtime.get('error_message')}")
                check(result.get("engine_mode") == mode and all(p["engine_mode"] == mode for p in preds),
                      f"engine_mode={mode} end to end")
                check(not any("vCPU" in w or "borrowed" in w for p in preds for w in p.get("warnings", [])),
                      "no borrowed / vCPU-scaled results")
                medians[mode] = {p["target_instance"]: p["predicted_median_ms"] for p in preds}
                print("  " + ", ".join(f"{k}={v:.3f}ms" for k, v in sorted(medians[mode].items())))
            if len(medians) == 2:
                check(medians["eager"] != medians["compiled"], f"{name}: eager and compiled differ")

        print("\n[rejections]")
        bad = client.post("/jobs", json={**all_jobs["decoder_llm"], "type": "audio_encdec"})
        check(bad.status_code == 400 and bad.json().get("error_code") == "unsupported_type",
              "audio_encdec rejected (400 unsupported_type)", bad.text[:200])
        bad = client.post("/jobs", json={**all_jobs["decoder_llm"], "engine_mode": "tensorrt"})
        check(bad.status_code == 400, "unknown engine_mode rejected (400)", bad.text[:200])

        print("\n[VRAM 40 GB]")
        VRAM_GB["value"] = 40.0
        job = wait(client, client.post("/jobs", json=all_jobs["decoder_llm"]).json()["job_id"])
        cands = sorted(c["instance_type"] for c in (job.get("result") or {}).get("candidates", []))
        check(cands == ["g6e.xlarge", "g7e.2xlarge"], "only g6e (48 GB) and g7e (96 GB) fit", str(cands))

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
    for failure in failures:
        print("  FAIL", failure)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main_()
