#!/usr/bin/env python3
"""DEV TOOL: send the example jobs in dev/jobs/ to a RUNNING server and print a summary.

Real flow, nothing mocked: POST /jobs -> xMem (Docker) -> AWS price list ->
instance filter -> eager_runtime. Start the server first:

    uvicorn main:app --port 8000

then, from orchestrator/:

    python dev/send_jobs.py                      # every file in dev/jobs/
    python dev/send_jobs.py dev/jobs/01_*.json   # just some
    python dev/send_jobs.py --url http://localhost:8000

Files starting with E are meant to be rejected (HTTP 400).
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

JOBS_DIR = Path(__file__).resolve().parent / "jobs"


def call(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except ValueError:
            return exc.code, {}


def run(url: str, path: Path, timeout_s: float) -> bool:
    body = json.loads(path.read_text(encoding="utf-8"))
    expect_reject = path.stem.startswith("E")
    status, resp = call("POST", f"{url}/jobs", body)
    print(f"\n=== {path.name}  (type={body.get('type')}, engine_mode={body.get('engine_mode')}, "
          f"region={body.get('deployment', {}).get('aws_region')}, budget={body.get('deployment', {}).get('budget_usd_per_hour')})")
    if expect_reject:
        ok = status == 400
        print(f"  {'OK' if ok else 'UNEXPECTED'}: HTTP {status} {resp}")
        return ok
    if status != 202:
        print(f"  FAIL: HTTP {status} {resp}")
        return False

    job_id = resp["job_id"]
    deadline = time.time() + timeout_s
    last = None
    while True:
        _, job = call("GET", f"{url}/jobs/{job_id}")
        if job.get("status") not in ("queued", "running"):
            break
        if job.get("progress") != last:
            last = job.get("progress")
            print(f"  ... {last}")
        if time.time() > deadline:
            print("  FAIL: timeout")
            return False
        time.sleep(2)

    if job.get("status") != "succeeded":
        print(f"  FAIL: {job.get('status')} {job.get('error')}")
        return False
    result = job.get("result") or {}
    vram = (result.get("vram") or {}).get("required_vram_gb")
    runtime = result.get("runtime") or {}
    preds = {p["target_instance"]: p for p in runtime.get("predictions", []) if p.get("status") == "ok"}
    print(f"  cached={job.get('cached')}  VRAM={vram} GB  candidates={len(result.get('candidates', []))}  "
          f"runtime={runtime.get('status')} ok={runtime.get('succeeded')} failed={runtime.get('failed')}")
    if result.get("catalog_error"):
        print(f"  catalog_error: {result['catalog_error']}")
    print(f"  {'instance':13s} {'GPU':6s} {'price/h':>8s} {'median_ms':>10s} {'p90_ms':>10s}")
    for c in result.get("candidates", []):
        p = preds.get(c["instance_type"], {})
        price = "-" if c["price_usd_per_hour"] is None else f"${c['price_usd_per_hour']:.3f}"
        med = p.get("predicted_median_ms")
        p90 = p.get("predicted_p90_ms")
        print(f"  {c['instance_type']:13s} {str(c['gpu_vram_gb'])+'GB':6s} {price:>8s} "
              f"{(f'{med:.2f}' if med is not None else 'ERROR'):>10s} {(f'{p90:.2f}' if p90 is not None else '-'):>10s}")
    return runtime.get("status") == "ok" or not result.get("candidates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=900, help="seconds per job")
    args = parser.parse_args()
    files = args.files or sorted(JOBS_DIR.glob("*.json"))
    results = {f.name: run(args.url.rstrip("/"), f, args.timeout) for f in files}
    print("\n=== summary")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in results.items() if not ok]
    print(f"\n{'ALL PASSED' if not failed else f'{len(failed)} FAILED'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
