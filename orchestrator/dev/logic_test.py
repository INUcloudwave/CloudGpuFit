#!/usr/bin/env python3
"""DEV TOOL: fast logic test for instance_filter / runtime_adapter /
workload_params / cache_key with a FAKE eager_runtime (no torch needed, ~1 s).
Checks control flow only; dev/api_smoke.py is the real end-to-end test.

    python dev/logic_test.py        (from orchestrator/)
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # orchestrator/
sys.path.insert(0, str(HERE))

INSTALLED = {
    "eager": ["g4dn.xlarge", "g5.xlarge", "g6.xlarge", "g6e.xlarge", "g7.2xlarge", "g7e.2xlarge", "gr6.4xlarge"],
    "compiled": ["g4dn.xlarge", "g5.xlarge", "g6.xlarge", "g6e.xlarge", "g7.2xlarge", "g7e.2xlarge", "gr6.4xlarge"],
}
GPU = {
    "g4dn": "Tesla T4", "g5": "NVIDIA A10G", "g6": "NVIDIA L4", "g6e": "NVIDIA L40S",
    "g7": "NVIDIA RTX PRO 4500 Blackwell Server Edition", "g7e": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
}
calls: list[dict] = []

fake = types.ModuleType("eager_runtime")
fake.available_targets = lambda engine_mode=None, runtime_root=None: list(INSTALLED[engine_mode or "eager"])
fake.target_provenance = lambda instance_type, runtime_root=None: {"status": "measured"}


def fake_predict_runtime_targets(request, targets, *, engine_mode=None, raise_on_error=False):
    calls.append({"request": dict(request), "targets": list(targets), "engine_mode": engine_mode})
    return {
        "status": "ok",
        "model": request["model"],
        "predictions": [
            {"status": "ok", "target_instance": t, "engine_mode": engine_mode,
             "predicted_median_ms": 5.0, "predicted_p90_ms": 7.0, "warnings": []}
            for t in targets
        ],
    }


fake.predict_runtime_targets = fake_predict_runtime_targets
sys.modules["eager_runtime"] = fake

# fake target manifests so instance_filter can resolve GPU families
tmp = Path(tempfile.mkdtemp())
for mode_targets in INSTALLED.values():
    for t in mode_targets:
        d = tmp / "targets" / t
        d.mkdir(parents=True, exist_ok=True)
        manifest = {"target_instance": t, "gpu_name": GPU["g6" if t.startswith("gr6") else t.split(".")[0]]}
        if t.startswith("gr6"):
            manifest["size_profile"] = {"gpu_family": "g6", "vcpu": 16, "ram_gib": None}
        (d / "runtime_manifest.json").write_text(json.dumps(manifest))
(tmp / "targets" / "g7.2xlarge" / "compiled_context_calibration.json").write_text('{"fixed_overhead_ms": 0.3}')

import instance_filter  # noqa: E402
import runtime_adapter  # noqa: E402
import cache_key  # noqa: E402
import workload_params  # noqa: E402

instance_filter.RUNTIME_ROOT = tmp
failures = []


def check(ok, label):
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        failures.append(label)


catalog = [{"instance_type": n, "aws_region": "r", "price_usd_per_hour": None}
           for n in INSTALLED["eager"] + ["g5g.xlarge", "g6.2xlarge", "g6.16xlarge"]]
specs = instance_filter.supported_instances("eager", tmp)
cands = instance_filter.filter_instances(2.0, "r", None, catalog, supported=specs)
names = [c["instance_type"] for c in cands]
check("g7.2xlarge" in names, "g7.2xlarge is a candidate")
check(sorted(names) == sorted(INSTALLED["eager"]), "only installed targets are candidates")
check(len(instance_filter.SUPPORTED_GPU_INSTANCES) == 26 and "g5g.xlarge" not in instance_filter.SUPPORTED_GPU_INSTANCES
      and "g6.2xlarge" in instance_filter.SUPPORTED_GPU_INSTANCES,
      "instance list = original minus g5g, plus g7.2xlarge (26)")
check(not {"g5g.xlarge", "g6.2xlarge", "g6.16xlarge"} & set(names), "g5g / unmeasured sizes excluded")
check(next(c for c in cands if c["instance_type"] == "g7.2xlarge")["gpu_vram_gb"] == 32, "g7.2xlarge = 32 GB GPU")
check(next(c for c in cands if c["instance_type"] == "gr6.4xlarge")["gpu_name"] == "NVIDIA L4", "gr6 resolved via size_profile family g6")

big = [c["instance_type"] for c in instance_filter.filter_instances(40.0, "r", None, catalog, supported=specs)]
check(sorted(big) == ["g6e.xlarge", "g7e.2xlarge"], "VRAM filter")

# wrong GPU in a manifest must not be shown with the family's specs
(tmp / "targets" / "g5.xlarge" / "runtime_manifest.json").write_text(json.dumps({"target_instance": "g5.xlarge", "gpu_name": "NVIDIA L4"}))
check("g5.xlarge" not in instance_filter.supported_instances("eager", tmp), "GPU mismatch -> not recommended")
(tmp / "targets" / "g5.xlarge" / "runtime_manifest.json").write_text(json.dumps({"target_instance": "g5.xlarge", "gpu_name": "NVIDIA A10G"}))

payload = {"model": "m", "type": "audio_encoder", "dtype": "fp16", "batch_size": 1,
           "params": {"mode": "prefill", "audio_samples": 16000}, "model_config_snapshot": {"model_type": "wav2vec2"}}
r = runtime_adapter.predict_candidate_runtimes(payload, ["g7.2xlarge", "g5g.xlarge"], "compiled")
check(r["status"] == "ok" and r["succeeded"] == 1, "audio_encoder accepted")
check(r["not_installed_targets"] == ["g5g.xlarge"], "target without artifacts reported, not proxied")
check(calls[-1]["targets"] == ["g7.2xlarge"] and calls[-1]["engine_mode"] == "compiled", "only direct targets predicted, engine_mode passed")
check("engine_mode" not in calls[-1]["request"], "engine_mode not smuggled in the request dict")
check(r["predictions"][0]["predicted_median_ms"] == 5.0, "no vCPU scaling applied")
check(runtime_adapter.predict_candidate_runtimes({**payload, "type": "audio_encdec"}, ["g6.xlarge"])["status"] == "skipped", "audio_encdec refused")
check(runtime_adapter.predict_candidate_runtimes(payload, ["g6.xlarge"], "bogus")["error_code"] == "unsupported_engine_mode", "bad engine_mode refused")

try:
    workload_params.validate_params("audio_encoder", {"audio_samples": 32000})
    check(True, "audio_encoder params accepted")
except ValueError as exc:
    check(False, f"audio_encoder params accepted ({exc})")
try:
    workload_params.validate_params("audio_encoder", {"audio_samples": 32000, "mode": "generate"})
    check(False, "audio_encoder generate refused")
except ValueError:
    check(True, "audio_encoder generate refused")
check(set(instance_filter.recommendable_instance_types(tmp)) == set(INSTALLED["eager"]) | set(INSTALLED["compiled"]),
      "price fetch set = recommendable instances")

k1 = cache_key.pipeline_cache_key_for({}, {**payload, "engine_mode": "eager"}, eager_runtime_root=tmp, catalog_version="v")
k2 = cache_key.pipeline_cache_key_for({}, {**payload, "engine_mode": "compiled"}, eager_runtime_root=tmp, catalog_version="v")
(tmp / "targets" / "g7.2xlarge" / "compiled_context_calibration.json").write_text('{"fixed_overhead_ms": 0.4}')
k3 = cache_key.pipeline_cache_key_for({}, {**payload, "engine_mode": "compiled"}, eager_runtime_root=tmp, catalog_version="v")
check(k1 != k2, "cache key includes engine_mode")
check(k2 != k3, "cache key includes compiled_context_calibration.json")

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
raise SystemExit(1 if failures else 0)
