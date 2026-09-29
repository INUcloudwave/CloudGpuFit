#!/usr/bin/env python3
"""DEV TOOL (not imported by the service): predict from a request JSON file.

Runs the same post-VRAM steps as main.process_job() -- instance_filter ->
runtime_adapter -> eager_runtime -- without xMem, the DB or the web app, and
prints a table of recommended instances. Run from orchestrator/:

    python dev/predict_from_file.py dev/requests/opt_decoder.json
    python dev/predict_from_file.py dev/requests/wav2vec2_audio.json --json result.json

Request file fields (see dev/requests/*.json):

  model, type, dtype, batch_size, params      same as the /jobs payload
  model_config_preset | model_config_snapshot the model's config:
      preset  = one of PRESETS below (offline, no download), or
      snapshot = the model's config.json contents, or
      neither  = fetched from the Hugging Face Hub by model id (needs internet)
  engine_mode         "eager" (default), "compiled", or "both"
  vram_required_gb    TEST value standing in for the xMem VRAM prediction
  aws_region          region name used for the catalog rows
  budget_usd_per_hour optional; only applied to instances with a price
  instances           "all" (every recommendable instance) or a list of names.
                      Rows are created with price unknown (None) -- this file
                      does not contain or invent AWS prices. Add
                      "prices": {"g6.xlarge": 0.98, ...} with prices you
                      looked up if you want the budget filter exercised.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ORCHESTRATOR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ORCHESTRATOR))

import eager_runtime as er  # noqa: E402
from cache_key import pipeline_cache_key_for  # noqa: E402
from instance_filter import filter_instances, recommendable_instance_types  # noqa: E402
from runtime_adapter import predict_candidate_runtimes  # noqa: E402
from workload_params import validate_params  # noqa: E402
from xmem_runner import build_xmem_args  # noqa: E402

SUPPORTED_TYPES = {"decoder_llm", "vision_cls", "text_encdec", "audio_encoder"}  # = main.SUPPORTED_TYPES


def run_pipeline(payload, vram_result, instance_catalog):
    """main.process_job()'s steps after a successful VRAM prediction."""
    engine_mode = str(payload.get("engine_mode") or "eager")
    if payload.get("type") not in SUPPORTED_TYPES:
        return {"status": "rejected", "error_code": "unsupported_type",
                "error_message": f"type must be one of {sorted(SUPPORTED_TYPES)}"}
    try:
        validate_params(payload["type"], payload.get("params") or {})
    except ValueError as exc:
        return {"status": "rejected", "error_code": "invalid_payload", "error_message": str(exc)}
    deployment = payload.get("deployment") or {}
    candidates = filter_instances(
        required_vram_gb=vram_result["required_vram_gb"],
        aws_region=deployment.get("aws_region"),
        budget_usd_per_hour=deployment.get("budget_usd_per_hour"),
        instance_catalog=instance_catalog,
        engine_mode=engine_mode,
    )
    runtime = predict_candidate_runtimes(payload, [c["instance_type"] for c in candidates], engine_mode)
    cache_key = pipeline_cache_key_for(
        build_xmem_args(payload), payload,
        eager_runtime_root=ORCHESTRATOR / "eager_runtime", catalog_version="dev",
    )
    return {"status": "succeeded", "cache_key": cache_key, "engine_mode": engine_mode,
            "vram": vram_result, "candidates": candidates, "runtime": runtime}

CATALOG: list[dict[str, Any]] = []
PRESETS = ("opt", "qwen2", "bert", "roberta", "t5", "wav2vec2", "resnet18_hf", "vit_base_hf")


def preset_config(name: str) -> dict[str, Any]:
    if name == "resnet18_hf":
        from transformers import ResNetConfig

        config = ResNetConfig(depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 256, 512], layer_type="basic").to_dict()
        config["architectures"] = ["ResNetForImageClassification"]
        return config
    if name == "vit_base_hf":
        from transformers import ViTConfig

        config = ViTConfig().to_dict()
        config["architectures"] = ["ViTForImageClassification"]
        return config
    import compiled_models  # importable once eager_runtime is imported

    return compiled_models.build_transformer_config(name).to_dict()


def build_payload(spec: dict[str, Any]) -> dict[str, Any]:
    payload = {key: spec[key] for key in ("model", "type", "dtype", "batch_size", "params") if key in spec}
    payload.setdefault("dtype", "fp16")
    payload["deployment"] = {
        "aws_region": spec.get("aws_region", "ap-northeast-2"),
        "budget_usd_per_hour": spec.get("budget_usd_per_hour"),
    }
    if spec.get("model_config_snapshot"):
        payload["model_config_snapshot"] = spec["model_config_snapshot"]
    elif spec.get("model_config_preset"):
        preset = spec["model_config_preset"]
        if preset not in PRESETS:
            raise SystemExit(f"model_config_preset must be one of {PRESETS}, got {preset!r}")
        payload["model_config_snapshot"] = preset_config(preset)
    return payload


def build_catalog(spec: dict[str, Any], region: str) -> list[dict[str, Any]]:
    names = spec.get("instances", "all")
    if names == "all":
        names = recommendable_instance_types()
    prices = spec.get("prices") or {}
    return [{"instance_type": n, "aws_region": region, "price_usd_per_hour": prices.get(n)} for n in names]


def print_result(mode: str, result: dict[str, Any]) -> None:
    print(f"\n=== engine_mode={mode}  status={result.get('status')} ===")
    if result.get("status") == "rejected":
        print(f"rejected: {result.get('error_code')} - {result.get('error_message')}")
        return
    runtime = result["runtime"]
    if runtime.get("status") not in ("ok", "partial"):
        print(f"runtime: {runtime.get('status')} - {runtime.get('error_message')}")
        return
    spec_by_name = {c["instance_type"]: c for c in result["candidates"]}
    rows = []
    for p in runtime["predictions"]:
        c = spec_by_name.get(p["target_instance"], {})
        if p.get("status") != "ok":
            rows.append((p["target_instance"], c.get("gpu_name", ""), "-", "-", "-", f"error: {p.get('error')}"))
            continue
        price = c.get("price_usd_per_hour")
        rows.append((
            p["target_instance"],
            c.get("gpu_name", ""),
            f"{p['predicted_median_ms']:.3f}",
            f"{p['predicted_throughput_rps']:.1f}",
            "-" if price is None else f"{price:.3f}",
            "; ".join(p.get("warnings", []))[:80],
        ))
    rows.sort(key=lambda r: float(r[2]) if r[2] != "-" else float("inf"))
    first = next((p for p in runtime["predictions"] if p.get("status") == "ok"), {})
    print(f"model={runtime.get('model')}  workload={first.get('workload_type')}  "
          f"scope={first.get('prediction_scope')}  (median is the official estimate)")
    print(f"{'instance':13s} {'GPU':46s} {'median_ms':>10s} {'req/s':>8s} {'$/h':>7s}  warnings")
    for r in rows:
        print(f"{r[0]:13s} {r[1]:46s} {r[2]:>10s} {r[3]:>8s} {r[4]:>7s}  {r[5]}")
    requested = {row["instance_type"] for row in CATALOG}
    excluded = sorted(requested - set(spec_by_name))
    if excluded:
        print("not recommended (no installed target, not enough GPU memory, or over budget):", excluded)
    if runtime.get("not_installed_targets"):
        print("not predicted (no installed target):", runtime["not_installed_targets"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict from a request JSON file.")
    parser.add_argument("request", type=Path)
    parser.add_argument("--json", type=Path, help="also write the full result JSON here")
    args = parser.parse_args()

    spec = json.loads(args.request.read_text(encoding="utf-8"))
    payload = build_payload(spec)
    catalog = build_catalog(spec, payload["deployment"]["aws_region"])
    CATALOG[:] = catalog
    vram = {"status": "ok", "required_vram_gb": float(spec.get("vram_required_gb", 0.0))}
    mode = str(spec.get("engine_mode", "eager"))
    modes = ["eager", "compiled"] if mode == "both" else [mode]

    results = {}
    for m in modes:
        results[m] = run_pipeline({**payload, "engine_mode": m}, vram, catalog)
        print_result(m, results[m])
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\nfull result written to {args.json}")


if __name__ == "__main__":
    main()
