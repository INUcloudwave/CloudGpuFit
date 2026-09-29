#!/usr/bin/env python3
"""Per-fusion-group Inductor latency predictor."""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

import predict_eager_runtime as eager_api
import run_eager_local as eager_local
from compiled_external_model import (
    CompiledExternalKernelRouter,
    CompiledFXTracer,
    propagate_graph_shapes,
)
from compiled_fusion_ir import CompiledFusionPlanner
from compiled_fusion_boundary_model import load_boundary_predictor
from compiled_fusion_model import CompiledFusionKernelRouter
from compiled_models import remove_unused_text_pooler
from runtime_manifest import DEFAULT_TARGET, validate_manifest


RUNTIME_ROOT = Path(__file__).resolve().parent


def load_compiled_context_calibration(
    target: str,
    runtime_root: Path = RUNTIME_ROOT,
) -> dict[str, Any] | None:
    path = runtime_root / "targets" / target / "compiled_context_calibration.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError(f"Unsupported Compiled context calibration: {path}")
    if payload.get("target_instance") != target:
        raise ValueError(f"Compiled context target mismatch: {path}")
    if payload.get("model_specific") is not False:
        raise ValueError("Compiled context calibration must be model-independent")
    required = (
        "kernel_scale",
        "external_launch_ms",
        "fusion_launch_ms",
        "fixed_overhead_ms",
    )
    for name in required:
        value = float(payload[name])
        if value < 0.0:
            raise ValueError(f"Compiled context value must be non-negative: {name}")
    return payload


def load_hardware_profile() -> dict[str, Any]:
    value = os.environ.get("EAGER_HARDWARE_PROFILE")
    if not value:
        raise RuntimeError("Target hardware profile is not configured")
    path = Path(value)
    return json.loads(path.read_text(encoding="utf-8"))


def pipeline_time(
    *,
    hardware: Mapping[str, Any],
    kernel_ms: float,
    external_launch_count: int,
    fusion_launch_count: int,
    input_bytes: int,
    output_bytes: int,
    context_calibration: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    h2d_ms = float(hardware["t_h2d_base"]) + input_bytes / (
        float(hardware["bw_h2d_GBs"]) * 1e6
    )
    d2h_ms = float(hardware["t_d2h_base"]) + output_bytes / (
        float(hardware["bw_d2h_GBs"]) * 1e6
    )
    graph_dispatch_ms = float(hardware["t_dispatch_heavy"])
    external_launch_count = max(int(external_launch_count), 0)
    fusion_launch_count = max(int(fusion_launch_count), 0)
    external_schedule_ms = (
        external_launch_count * float(hardware["t_dispatch_heavy"])
    )
    fusion_schedule_ms = fusion_launch_count * float(hardware["t_dispatch_light"])
    host_schedule_ms = external_schedule_ms + fusion_schedule_ms
    if context_calibration is None:
        kernel_scale = 1.0
        context_fixed_overhead_ms = 0.0
        scheduled_kernel_ms = max(float(kernel_ms), host_schedule_ms)
        context_calibration_applied = 0.0
    else:
        kernel_scale = float(context_calibration["kernel_scale"])
        external_schedule_ms = external_launch_count * float(
            context_calibration["external_launch_ms"]
        )
        fusion_schedule_ms = fusion_launch_count * float(
            context_calibration["fusion_launch_ms"]
        )
        host_schedule_ms = external_schedule_ms + fusion_schedule_ms
        context_fixed_overhead_ms = float(
            context_calibration["fixed_overhead_ms"]
        )
        scheduled_kernel_ms = (
            float(kernel_ms) * kernel_scale
            + host_schedule_ms
            + context_fixed_overhead_ms
        )
        context_calibration_applied = 1.0
    sync_ms = float(hardware["t_sync"])
    total_ms = (
        max(h2d_ms, graph_dispatch_ms)
        + scheduled_kernel_ms
        + d2h_ms
        + sync_ms
    )
    return {
        "h2d_ms": h2d_ms,
        "graph_dispatch_ms": graph_dispatch_ms,
        "kernel_sum_ms": kernel_ms,
        "kernel_launch_count": float(
            external_launch_count + fusion_launch_count
        ),
        "external_launch_count": float(external_launch_count),
        "fusion_launch_count": float(fusion_launch_count),
        "external_schedule_ms": external_schedule_ms,
        "fusion_schedule_ms": fusion_schedule_ms,
        "kernel_launch_scale": kernel_scale,
        "kernel_launch_ms": host_schedule_ms,
        "kernel_schedule_overlap_ms": min(float(kernel_ms), host_schedule_ms),
        "scheduled_kernel_ms": scheduled_kernel_ms,
        "context_fixed_overhead_ms": context_fixed_overhead_ms,
        "context_calibration_applied": context_calibration_applied,
        "d2h_ms": d2h_ms,
        "sync_ms": sync_ms,
        "e2e_ms": total_ms,
    }


def support_warnings(
    workload_type: str,
    normalized: Mapping[str, int],
    batch_size: int,
    original: list[str],
) -> list[str]:
    warnings = [
        value
        for value in original
        if "outside the measured range 64-256" not in value
    ]
    if batch_size not in {1, 8, 16}:
        warnings.append(
            "batch_size is outside the validated Compiled set {1, 8, 16}"
        )
    if workload_type == "text_encoder":
        sequence = int(normalized["seq_len"])
        if not 64 <= sequence <= 512:
            warnings.append(
                "text-encoder seq_len is outside the validated Compiled range 64-512"
            )
    elif workload_type == "decoder_llm":
        sequence = int(normalized["seq_len"])
        if not 64 <= sequence <= 1024:
            warnings.append(
                "decoder-only seq_len is outside the validated Compiled range 64-1024"
            )
    elif workload_type == "text_encdec":
        sequence = int(normalized["encoder_seq_len"])
        decoder = int(normalized["decoder_seq_len"])
        if not 64 <= sequence <= 1024:
            warnings.append(
                "encoder_seq_len is outside the validated Compiled range 64-1024"
            )
        if decoder != 1:
            warnings.append(
                "Compiled text encoder-decoder validation covers decoder_seq_len=1 only"
            )
    elif workload_type == "vision_cls" and int(normalized["resolution"]) != 224:
        warnings.append("Compiled vision validation covers resolution=224 only")
    elif workload_type == "audio_encoder" and int(normalized["audio_samples"]) != 32000:
        warnings.append("Compiled audio validation covers audio_samples=32000 only")
    return list(dict.fromkeys(warnings))


def _predict_or_raise(
    request: Mapping[str, Any],
    target: str,
    manifest_path: Path,
) -> dict[str, Any]:
    trace_name, workload_type, runtime_config, warnings, normalized = (
        eager_api.build_dynamic_model_config(request)
    )
    batch_size = int(normalized["batch_size"])
    manifest = validate_manifest(
        manifest_path,
        expected_target=target,
        check_gpu=False,
    )
    hardware = load_hardware_profile()
    context_calibration = load_compiled_context_calibration(target)
    eager_local.enable_unicode_safe_lightgbm_loading()
    tracer = CompiledFXTracer()
    external_router = CompiledExternalKernelRouter(
        runtime_root=RUNTIME_ROOT,
        target=target,
        hardware_profile=hardware,
    )
    fusion_router = CompiledFusionKernelRouter(
        runtime_root=RUNTIME_ROOT,
        target=target,
        hardware_profile=hardware,
    )
    boundary_predictor = load_boundary_predictor(RUNTIME_ROOT, target)

    load_fn = runtime_config["load_fn"]
    dummy_fn = runtime_config["dummy_fn"]
    bytes_fn = runtime_config["bytes_fn"]
    assert callable(load_fn) and callable(dummy_fn) and callable(bytes_fn)
    with torch.device(eager_local.ANALYSIS_DEVICE):
        model = load_fn().eval()
    if workload_type == "text_encoder":
        remove_unused_text_pooler(model)
    trace_input = dummy_fn(1)
    gm = eager_local.trace_model_compat(
        tracer,
        model,
        trace_input,
        bool(runtime_config["is_hf"]),
        trace_name,
    )
    gm = gm.half().to(eager_local.ANALYSIS_DEVICE)
    dummy_input = dummy_fn(batch_size)
    propagate_graph_shapes(gm, dummy_input)
    fusion_gm = gm
    external_median = external_router.predict_graph(gm, "median")
    external_p90 = external_router.predict_graph(gm, "p90")
    external_median_ms = sum(item["predicted_ms"] for item in external_median)
    external_p90_ms = sum(item["predicted_ms"] for item in external_p90)

    planner = CompiledFusionPlanner(
        l2_cache_bytes=int(hardware["l2_cache_bytes"]),
        sm_count=int(hardware["sm_count"]),
        precision_bytes=int(hardware["precision_bytes"]),
        boundary_predictor=boundary_predictor,
    )
    fusion_plan = planner.plan(fusion_gm)
    if fusion_plan.unsupported:
        details = ", ".join(
            sorted({item["target"] for item in fusion_plan.unsupported})
        )
        raise NotImplementedError(
            f"Compiled fusion coverage is incomplete for this FX graph "
            f"({len(fusion_plan.unsupported)} nodes): "
            + details
        )

    median_groups = []
    p90_groups = []
    for index, group in enumerate(fusion_plan.groups):
        median = fusion_router.predict_group(group, "median")
        p90 = fusion_router.predict_group(group, "p90")
        common = {
            "group_index": index,
            "kind": group["kind"],
            "operations": group["operations"],
            "node_names": group["node_names"],
            "features": group["features"],
        }
        median_groups.append({**common, **median})
        p90_groups.append({**common, **p90})

    fusion_median_raw_ms = sum(item["predicted_ms"] for item in median_groups)
    fusion_p90_ms = sum(item["predicted_ms"] for item in p90_groups)
    fusion_median_ms = fusion_median_raw_ms
    external_launch_count = len(external_median)
    fusion_launch_count = len(fusion_plan.groups)
    input_bytes, output_bytes = bytes_fn(batch_size, int(hardware["precision_bytes"]))
    median_pipeline = pipeline_time(
        hardware=hardware,
        kernel_ms=external_median_ms + fusion_median_ms,
        external_launch_count=external_launch_count,
        fusion_launch_count=fusion_launch_count,
        input_bytes=int(input_bytes),
        output_bytes=int(output_bytes),
        context_calibration=context_calibration,
    )
    p90_pipeline = pipeline_time(
        hardware=hardware,
        kernel_ms=external_p90_ms + fusion_p90_ms,
        external_launch_count=external_launch_count,
        fusion_launch_count=fusion_launch_count,
        input_bytes=int(input_bytes),
        output_bytes=int(output_bytes),
        context_calibration=context_calibration,
    )
    predicted_median_ms = median_pipeline["e2e_ms"]
    predicted_p90_ms = max(
        p90_pipeline["e2e_ms"], predicted_median_ms * 1.01
    )
    response = {
        "status": "ok",
        "model": str(request["model"]),
        "model_type": str(runtime_config["hf_model_type"]),
        "workload_type": workload_type,
        "target_instance": manifest["target_instance"],
        "target_gpu": manifest["gpu_name"],
        "engine_mode": "compiled",
        "compile_backend": "inductor",
        "compile_mode": "default",
        "fullgraph": True,
        "dynamic_shapes": False,
        "compile_time_included": False,
        "measurement_scope": "steady_state_e2e_h2d_forward_d2h_sync",
        "prediction_method": (
            "symbolic_fx_semantic_fusion_roofline_lightgbm_overlap_scheduler"
        ),
        "dtype": "fp16",
        "batch_size": batch_size,
        "params": {
            key: value for key, value in normalized.items() if key != "batch_size"
        },
        "predicted_median_ms": predicted_median_ms,
        "predicted_p90_ms": predicted_p90_ms,
        "predicted_throughput_rps": batch_size / (predicted_median_ms / 1000.0),
        "warnings": support_warnings(
            workload_type, normalized, batch_size, warnings
        ),
        "breakdown": {
            "external_kernel_count": int(len(external_median)),
            "external_kernel_median_raw_ms": external_median_ms,
            "external_kernel_median_ms": external_median_ms,
            "external_kernel_p90_ms": external_p90_ms,
            "external_family_counts": {
                family: sum(item["family"] == family for item in external_median)
                for family in ("gemm", "conv", "sdpa")
            },
            "external_family_median_ms": {
                family: sum(
                    item["predicted_ms"]
                    for item in external_median
                    if item["family"] == family
                )
                for family in ("gemm", "conv", "sdpa")
            },
            "analytical_fusion_group_count": len(fusion_plan.groups),
            "fusion_boundary_source": (
                "fx_edge_lightgbm"
                if boundary_predictor is not None
                else "symbolic_fx_semantic_analytical"
            ),
            "fusion_group_median_raw_ms": fusion_median_raw_ms,
            "fusion_group_median_ms": fusion_median_ms,
            "fusion_group_p90_ms": fusion_p90_ms,
            "median_pipeline": median_pipeline,
            "p90_pipeline": p90_pipeline,
            "median_external_kernels": external_median,
            "p90_external_kernels": external_p90,
            "median_fusion_groups": median_groups,
            "p90_fusion_groups": p90_groups,
        },
    }
    del model, gm, fusion_gm, trace_input, dummy_input
    gc.collect()
    return response


def predict_runtime(
    request: Mapping[str, Any],
    target: str = DEFAULT_TARGET,
    *,
    manifest_path: str | Path | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        error = TypeError("request must be a mapping")
        if raise_on_error:
            raise error
        return {
            "status": "error",
            "target_instance": target,
            "engine_mode": "compiled",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    selected_manifest = eager_api.resolve_manifest_path(target, manifest_path)
    try:
        with eager_api.runtime_context(selected_manifest):
            return _predict_or_raise(request, target, selected_manifest)
    except Exception as exc:
        if raise_on_error:
            raise
        return {
            "status": "error",
            "model": str(request.get("model", "")),
            "target_instance": target,
            "engine_mode": "compiled",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def predict_runtime_targets(
    request: Mapping[str, Any],
    targets: Iterable[str],
    *,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    predictions = [
        predict_runtime(request, target, raise_on_error=raise_on_error)
        for target in dict.fromkeys(str(value) for value in targets)
    ]
    return {
        "status": "ok" if all(item["status"] == "ok" for item in predictions) else "error",
        "model": str(request.get("model", "")),
        "predictions": predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = predict_runtime(request, target=args.target)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            default=str,
        )
    )
    raise SystemExit(0 if result.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
