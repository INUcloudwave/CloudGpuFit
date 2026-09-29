#!/usr/bin/env python3
"""Copy of the checkpoints' validate_compiled_predictor_e2e.py (g7/g7e/gr6 version,
with Compiled context calibration). Changes: sys.path bootstrap, --runtime-root
default, and --validation-data override. Prediction logic is untouched."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

# Regression copy: run from anywhere against the eager_runtime package that
# contains this regression/ folder (sibling modules use bare imports).
import sys as _sys
_RUNTIME_ROOT = Path(__file__).resolve().parent.parent
if str(_RUNTIME_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_RUNTIME_ROOT))

import numpy as np
import pandas as pd
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
from compiled_models import (
    AUDIO_MODELS,
    VISION_MODELS,
    build_model,
    build_transformer_config,
    make_cpu_inputs,
    remove_unused_text_pooler,
)
from predict_compiled_runtime import load_compiled_context_calibration, pipeline_time
from runtime_manifest import validate_manifest


DATASET_FILES = (
    "compiled_core.csv",
    "compiled_longseq.csv",
    "compiled_encoder_seq512.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=_RUNTIME_ROOT)
    parser.add_argument("--validation-data", type=Path, default=None)
    parser.add_argument("--target", default="g6.xlarge")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--one-per-model", action="store_true")
    return parser.parse_args()


def optional_int(value: Any, default: int) -> int:
    if value is None or pd.isna(value):
        return default
    return int(value)


def to_analysis_device(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.empty_like(value, device=eager_local.ANALYSIS_DEVICE)
    if isinstance(value, dict):
        return {key: to_analysis_device(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_analysis_device(item) for item in value)
    if isinstance(value, list):
        return [to_analysis_device(item) for item in value]
    return value


def tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def output_bytes(model_name: str, row: pd.Series) -> int:
    batch = int(row["batch_size"])
    precision = 2
    sequence = optional_int(row.get("sequence_length"), 128)
    if model_name in VISION_MODELS:
        return batch * 1000 * precision
    config = build_transformer_config(model_name)
    if model_name in {"bert", "roberta"}:
        return batch * sequence * int(config.hidden_size) * precision
    if model_name in {"opt", "qwen2", "t5"}:
        return batch * int(config.vocab_size) * precision
    if model_name in AUDIO_MODELS:
        length = optional_int(row.get("audio_samples"), 32000)
        for kernel, stride in zip(config.conv_kernel, config.conv_stride):
            length = (length - int(kernel)) // int(stride) + 1
        return batch * max(length, 1) * int(config.hidden_size) * precision
    raise ValueError(f"Unknown model: {model_name}")


def predict_case(
    row: pd.Series,
    *,
    tracer: CompiledFXTracer,
    external_router: CompiledExternalKernelRouter,
    fusion_router: CompiledFusionKernelRouter,
    hardware: dict[str, Any],
    context_calibration: dict[str, Any] | None = None,
    boundary_predictor=None,
) -> dict[str, Any]:
    model_name = str(row["model"])
    batch = int(row["batch_size"])
    sequence = optional_int(row.get("sequence_length"), 128)
    resolution = optional_int(row.get("resolution"), 224)
    audio_samples = optional_int(row.get("audio_samples"), 32000)

    wrapped_model = build_model(
        model_name, str(eager_local.ANALYSIS_DEVICE)
    ).eval()
    model = (
        wrapped_model.model
        if model_name not in VISION_MODELS and hasattr(wrapped_model, "model")
        else wrapped_model
    )
    if model_name in {"bert", "roberta"}:
        remove_unused_text_pooler(model)
    dummy_input = to_analysis_device(
        make_cpu_inputs(
            model_name,
            batch,
            sequence,
            resolution=resolution,
            audio_samples=audio_samples,
        )
    )
    trace_name = "Wav2Vec2" if model_name == "wav2vec2" else model_name
    gm = eager_local.trace_model_compat(
        tracer,
        model,
        dummy_input,
        model_name not in VISION_MODELS,
        trace_name,
    )
    gm = gm.half().to(eager_local.ANALYSIS_DEVICE)
    propagate_graph_shapes(gm, dummy_input)
    fusion_gm = gm
    external_median = external_router.predict_graph(gm, "median")
    external_p90 = external_router.predict_graph(gm, "p90")

    planner = CompiledFusionPlanner(
        l2_cache_bytes=int(hardware["l2_cache_bytes"]),
        sm_count=int(hardware["sm_count"]),
        precision_bytes=int(hardware["precision_bytes"]),
        boundary_predictor=boundary_predictor,
    )
    plan = planner.plan(fusion_gm)
    if plan.unsupported:
        targets = sorted({item["target"] for item in plan.unsupported})
        raise NotImplementedError(
            f"Incomplete fusion coverage ({len(plan.unsupported)} nodes): "
            + ", ".join(targets)
        )

    median_groups = [
        fusion_router.predict_group(group, "median") for group in plan.groups
    ]
    p90_groups = [fusion_router.predict_group(group, "p90") for group in plan.groups]
    external_median_ms = sum(item["predicted_ms"] for item in external_median)
    external_p90_ms = sum(item["predicted_ms"] for item in external_p90)
    fusion_median_raw_ms = sum(item["predicted_ms"] for item in median_groups)
    fusion_p90_ms = sum(item["predicted_ms"] for item in p90_groups)
    fusion_median_ms = fusion_median_raw_ms
    external_launch_count = len(external_median)
    fusion_launch_count = len(plan.groups)
    fusion_kind_counts = {
        kind: sum(group["kind"] == kind for group in plan.groups)
        for kind in ("pointwise", "reduction", "memory")
    }
    median_pipeline = pipeline_time(
        hardware=hardware,
        kernel_ms=external_median_ms + fusion_median_ms,
        external_launch_count=external_launch_count,
        fusion_launch_count=fusion_launch_count,
        input_bytes=tensor_bytes(dummy_input),
        output_bytes=output_bytes(model_name, row),
        context_calibration=context_calibration,
    )
    p90_pipeline = pipeline_time(
        hardware=hardware,
        kernel_ms=external_p90_ms + fusion_p90_ms,
        external_launch_count=external_launch_count,
        fusion_launch_count=fusion_launch_count,
        input_bytes=tensor_bytes(dummy_input),
        output_bytes=output_bytes(model_name, row),
        context_calibration=context_calibration,
    )
    result = {
        "predicted_median_ms": median_pipeline["e2e_ms"],
        "predicted_p90_ms": max(
            p90_pipeline["e2e_ms"], median_pipeline["e2e_ms"] * 1.01
        ),
        "external_kernel_count": int(len(external_median)),
        "external_gemm_count": sum(
            item["family"] == "gemm" for item in external_median
        ),
        "external_conv_count": sum(
            item["family"] == "conv" for item in external_median
        ),
        "external_sdpa_count": sum(
            item["family"] == "sdpa" for item in external_median
        ),
        "analytical_fusion_group_count": int(len(plan.groups)),
        "fusion_boundary_source": (
            "fx_edge_lightgbm"
            if boundary_predictor is not None
            else "symbolic_fx_semantic_analytical"
        ),
        "fusion_pointwise_group_count": fusion_kind_counts["pointwise"],
        "fusion_reduction_group_count": fusion_kind_counts["reduction"],
        "fusion_memory_group_count": fusion_kind_counts["memory"],
        "fusion_single_node_group_count": sum(
            len(group["node_names"]) == 1 for group in plan.groups
        ),
        "fusion_operation_count": sum(
            len(group["node_names"]) for group in plan.groups
        ),
        "fusion_max_operations_per_group": max(
            (len(group["node_names"]) for group in plan.groups),
            default=0,
        ),
        "external_kernel_median_raw_ms": external_median_ms,
        "external_kernel_median_ms": external_median_ms,
        "fusion_group_median_raw_ms": fusion_median_raw_ms,
        "fusion_group_median_ms": fusion_median_ms,
        "predicted_kernel_sum_ms": median_pipeline["kernel_sum_ms"],
        "predicted_kernel_launch_count": int(
            median_pipeline["kernel_launch_count"]
        ),
        "predicted_external_launch_count": int(
            median_pipeline["external_launch_count"]
        ),
        "predicted_fusion_launch_count": int(
            median_pipeline["fusion_launch_count"]
        ),
        "predicted_external_schedule_ms": median_pipeline[
            "external_schedule_ms"
        ],
        "predicted_fusion_schedule_ms": median_pipeline[
            "fusion_schedule_ms"
        ],
        "predicted_kernel_launch_ms": median_pipeline["kernel_launch_ms"],
        "predicted_kernel_launch_scale": median_pipeline["kernel_launch_scale"],
        "predicted_kernel_schedule_overlap_ms": median_pipeline[
            "kernel_schedule_overlap_ms"
        ],
        "predicted_scheduled_kernel_ms": median_pipeline["scheduled_kernel_ms"],
        "predicted_h2d_ms": median_pipeline["h2d_ms"],
        "predicted_d2h_ms": median_pipeline["d2h_ms"],
        "predicted_sync_ms": median_pipeline["sync_ms"],
        "predicted_context_fixed_overhead_ms": median_pipeline[
            "context_fixed_overhead_ms"
        ],
        "predicted_context_calibration_applied": int(
            median_pipeline["context_calibration_applied"]
        ),
    }
    del wrapped_model, model, gm, fusion_gm, dummy_input
    gc.collect()
    return result


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    actual = frame["actual_median_ms"].to_numpy(dtype=float)
    predicted = frame["predicted_median_ms"].to_numpy(dtype=float)
    error = predicted - actual
    ape = np.abs(error) / np.maximum(actual, 1e-9) * 100.0
    return {
        "rows": int(len(frame)),
        "mape_percent": float(np.mean(ape)),
        "median_ape_percent": float(np.median(ape)),
        "p90_ape_percent": float(np.quantile(ape, 0.90, method="higher")),
        "max_ape_percent": float(np.max(ape)),
        "mean_bias_percent": float(
            np.mean(error / np.maximum(actual, 1e-9) * 100.0)
        ),
        "rmse_ms": float(np.sqrt(np.mean(error**2))),
        "p90_coverage": float(
            np.mean(
                frame["predicted_p90_ms"].to_numpy(dtype=float)
                >= frame["actual_p90_ms"].to_numpy(dtype=float)
            )
        ),
    }


def grouped_metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    return {
        str(name): metrics(group)
        for name, group in frame.groupby(column, sort=True)
    }


def write_checkpoint(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    args = parse_args()
    root = args.runtime_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    manifest_path = root / "targets" / args.target / "runtime_manifest.json"

    shared_data_root = root / "compiled_validation_data"
    target_data_root = root / "targets" / args.target / "compiled_validation_data"
    if args.validation_data is not None:
        target_data_root = args.validation_data.expanduser().resolve()
    if target_data_root.is_dir():
        data_root = target_data_root
        validation_data_scope = "target_specific"
        dataset_files = [
            filename for filename in DATASET_FILES if (data_root / filename).is_file()
        ]
        if not dataset_files:
            raise RuntimeError(f"No target validation datasets found: {data_root}")
    else:
        data_root = shared_data_root
        validation_data_scope = "shared_legacy"
        dataset_files = list(DATASET_FILES)

    sources = []
    for filename in dataset_files:
        path = data_root / filename
        frame = pd.read_csv(path)
        frame.insert(0, "validation_set", path.stem)
        sources.append(frame)
    actuals = pd.concat(sources, ignore_index=True)
    if args.models:
        unknown = sorted(set(args.models) - set(actuals["model"].unique()))
        if unknown:
            raise ValueError("Unknown validation models: " + ", ".join(unknown))
        actuals = actuals.loc[actuals["model"].isin(args.models)].copy()
    if args.one_per_model:
        actuals = actuals.groupby("model", sort=False, as_index=False).head(1)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        actuals = actuals.head(args.limit).copy()

    rows: list[dict[str, Any]] = []
    with eager_api.runtime_context(manifest_path):
        validate_manifest(manifest_path, expected_target=args.target, check_gpu=False)
        hardware = json.loads(
            Path(os.environ["EAGER_HARDWARE_PROFILE"]).read_text(encoding="utf-8")
        )
        context_calibration = load_compiled_context_calibration(args.target, root)
        eager_local.enable_unicode_safe_lightgbm_loading()
        tracer = CompiledFXTracer()
        external_router = CompiledExternalKernelRouter(
            runtime_root=root,
            target=args.target,
            hardware_profile=hardware,
        )
        fusion_router = CompiledFusionKernelRouter(
            runtime_root=root,
            target=args.target,
            hardware_profile=hardware,
        )
        boundary_predictor = load_boundary_predictor(root, args.target)
        for index, actual in actuals.iterrows():
            print(
                f"[{index + 1}/{len(actuals)}] {actual['validation_set']} "
                f"{actual['model']} b={int(actual['batch_size'])} "
                f"s={optional_int(actual.get('sequence_length'), 0)}"
            )
            row: dict[str, Any] = {
                "status": "ok",
                "validation_set": actual["validation_set"],
                "model": actual["model"],
                "model_id": actual["model_id"],
                "workload_family": actual["workload_family"],
                "batch_size": int(actual["batch_size"]),
                "sequence_length": optional_int(actual.get("sequence_length"), 0),
                "decoder_sequence_length": optional_int(
                    actual.get("decoder_sequence_length"), 0
                ),
                "resolution": optional_int(actual.get("resolution"), 0),
                "audio_samples": optional_int(actual.get("audio_samples"), 0),
                "generated_kernel_count": optional_int(
                    actual.get("generated_kernel_count"), 0
                ),
                "actual_median_ms": float(actual["actual_median_ms"]),
                "actual_p90_ms": float(actual["actual_p90_ms"]),
            }
            try:
                prediction = predict_case(
                    actual,
                    tracer=tracer,
                    external_router=external_router,
                    fusion_router=fusion_router,
                    hardware=hardware,
                    context_calibration=context_calibration,
                    boundary_predictor=boundary_predictor,
                )
                row.update(prediction)
                row["median_ape_percent"] = (
                    abs(row["predicted_median_ms"] - row["actual_median_ms"])
                    / max(row["actual_median_ms"], 1e-9)
                    * 100.0
                )
            except Exception as exc:
                row.update(
                    status="error",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            rows.append(row)
            write_checkpoint(rows, output)

    result = pd.DataFrame(rows)
    valid = result.loc[result["status"] == "ok"].copy()
    failed = result.loc[result["status"] != "ok"]
    summary: dict[str, Any] = {
        "target_instance": args.target,
        "validation_data_scope": validation_data_scope,
        "validation_data_root": str(data_root),
        "validation_datasets": dataset_files,
        "total_rows": int(len(result)),
        "successful_rows": int(len(valid)),
        "failed_rows": int(len(failed)),
        "overall": metrics(valid) if not valid.empty else {},
        "by_validation_set": grouped_metrics(valid, "validation_set"),
        "by_workload_family": grouped_metrics(valid, "workload_family"),
        "by_model": grouped_metrics(valid, "model"),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(1 if not failed.empty else 0)


if __name__ == "__main__":
    main()
