#!/usr/bin/env python3
"""Copy of the checkpoints' validate_eager_predictor_e2e.py (g7e/gr6 version).
Changes: sys.path bootstrap and --runtime-root default. Prediction logic is untouched."""

from __future__ import annotations

import argparse
import json
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

import compiled_models
import fxgraph_conv1d_v10 as v10
import fxgraph_eager_final as final
import predict_eager_runtime as eager
import run_eager_local as local
from runtime_manifest import manifest_for_target, validate_manifest


MODEL_TYPES = {
    "bert": "text_encoder",
    "roberta": "text_encoder",
    "opt": "decoder_llm",
    "t5": "text_encdec",
    "wav2vec2": "audio_encoder",
    "qwen2": "decoder_llm",
}

VISION_CONFIGS = {
    "resnet18": "ResNet-18",
    "mobilenet_v2": "MobileNetV2",
}


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


def grouped_metrics(frame: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    return {
        str(name): metrics(group.reset_index(drop=True))
        for name, group in frame.groupby(column, dropna=False)
    }


def dynamic_payload(model: str, row: pd.Series) -> dict[str, Any]:
    config = compiled_models.build_transformer_config(model)
    params: dict[str, int | str] = {"mode": "prefill"}
    if model == "t5":
        params.update(
            encoder_seq_len=int(row["sequence_length"]),
            decoder_seq_len=1,
        )
    elif model == "wav2vec2":
        params["audio_samples"] = int(row["audio_samples"])
    else:
        params["seq_len"] = int(row["sequence_length"])
    return {
        "model": str(row["model_id"]),
        "model_config": config.to_dict(),
        "type": MODEL_TYPES[model],
        "dtype": "fp16",
        "batch_size": int(row["batch_size"]),
        "params": params,
    }


def prediction_rows(actual: pd.DataFrame, target: str, root: Path) -> list[dict[str, Any]]:
    manifest_path = manifest_for_target(root, target)
    validate_manifest(manifest_path, expected_target=target, check_gpu=False)
    local.enable_unicode_safe_lightgbm_loading()
    parser = v10.Conv1dContextFXGraphParser()
    router = final.CalibratedEagerSimulatorRouter(engine_mode="eager")
    output: list[dict[str, Any]] = []

    configs = local.build_model_configs()
    for model, local_name in VISION_CONFIGS.items():
        selected = actual.loc[actual["model"] == model].copy()
        batches = sorted(selected["batch_size"].astype(int).unique().tolist())
        rows = local.predict_model(
            model_name=local_name,
            config=configs[local_name],
            batch_sizes=batches,
            parser=parser,
            router=router,
            construction_device="meta",
        )
        by_batch = {int(item["batch_size"]): item for item in rows}
        for _, item in selected.iterrows():
            prediction = by_batch[int(item["batch_size"])]
            output.append(
                {
                    "model": model,
                    "batch_size": int(item["batch_size"]),
                    "sequence_length": 0,
                    "predicted_median_ms": float(prediction["predicted_e2e_ms"]),
                    "predicted_p90_ms": float(prediction["predicted_p90_ms"]),
                }
            )

    nonvision = actual.loc[~actual["model"].isin(VISION_CONFIGS)].copy()
    grouping = ["model", "sequence_length", "audio_samples"]
    for _, group in nonvision.groupby(grouping, dropna=False):
        sample = group.iloc[0]
        payload = dynamic_payload(str(sample["model"]), sample)
        trace_name, workload, runtime_config, _, normalized = eager.build_dynamic_model_config(payload)
        batches = sorted(group["batch_size"].astype(int).unique().tolist())
        rows = local.predict_model(
            model_name=trace_name,
            config=runtime_config,
            batch_sizes=batches,
            parser=parser,
            router=router,
            construction_device="meta",
        )
        by_batch = {int(item["batch_size"]): item for item in rows}
        for _, item in group.iterrows():
            prediction = by_batch[int(item["batch_size"])]
            raw_median = float(prediction["predicted_e2e_ms"])
            calibrated, _ = eager.apply_first_token_calibration(
                raw_median,
                workload_type=workload,
                normalized={**normalized, "batch_size": int(item["batch_size"])},
                breakdown=prediction,
            )
            output.append(
                {
                    "model": str(item["model"]),
                    "batch_size": int(item["batch_size"]),
                    "sequence_length": int(item["sequence_length"] or 0),
                    "predicted_median_ms": calibrated,
                    "predicted_p90_ms": float(prediction["predicted_p90_ms"]),
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=_RUNTIME_ROOT)
    parser.add_argument("--target", default="g7e.2xlarge")
    parser.add_argument(
        "--actual",
        type=Path,
        default=Path("results/g7e_eager_e2e_actual/eager_e2e_actual.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/g7e_eager_predictor_e2e.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/g7e_eager_predictor_e2e_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.runtime_root.resolve()
    actual = pd.read_csv(args.actual)
    actual = actual.loc[actual["status"] == "ok"].copy()
    actual["sequence_length"] = actual["sequence_length"].fillna(0).astype(int)
    predictions = pd.DataFrame(prediction_rows(actual, args.target, root))
    merged = actual.merge(
        predictions,
        on=["model", "batch_size", "sequence_length"],
        how="left",
        validate="one_to_one",
    )
    if merged["predicted_median_ms"].isna().any():
        missing = merged.loc[merged["predicted_median_ms"].isna(), "model"].tolist()
        raise RuntimeError(f"Missing Eager predictions: {missing}")
    merged["ape_percent"] = (
        (merged["predicted_median_ms"] - merged["actual_median_ms"]).abs()
        / merged["actual_median_ms"].clip(lower=1e-9)
        * 100.0
    )
    merged["bias_percent"] = (
        (merged["predicted_median_ms"] - merged["actual_median_ms"])
        / merged["actual_median_ms"].clip(lower=1e-9)
        * 100.0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    summary = {
        "target_instance": args.target,
        "total_rows": int(len(merged)),
        "overall": metrics(merged),
        "by_validation_set": grouped_metrics(merged, "validation_set"),
        "by_workload_family": grouped_metrics(merged, "workload_family"),
        "by_model": grouped_metrics(merged, "model"),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
