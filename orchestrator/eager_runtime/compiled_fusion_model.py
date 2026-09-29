#!/usr/bin/env python3
"""Per-fusion-group LightGBM runtime model and artifact validation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from compiled_fusion_ir import FUSION_FEATURE_NAMES, theoretical_time_ms


SCHEMA_VERSION = 2
PREDICTOR_VERSION = "compiled_fusion_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fusion_model_root(runtime_root: Path, target: str) -> Path:
    return runtime_root / "targets" / target / "compiled_fusion_model"


def load_fusion_manifest(runtime_root: Path, target: str) -> dict[str, Any]:
    root = fusion_model_root(runtime_root, target)
    path = root / "compiled_fusion_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Compiled fusion manifest is missing: {path}. "
            "Collect and train per-fusion-group models first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported Compiled fusion schema: {path}")
    if payload.get("predictor_version") != PREDICTOR_VERSION:
        raise ValueError(f"Unsupported Compiled fusion predictor: {path}")
    if payload.get("target_instance") != target:
        raise ValueError(
            f"Compiled fusion target mismatch: requested={target}, "
            f"artifact={payload.get('target_instance')}"
        )
    if payload.get("feature_schema") != list(FUSION_FEATURE_NAMES):
        raise ValueError("Compiled fusion feature schema mismatch")
    for artifact in payload.get("artifacts", {}).values():
        artifact_path = root / artifact["path"]
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if artifact_path.stat().st_size != int(artifact["size_bytes"]):
            raise RuntimeError(f"Compiled fusion artifact size mismatch: {artifact_path}")
        if sha256(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"Compiled fusion artifact hash mismatch: {artifact_path}")
    return payload


def _load_booster(path: Path) -> lgb.Booster:
    return lgb.Booster(model_str=path.read_text(encoding="utf-8"))


class CompiledFusionKernelRouter:
    def __init__(
        self,
        *,
        runtime_root: Path,
        target: str,
        hardware_profile: dict[str, Any],
    ):
        self.root = fusion_model_root(runtime_root, target)
        self.manifest = load_fusion_manifest(runtime_root, target)
        self.peak_flops = float(hardware_profile["peak_flops"])
        self.peak_bandwidth = float(hardware_profile["peak_bandwidth"])
        self.kernel_floor_ms = float(self.manifest["kernel_floor"]["value_ms"])
        self.median = _load_booster(
            self.root / self.manifest["artifacts"]["median_model"]["path"]
        )
        self.p90 = _load_booster(
            self.root / self.manifest["artifacts"]["p90_model"]["path"]
        )
        expected = list(FUSION_FEATURE_NAMES)
        if self.median.feature_name() != expected or self.p90.feature_name() != expected:
            raise ValueError("Compiled fusion LightGBM feature order mismatch")

    def predict_group(
        self,
        group: dict[str, Any],
        prediction_mode: str,
    ) -> dict[str, float]:
        if prediction_mode not in {"median", "p90"}:
            raise ValueError(f"Unknown prediction mode: {prediction_mode}")
        features = group["features"]
        frame = pd.DataFrame([features], columns=FUSION_FEATURE_NAMES)
        if frame.isna().any().any():
            missing = frame.columns[frame.isna().any()].tolist()
            raise ValueError("Missing Compiled fusion features: " + ", ".join(missing))
        booster = self.median if prediction_mode == "median" else self.p90
        ratio = max(float(np.expm1(booster.predict(frame)[0])), 0.0)
        theoretical_ms = theoretical_time_ms(
            features,
            peak_flops=self.peak_flops,
            peak_bandwidth=self.peak_bandwidth,
            kernel_floor_ms=self.kernel_floor_ms,
        )
        conformal_factor = (
            float(self.manifest["p90_calibration"]["factor"])
            if prediction_mode == "p90"
            else 1.0
        )
        predicted_ms = theoretical_ms * ratio * conformal_factor
        if not math.isfinite(predicted_ms) or predicted_ms <= 0.0:
            raise RuntimeError(
                f"Invalid Compiled fusion prediction: {predicted_ms}"
            )
        return {
            "theoretical_ms": theoretical_ms,
            "ratio": ratio,
            "conformal_factor": conformal_factor,
            "predicted_ms": predicted_ms,
        }
