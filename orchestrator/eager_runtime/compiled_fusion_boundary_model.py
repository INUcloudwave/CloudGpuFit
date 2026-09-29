#!/usr/bin/env python3
"""Model-independent FX-edge fusion boundary prediction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import lightgbm as lgb
import pandas as pd

from compiled_fusion_ir import FUSION_EDGE_FEATURE_NAMES


SCHEMA_VERSION = 1
PREDICTOR_VERSION = "compiled_fusion_boundary_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CompiledFusionBoundaryPredictor:
    def __init__(self, model_path: Path, threshold: float):
        self.model_path = model_path
        self.threshold = float(threshold)
        self.booster = lgb.Booster(model_file=str(model_path))

    def probability(self, features: Mapping[str, float]) -> float:
        frame = pd.DataFrame(
            [[float(features[name]) for name in FUSION_EDGE_FEATURE_NAMES]],
            columns=FUSION_EDGE_FEATURE_NAMES,
        )
        return float(self.booster.predict(frame)[0])

    def should_fuse(self, features: dict[str, float]) -> bool:
        return self.probability(features) >= self.threshold


def load_boundary_predictor(
    runtime_root: Path,
    target: str,
    *,
    required: bool = False,
) -> CompiledFusionBoundaryPredictor | None:
    root = runtime_root / "targets" / target / "compiled_fusion_boundary_model"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        if required:
            raise FileNotFoundError(manifest_path)
        return None
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported fusion-boundary schema: {manifest_path}")
    if payload.get("predictor_version") != PREDICTOR_VERSION:
        raise ValueError(f"Unsupported fusion-boundary predictor: {manifest_path}")
    if payload.get("target_instance") != target:
        raise ValueError(f"Fusion-boundary target mismatch: {manifest_path}")
    if payload.get("model_name_is_feature") is not False:
        raise ValueError("Model name must not be a fusion-boundary feature")
    if payload.get("feature_schema") != list(FUSION_EDGE_FEATURE_NAMES):
        raise ValueError("Fusion-boundary feature schema mismatch")
    artifact = payload["artifact"]
    model_path = root / artifact["path"]
    if model_path.stat().st_size != int(artifact["size_bytes"]):
        raise RuntimeError(f"Fusion-boundary artifact size mismatch: {model_path}")
    if sha256(model_path) != artifact["sha256"]:
        raise RuntimeError(f"Fusion-boundary artifact hash mismatch: {model_path}")
    return CompiledFusionBoundaryPredictor(
        model_path,
        threshold=float(payload["decision_threshold"]),
    )
