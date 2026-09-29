#!/usr/bin/env python3
"""Final target-aware Eager predictor with calibrated end-to-end P90."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import fxgraph_conv1d_v10 as v10
import fxgraph_host_v9 as v9
import fxgraph_pointwise_v7 as v7
import fxgraph_reduction_v2 as base
from runtime_manifest import DEFAULT_TARGET, manifest_for_target, validate_manifest


RUNTIME_ROOT = Path(__file__).resolve().parent
DEFAULT_FACTOR_PATH = (
    RUNTIME_ROOT
    / "targets"
    / DEFAULT_TARGET
    / "p90_calibration_factor.json"
)
P90_MEDIAN_FLOOR_MULTIPLIER = 1.10


def load_calibration(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            f"P90 calibration file not found: {path}. "
            "Run calibrate_eager_p90.py first or set EAGER_P90_CALIBRATION_FILE."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "eager":
        raise ValueError(f"Expected mode='eager' in {path}")
    method = payload.get("method")
    factor = float(payload.get("factor", float("nan")))
    # "uncalibrated_identity" is how a target without Eager E2E P90 data yet
    # (e.g. g7.2xlarge) says "no P90 correction": allowed only as an exact
    # identity factor, so it can never scale anything. Median is unaffected.
    if method == "uncalibrated_identity":
        if factor != 1.0:
            raise ValueError(
                f"uncalibrated_identity requires factor == 1.0 in {path}: {factor}"
            )
    elif method != "global_conformal_ratio":
        raise ValueError(f"Unsupported calibration method in {path}")
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError(f"Calibration factor must be finite and >= 1.0: {factor}")
    target = float(payload.get("target_coverage", float("nan")))
    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError(f"Invalid target coverage: {target}")
    return payload


def calibrate_p90(
    median_e2e_ms: float,
    raw_model_p90_ms: float,
    factor: float,
) -> tuple[float, float]:
    values = {
        "median_e2e_ms": median_e2e_ms,
        "raw_model_p90_ms": raw_model_p90_ms,
        "factor": factor,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive: {value}")
    if factor < 1.0:
        raise ValueError(f"factor must be >= 1.0: {factor}")

    existing_p90_ms = max(
        median_e2e_ms * P90_MEDIAN_FLOOR_MULTIPLIER,
        raw_model_p90_ms,
    )
    return existing_p90_ms * factor, existing_p90_ms


class CalibratedEagerSimulatorRouter(v10.Conv1dContextSimulatorRouter):
    def __init__(self, engine_mode="eager"):
        if engine_mode != "eager":
            raise ValueError("P90 calibration is valid only for Eager mode")
        super().__init__(engine_mode=engine_mode)
        factor_path = Path(
            os.environ.get("EAGER_P90_CALIBRATION_FILE", str(DEFAULT_FACTOR_PATH))
        )
        self.p90_calibration = load_calibration(factor_path)
        self.p90_calibration_factor = float(self.p90_calibration["factor"])
        self.p90_calibration_path = factor_path
        print(
            "Eager E2E P90 calibration active: "
            f"factor={self.p90_calibration_factor:.6f}, file={factor_path}"
        )

    def finalize_p90_prediction(
        self,
        median_e2e_ms: float,
        raw_model_p90_ms: float,
    ) -> float:
        calibrated_p90_ms, _ = calibrate_p90(
            median_e2e_ms=median_e2e_ms,
            raw_model_p90_ms=raw_model_p90_ms,
            factor=self.p90_calibration_factor,
        )
        return calibrated_p90_ms


def build_prediction_breakdown(*args, **kwargs):
    record = v9.build_prediction_breakdown(*args, **kwargs)
    router = kwargs.get("router")
    if router is None and args:
        router = args[0]
    record["p90_calibration_method"] = "global_conformal_ratio"
    record["p90_calibration_factor"] = router.p90_calibration_factor
    record["p90_median_floor_multiplier"] = P90_MEDIAN_FLOOR_MULTIPLIER
    record["p90_target_coverage"] = float(
        router.p90_calibration["target_coverage"]
    )
    return record


def main():
    target = os.environ.get("EAGER_TARGET_INSTANCE", DEFAULT_TARGET)
    default_manifest = manifest_for_target(RUNTIME_ROOT, target)
    manifest_path = Path(
        os.environ.get("EAGER_RUNTIME_MANIFEST", str(default_manifest))
    )
    manifest = validate_manifest(
        manifest_path,
        expected_target=target,
    )
    os.environ["EAGER_P90_CALIBRATION_ACTIVE"] = "1"
    print(
        "Eager runtime manifest valid: "
        f"instance={manifest['target_instance']}, "
        f"predictor={manifest['predictor_version']}, "
        f"GPU={manifest['gpu_name']}, "
        f"artifacts={len(manifest['artifacts'])}"
    )
    base.FXGraphParser = v10.Conv1dContextFXGraphParser
    base.SimulatorRouter = CalibratedEagerSimulatorRouter
    base.prediction_family = v7.prediction_family
    base.build_prediction_breakdown = build_prediction_breakdown
    base.breakdown_output_path = os.environ.get(
        "EAGER_FINAL_OUTPUT",
        "fx_breakdown_eager_final.csv",
    )
    print("Target-aware Eager predictor active")
    base.run_all_simulations(target_mode="eager")


if __name__ == "__main__":
    main()
