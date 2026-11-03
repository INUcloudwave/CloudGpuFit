"""Pure memory accounting. GPU overhead is calibrated, never inferred from CPU RSS."""
from __future__ import annotations

import json
import math
from pathlib import Path

VERSION = "3-native-inference"
PROFILE_PATH = Path(__file__).with_name("gpu_overheads.json")
ALIASES = {"Tesla T4": "NVIDIA T4", "T4": "NVIDIA T4", "A10G": "NVIDIA A10G",
           "L4": "NVIDIA L4", "L40S": "NVIDIA L40S"}


def nonnegative_bytes(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative byte count")
    return math.ceil(value)


def process_estimate(allocated, reserved, gpu_name, dtype, profiles=None):
    allocated = nonnegative_bytes(allocated, "allocated")
    reserved = nonnegative_bytes(reserved, "reserved")
    if reserved < allocated:
        raise ValueError("Reserved estimate is smaller than allocated estimate")
    name = ALIASES.get(gpu_name, gpu_name)
    if profiles is None:
        profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    profile = profiles["gpus"].get(name)
    if profile is None:
        return {"status": "uncalibrated_gpu", "gpu": name}
    if dtype not in profile["supported_dtypes"]:
        return {"status": "unsupported_dtype", "gpu": name, "dtype": dtype}
    outside = nonnegative_bytes(profile["outside_median_bytes"], "outside")
    upper = nonnegative_bytes(profile["outside_max_training_sample_bytes"], "outside upper")
    if upper < outside:
        raise ValueError("Overhead upper bound is smaller than its median")
    return {
        "status": "ok", "gpu": name,
        "allocated_bytes": allocated, "reserved_bytes": reserved,
        "outside_estimate_bytes": outside,
        "outside_observed_upper_bytes": upper,
        "estimated_nvml_bytes": reserved + outside,
        "required_vram_bytes": reserved + upper,
        "required_vram_gb": (reserved + upper) / 1024**3,
        "policy": "simulated_reserved_plus_training_observed_upper",
        "calibration_id": profiles["calibration_id"],
        "calibration_scope": profile["runtime_scope"],
    }


def memory_result(allocated, reserved, dtype, gpu_name=None, profiles=None):
    allocated = nonnegative_bytes(allocated, "allocated")
    reserved = nonnegative_bytes(reserved, "reserved")
    if reserved < allocated:
        raise ValueError("Reserved estimate is smaller than allocated estimate")
    if profiles is None:
        profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    by_gpu = {name: process_estimate(allocated, reserved, name, dtype, profiles)
              for name in profiles["gpus"]}
    target = process_estimate(allocated, reserved, gpu_name, dtype, profiles) if gpu_name else None
    if target and target["status"] != "ok":
        raise ValueError(f"Target GPU has no valid memory calibration: {target}")
    # The legacy top-level display needs a process requirement, not a tensor-only
    # lower bound. Until a GPU is selected, use the largest calibrated requirement;
    # the candidate filter still uses each GPU's own unrounded byte requirement.
    calibrated = [v["required_vram_bytes"] for v in by_gpu.values() if v["status"] == "ok"]
    if not calibrated:
        raise ValueError("No calibrated GPU supports the requested dtype")
    required = target["required_vram_bytes"] if target else max(calibrated)
    return {
        "prediction_version": VERSION,
        "predicted_allocated_bytes": allocated,
        "predicted_reserved_bytes": reserved,
        "required_vram_bytes": required,
        "required_vram_gb": round(required / 1024**3, 2),
        "required_vram_scope": "target_process" if target else "maximum_among_calibrated_gpus",
        "target_gpu_estimate": target,
        "gpu_requirements": by_gpu,
        "breakdown_bytes": {
            "predicted_allocated_native_dtype": allocated,
            "predicted_reserved_native_dtype": reserved,
            "empirical_tensor_correction": 0,
            "fixed_margin": 0,
            "outside_gpu_estimate": target["outside_estimate_bytes"] if target else None,
        },
    }
