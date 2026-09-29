#!/usr/bin/env python3
"""Build and validate target-specific Eager runtime artifact manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCHEMA_VERSION = 2
PREDICTOR_VERSION = "eager_v6_first_token_structural"
DEFAULT_TARGET = "g6.xlarge"
TARGETS_DIRECTORY = Path("targets")
DEFAULT_MANIFEST_NAME = "runtime_manifest.json"
DEFAULT_FACTOR_NAME = "p90_calibration_factor.json"
DEFAULT_HARDWARE_PROFILE_NAME = "hardware_profile.json"
DEFAULT_FIRST_TOKEN_CALIBRATION_NAME = "first_token_calibration.json"

MODEL_DIRECTORIES = {
    "cache": Path("cache_model"),
    "memory": Path("memory_model"),
    "sdpa": Path("sdpa_model"),
    "linear_context": Path("linear_context_model_deep_v2"),
    "pointwise": Path("pointwise_model_v4"),
    "conv1d": Path("conv1d_model"),
}

ROOT_MODELS = (
    "gemm_eager_median.pkl",
    "gemm_eager_90th.pkl",
    "reduction_eager_median.pkl",
    "reduction_eager_90th.pkl",
)

REQUIRED_POINTWISE_MODELS = (
    "pointwise_eager_median.pkl",
    "pointwise_eager_90th.pkl",
)


def target_root(runtime_root: Path, target: str) -> Path:
    return runtime_root.expanduser().resolve() / TARGETS_DIRECTORY / target


def manifest_for_target(runtime_root: Path, target: str) -> Path:
    return target_root(runtime_root, target) / DEFAULT_MANIFEST_NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"Artifact must be inside target root: {path}") from exc


def discover_artifacts(root: Path, factor_path: Path) -> list[Path]:
    missing = [root / name for name in ROOT_MODELS if not (root / name).is_file()]

    pointwise_dir = root / MODEL_DIRECTORIES["pointwise"]
    missing.extend(
        pointwise_dir / name
        for name in REQUIRED_POINTWISE_MODELS
        if not (pointwise_dir / name).is_file()
    )

    empty_directories = []
    artifacts = [root / name for name in ROOT_MODELS if (root / name).is_file()]
    for label, relative_dir in MODEL_DIRECTORIES.items():
        directory = root / relative_dir
        models = sorted(directory.glob("*.pkl")) if directory.is_dir() else []
        if not models:
            empty_directories.append(f"{label}: {directory}")
        artifacts.extend(models)

    if not factor_path.is_file():
        missing.append(factor_path)
    else:
        artifacts.append(factor_path)

    hardware_profile = root / DEFAULT_HARDWARE_PROFILE_NAME
    if not hardware_profile.is_file():
        missing.append(hardware_profile)
    else:
        artifacts.append(hardware_profile)

    # Optional (deviates from the original bundle): a target without a
    # first_token_calibration.json is not treated as broken -- it simply
    # runs uncalibrated for decoder_llm/text_encdec first-token predictions.
    first_token_calibration = root / DEFAULT_FIRST_TOKEN_CALIBRATION_NAME
    if first_token_calibration.is_file():
        artifacts.append(first_token_calibration)

    if missing or empty_directories:
        lines = ["Cannot build Eager runtime manifest."]
        lines.extend(f"Missing file: {path}" for path in missing)
        lines.extend(
            f"Missing model directory contents: {item}"
            for item in empty_directories
        )
        raise FileNotFoundError("\n".join(lines))

    return sorted(set(path.resolve() for path in artifacts))


def detect_gpu_name() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")
    return torch.cuda.get_device_name(0)


def build_manifest(
    root: Path,
    target: str,
    factor_path: Path,
    gpu_name: str,
) -> dict:
    artifacts = discover_artifacts(root, factor_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "predictor_version": PREDICTOR_VERSION,
        "target_instance": target,
        "mode": "eager",
        "gpu_name": gpu_name,
        "environment": {
            "EAGER_BASE_MODEL_DIR": ".",
            "CACHE_MODEL_DIR": str(MODEL_DIRECTORIES["cache"]),
            "MEMORY_MODEL_DIR": str(MODEL_DIRECTORIES["memory"]),
            "SDPA_MODEL_DIR": str(MODEL_DIRECTORIES["sdpa"]),
            "POINTWISE_MODEL_DIR": str(MODEL_DIRECTORIES["pointwise"]),
            "LINEAR_CONTEXT_MODEL_DIR": str(
                MODEL_DIRECTORIES["linear_context"]
            ),
            "CONV1D_MODEL_DIR": str(MODEL_DIRECTORIES["conv1d"]),
            "EAGER_P90_CALIBRATION_FILE": relative_to_root(factor_path, root),
            "EAGER_HARDWARE_PROFILE": DEFAULT_HARDWARE_PROFILE_NAME,
            "EAGER_FIRST_TOKEN_CALIBRATION_FILE": (
                DEFAULT_FIRST_TOKEN_CALIBRATION_NAME
            ),
        },
        "artifacts": [
            {
                "path": relative_to_root(path, root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
    }


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Runtime manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported runtime manifest schema: {path}")
    if payload.get("predictor_version") != PREDICTOR_VERSION:
        raise ValueError(f"Unsupported predictor version: {path}")
    if payload.get("mode") != "eager":
        raise ValueError(f"Manifest is not for Eager mode: {path}")
    target = payload.get("target_instance")
    if not isinstance(target, str) or not target:
        raise ValueError(f"Manifest has no target_instance: {path}")
    return payload


def configure_environment(payload: dict, root: Path) -> None:
    os.environ["EAGER_TARGET_ROOT"] = str(root.resolve())
    os.environ["EAGER_TARGET_INSTANCE"] = str(payload["target_instance"])
    for name, relative_path in payload.get("environment", {}).items():
        expected = (root / relative_path).resolve()
        configured = os.environ.get(name)
        if configured is not None and Path(configured).expanduser().resolve() != expected:
            raise RuntimeError(
                f"{name} conflicts with the selected target: "
                f"configured={configured}, expected={expected}"
            )
        os.environ[name] = str(expected)


def validate_manifest(
    path: Path,
    expected_target: str | None = None,
    check_gpu: bool = True,
) -> dict:
    path = path.expanduser().resolve()
    payload = load_manifest(path)
    root = path.parent

    if expected_target and payload["target_instance"] != expected_target:
        raise RuntimeError(
            "Runtime target mismatch: "
            f"requested={expected_target}, manifest={payload['target_instance']}"
        )

    configure_environment(payload, root)
    errors = []
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("Manifest has no artifacts")
    else:
        for item in artifacts:
            artifact_path = root / item["path"]
            if not artifact_path.is_file():
                errors.append(f"Missing artifact: {artifact_path}")
                continue
            actual_size = artifact_path.stat().st_size
            if actual_size != item["size_bytes"]:
                errors.append(
                    f"Size mismatch: {artifact_path} "
                    f"({actual_size} != {item['size_bytes']})"
                )
                continue
            if sha256(artifact_path) != item["sha256"]:
                errors.append(f"SHA-256 mismatch: {artifact_path}")

    if check_gpu:
        actual_gpu = detect_gpu_name()
        if actual_gpu != payload.get("gpu_name"):
            errors.append(
                f"GPU mismatch: actual={actual_gpu}, expected={payload.get('gpu_name')}"
            )

    if errors:
        raise RuntimeError("Runtime manifest validation failed:\n" + "\n".join(errors))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--factor", type=Path)
    parser.add_argument("--gpu-name", default="auto")
    parser.add_argument("--check", type=Path)
    parser.add_argument("--skip-gpu-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_root = args.root.expanduser().resolve()

    if args.check:
        payload = validate_manifest(
            args.check,
            expected_target=args.target,
            check_gpu=not args.skip_gpu_check,
        )
        print(
            f"Runtime manifest valid: {args.check} "
            f"(target={payload['target_instance']}, "
            f"artifacts={len(payload['artifacts'])}, GPU={payload['gpu_name']})"
        )
        return

    root = target_root(runtime_root, args.target)
    factor_path = (
        args.factor.expanduser().resolve()
        if args.factor
        else root / DEFAULT_FACTOR_NAME
    )
    gpu_name = detect_gpu_name() if args.gpu_name == "auto" else args.gpu_name
    payload = build_manifest(root, args.target, factor_path, gpu_name)
    output = (
        args.output.expanduser().resolve()
        if args.output
        else root / DEFAULT_MANIFEST_NAME
    )
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Saved runtime manifest: {output} "
        f"(target={args.target}, artifacts={len(payload['artifacts'])}, GPU={gpu_name})"
    )


if __name__ == "__main__":
    main()
