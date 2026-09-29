#!/usr/bin/env python3
"""Build a size-specific target that shares its GPU kernel models with a base.

Instances that use the same GPU and differ only in vCPU/RAM (g6.xlarge,
g6.2xlarge, g6.4xlarge, gr6.4xlarge, ...) run the same GPU kernels, so their
kernel-time models (Eager *.pkl, Compiled LightGBM *.txt) are shared. Only the
host-side overhead profile differs: H2D/D2H transfer, dispatch light/heavy,
synchronization, Python loop and (optionally) the Compiled fixed overhead.

This tool turns one size profile JSON into a *thin* runnable target:

    size_profiles/<gpu_family>/family.json              GPU family -> base target
    size_profiles/<gpu_family>/sizes/<instance>.json    one instance size
        |
        v  python build_size_target.py --profile size_profiles/g6/sizes/gr6.4xlarge.json
    targets/<instance>/
        runtime_manifest.json       GPU artifacts point into ../<base>/ (sha256 pinned)
        hardware_profile.json       base profile + this size's overheads
        p90_calibration_factor.json / first_token_calibration.json
        compiled_external_model/compiled_external_manifest.json   (paths -> base)
        compiled_fusion_model/compiled_fusion_manifest.json       (paths -> base)
        compiled_context_calibration.json   only if the profile asks for one
                                            (null = none, "inherit_base", or values)

The runtime needs no changes for this: a thin target is an ordinary
targets/<instance>/ directory with a valid manifest, so it is discovered by
available_targets() and validated by the same size/sha256/target checks as a
full target. Because every shared GPU artifact is pinned by sha256, changing
or retraining the base target makes the thin target fail validation until it
is rebuilt -- it can never silently run on different GPU models.

What this deliberately does NOT do: scale the whole GPU latency by vCPU
count. GPU kernel time comes from the shared GPU models; only the host-side
overhead fields listed in OVERHEAD_FIELDS are taken from the size profile.

Provenance (measured / interpolated / extrapolated) is metadata describing how
trustworthy a size's overhead numbers are. It is copied into the target's
manifest and surfaced in every prediction result as ``profile_provenance``;
it never changes how a prediction is computed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_MODULE_ROOT = Path(__file__).resolve().parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from runtime_manifest import validate_manifest  # noqa: E402


SIZE_PROFILE_SCHEMA_VERSION = 1
PROVENANCE_STATUSES = ("measured", "interpolated", "extrapolated")

# hardware_profile.json fields that depend on the instance size (host CPU,
# memory, PCIe path), not on the GPU. Units follow hardware_profile.json:
# *_base / t_* in milliseconds, bw_* in GB/s.
OVERHEAD_FIELDS = (
    "t_h2d_base",
    "t_d2h_base",
    "bw_h2d_GBs",
    "bw_d2h_GBs",
    "t_dispatch_light",
    "t_dispatch_heavy",
    "t_sync",
    "t_python_loop",
)

# Files inside a target that belong to the instance size, not the GPU.
SIZE_LEVEL_FILES = {
    "hardware_profile.json",
    "p90_calibration_factor.json",
    "first_token_calibration.json",
}

COMPILED_MANIFESTS = (
    ("compiled_external_model", "compiled_external_manifest.json"),
    ("compiled_fusion_model", "compiled_fusion_manifest.json"),
)

CONTEXT_CALIBRATION_FIELDS = (
    "kernel_scale",
    "external_launch_ms",
    "fusion_launch_ms",
    "fixed_overhead_ms",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _relative(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()


def _non_negative(value: Any, name: str) -> float:
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative, got {number}")
    return number


def _validate_isolated(manifest_path: Path, target: str) -> dict[str, Any]:
    """validate_manifest() exports the target's paths as environment variables
    and refuses to overwrite ones that point elsewhere (the guard that keeps
    one GPU's artifacts from leaking into another target). Validate each
    target with a clean slate and restore the caller's environment after."""
    payload = _read_json(manifest_path)
    names = {"EAGER_TARGET_ROOT", "EAGER_TARGET_INSTANCE"} | set(
        payload.get("environment", {})
    )
    saved = {name: os.environ.pop(name, None) for name in names}
    try:
        return validate_manifest(manifest_path, expected_target=target, check_gpu=False)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def load_family(profile_path: Path, runtime_root: Path) -> dict[str, Any]:
    family_path = profile_path.parent.parent / "family.json"
    family = _read_json(family_path)
    if family.get("schema_version") != SIZE_PROFILE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported family schema: {family_path}")
    for key in ("gpu_family", "gpu_name", "gpu_base_target"):
        if not family.get(key):
            raise ValueError(f"{family_path}: '{key}' is required")
    base_dir = runtime_root / "targets" / family["gpu_base_target"]
    if not (base_dir / "runtime_manifest.json").is_file():
        raise FileNotFoundError(
            f"GPU base target {family['gpu_base_target']!r} has no runtime_manifest.json"
        )
    family["_path"] = family_path
    return family


def load_size_profile(profile_path: Path) -> dict[str, Any]:
    profile = _read_json(profile_path)
    if profile.get("schema_version") != SIZE_PROFILE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported size profile schema: {profile_path}")
    instance_type = profile.get("instance_type")
    if not instance_type or "/" in instance_type or "\\" in instance_type:
        raise ValueError(f"{profile_path}: invalid instance_type {instance_type!r}")

    overheads = profile.get("overheads")
    if not isinstance(overheads, dict):
        raise ValueError(f"{profile_path}: 'overheads' object is required")
    unknown = sorted(set(overheads) - set(OVERHEAD_FIELDS))
    if unknown:
        raise ValueError(
            f"{profile_path}: only host-side overhead fields may be overridden, "
            f"got {unknown} (allowed: {list(OVERHEAD_FIELDS)})"
        )
    missing = [name for name in OVERHEAD_FIELDS if name not in overheads]
    if missing:
        raise ValueError(f"{profile_path}: missing overhead fields {missing}")
    for name in OVERHEAD_FIELDS:
        _non_negative(overheads[name], name)
    for name in ("bw_h2d_GBs", "bw_d2h_GBs"):
        if float(overheads[name]) <= 0.0:
            raise ValueError(f"{profile_path}: {name} must be positive")

    provenance = profile.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{profile_path}: 'provenance' object is required")
    status = provenance.get("status")
    if status not in PROVENANCE_STATUSES:
        raise ValueError(
            f"{profile_path}: provenance.status must be one of {PROVENANCE_STATUSES}"
        )
    for field, field_status in (provenance.get("fields") or {}).items():
        if field not in OVERHEAD_FIELDS and field != "compiled_context_calibration":
            raise ValueError(f"{profile_path}: provenance for unknown field {field!r}")
        if field_status not in PROVENANCE_STATUSES:
            raise ValueError(
                f"{profile_path}: provenance.fields.{field} must be one of "
                f"{PROVENANCE_STATUSES}"
            )
    if status != "measured" and not provenance.get("basis"):
        raise ValueError(
            f"{profile_path}: an {status} profile must list the measured "
            "instance sizes it was derived from in provenance.basis"
        )

    if "compiled_context_calibration" not in profile:
        raise ValueError(
            f"{profile_path}: 'compiled_context_calibration' must be stated "
            "explicitly: null (none), \"inherit_base\", or an object"
        )
    context = profile["compiled_context_calibration"]
    if context is not None and context != "inherit_base":
        if not isinstance(context, dict):
            raise ValueError(
                f"{profile_path}: compiled_context_calibration must be null, "
                "\"inherit_base\" or an object"
            )
        for name in CONTEXT_CALIBRATION_FIELDS:
            _non_negative(context[name], f"compiled_context_calibration.{name}")
    return profile


def _thin_eager_manifest(
    base_manifest: dict[str, Any],
    *,
    base_dir: Path,
    target_dir: Path,
    instance_type: str,
    size_block: dict[str, Any],
) -> dict[str, Any]:
    manifest = copy.deepcopy(base_manifest)
    manifest["target_instance"] = instance_type
    environment = {}
    for name, relative_path in base_manifest.get("environment", {}).items():
        if Path(relative_path).name in SIZE_LEVEL_FILES:
            environment[name] = relative_path
        else:
            environment[name] = _relative(base_dir / relative_path, target_dir)
    manifest["environment"] = environment

    artifacts = []
    for item in base_manifest["artifacts"]:
        if Path(item["path"]).name in SIZE_LEVEL_FILES:
            local = target_dir / item["path"]
            if not local.is_file():
                continue
            artifacts.append(
                {
                    "path": item["path"],
                    "size_bytes": local.stat().st_size,
                    "sha256": sha256(local),
                }
            )
        else:
            artifacts.append(
                {
                    **item,
                    "path": _relative(base_dir / item["path"], target_dir),
                }
            )
    manifest["artifacts"] = artifacts
    manifest["size_profile"] = size_block
    return manifest


def _thin_compiled_manifest(
    base_manifest: dict[str, Any],
    *,
    base_root: Path,
    target_root: Path,
    instance_type: str,
) -> dict[str, Any]:
    manifest = copy.deepcopy(base_manifest)
    manifest["target_instance"] = instance_type
    for artifact in manifest.get("artifacts", {}).values():
        artifact["path"] = _relative(base_root / artifact["path"], target_root)
    kernel_floor = manifest.get("kernel_floor")
    if isinstance(kernel_floor, dict) and kernel_floor.get("profile_path"):
        kernel_floor["profile_path"] = _relative(
            base_root.parent / kernel_floor["profile_path"], target_root.parent
        )
    return manifest


def build(
    profile_path: Path,
    runtime_root: Path,
    *,
    force: bool = False,
) -> Path:
    profile_path = profile_path.resolve()
    runtime_root = runtime_root.resolve()
    profile = load_size_profile(profile_path)
    family = load_family(profile_path, runtime_root)

    instance_type = profile["instance_type"]
    base_target = family["gpu_base_target"]
    if instance_type == base_target:
        raise ValueError(
            f"{instance_type} is the GPU base target itself (a full, directly "
            "calibrated target); its size profile documents the base and is "
            "not built."
        )
    base_dir = runtime_root / "targets" / base_target
    target_dir = runtime_root / "targets" / instance_type

    base_manifest = _validate_isolated(base_dir / "runtime_manifest.json", base_target)
    if base_manifest.get("gpu_name") != family["gpu_name"]:
        raise ValueError(
            f"GPU mismatch: family {family['gpu_family']} is {family['gpu_name']!r} "
            f"but base target {base_target} was calibrated on "
            f"{base_manifest.get('gpu_name')!r}"
        )

    if target_dir.exists():
        existing = target_dir / "runtime_manifest.json"
        is_thin = existing.is_file() and "size_profile" in _read_json(existing)
        if not is_thin and not force:
            raise FileExistsError(
                f"{target_dir} already holds a full target. Refusing to replace "
                "directly calibrated artifacts (pass --force to override)."
            )
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    # hardware_profile.json: GPU fields from the base, overheads from the size.
    hardware = _read_json(base_dir / "hardware_profile.json")
    hardware["target_instance"] = instance_type
    for name in OVERHEAD_FIELDS:
        hardware[name] = float(profile["overheads"][name])
    _write_json(target_dir / "hardware_profile.json", hardware)

    # Size-level calibration files: inherit the base's unless the profile
    # points at files measured on this size.
    calibration_files = profile.get("calibration_files") or {}
    for filename in ("p90_calibration_factor.json", "first_token_calibration.json"):
        key = filename.removesuffix(".json")
        source_value = calibration_files.get(key, "inherit_base")
        if source_value is None:
            continue
        source = (
            base_dir / filename
            if source_value == "inherit_base"
            else (profile_path.parent / source_value).resolve()
        )
        if not source.is_file():
            continue
        payload = _read_json(source)
        if "target_instance" in payload:
            payload["target_instance"] = instance_type
        _write_json(target_dir / filename, payload)

    field_provenance = {
        name: (profile["provenance"].get("fields") or {}).get(
            name, profile["provenance"]["status"]
        )
        for name in OVERHEAD_FIELDS
    }
    size_block = {
        "schema_version": SIZE_PROFILE_SCHEMA_VERSION,
        "gpu_family": family["gpu_family"],
        "gpu_base_target": base_target,
        "profile_path": _relative(profile_path, runtime_root),
        "profile_sha256": sha256(profile_path),
        "provenance_status": profile["provenance"]["status"],
        "field_provenance": field_provenance,
        "provenance_basis": profile["provenance"].get("basis", []),
        "vcpu": profile.get("vcpu"),
        "ram_gib": profile.get("ram_gib"),
        "shared_gpu_artifacts": True,
        "built_by": "build_size_target.py",
    }
    manifest = _thin_eager_manifest(
        base_manifest,
        base_dir=base_dir,
        target_dir=target_dir,
        instance_type=instance_type,
        size_block=size_block,
    )
    _write_json(target_dir / "runtime_manifest.json", manifest)

    for folder, manifest_name in COMPILED_MANIFESTS:
        base_manifest_path = base_dir / folder / manifest_name
        if not base_manifest_path.is_file():
            continue
        thin = _thin_compiled_manifest(
            _read_json(base_manifest_path),
            base_root=base_dir / folder,
            target_root=target_dir / folder,
            instance_type=instance_type,
        )
        _write_json(target_dir / folder / manifest_name, thin)

    context = profile["compiled_context_calibration"]
    if context == "inherit_base":
        base_context = base_dir / "compiled_context_calibration.json"
        context = _read_json(base_context) if base_context.is_file() else None
    if context is not None:
        _write_json(
            target_dir / "compiled_context_calibration.json",
            {
                "schema_version": 2,
                "target_instance": instance_type,
                "engine_mode": "compiled",
                "model_specific": False,
                "model_name_is_feature": False,
                **{name: float(context[name]) for name in CONTEXT_CALIBRATION_FIELDS},
                "source": context.get("source"),
            },
        )

    _validate_isolated(target_dir / "runtime_manifest.json", instance_type)
    return target_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", type=Path, required=True, nargs="+")
    parser.add_argument("--runtime-root", type=Path, default=_MODULE_ROOT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing FULL target directory (never needed for rebuilds of thin targets)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for profile in args.profile:
        target_dir = build(profile, args.runtime_root, force=args.force)
        manifest = _read_json(target_dir / "runtime_manifest.json")
        block = manifest["size_profile"]
        print(
            f"built {manifest['target_instance']}: GPU models shared from "
            f"{block['gpu_base_target']}, overheads={block['provenance_status']} "
            f"({target_dir})"
        )


if __name__ == "__main__":
    main()
