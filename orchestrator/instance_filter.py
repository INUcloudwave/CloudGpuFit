"""AWS GPU instance candidate filter.

Turns a VRAM requirement into the AWS single-GPU instances that can be
recommended. An instance is a candidate ONLY if eager_runtime has a target
for it -- targets/<instance>/ with its own validated manifest, either a full
calibrated target or a size target built by build_size_target.py. There is no
proxy: an instance that was never measured (g6.2xlarge, ... ) is not
offered, and no latency is ever scaled by vCPU count. A new size becomes a
candidate the moment its target directory exists, with no code change here.

GPU/host specs come from SUPPORTED_GPU_INSTANCES; the target's manifest must
have been calibrated on the family's GPU (GPU_FAMILIES).

AWS prices and region availability are not hardcoded: the caller passes
`instance_catalog` (aws_pricing.py / instance_catalog.py). A missing catalog
row means "availability unknown" and the instance is dropped; a row with
price None means "available, price unknown" and the budget check is skipped.

History: unmeasured sizes (2xlarge-16xlarge, g5g, ...) used to borrow a
calibrated .xlarge result scaled by a guessed vCPU curve
(latency_scale_for_vcpu). That proxy is removed: they stay in
SUPPORTED_GPU_INSTANCES but are recommended only once they have their own
target (measured, or built from a size profile with build_size_target.py --
GPU kernel models shared, only the measured host overheads differ).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from eager_runtime import available_targets, target_provenance

RUNTIME_ROOT = Path(__file__).resolve().parent / "eager_runtime"
ENGINE_MODES = ("eager", "compiled")

# Every single-GPU instance the service knows about: the original list from
# EC2 DescribeInstanceTypes (ap-northeast-2) minus g5g (Graviton/T4G, decided
# not to support), plus g7.2xlarge from
# https://aws.amazon.com/ec2/instance-types/g7/. aws_pricing.py fetches prices
# for exactly these keys.
#
# Being listed here does NOT make an instance recommendable. It is recommended
# only when eager_runtime has its own target for it (targets/<instance>/ with a
# validated manifest). Today that is g4dn.xlarge, g5.xlarge, g6.xlarge,
# g6e.xlarge, g7.2xlarge, g7e.2xlarge and gr6.4xlarge. The others stay listed
# and start being recommended automatically once their measured target is
# added -- there is no proxy and no vCPU scaling in between.
SUPPORTED_GPU_INSTANCES: dict[str, dict[str, Any]] = {
    "g4dn.xlarge": {"gpu_name": "NVIDIA T4", "gpu_count": 1, "gpu_vram_gb": 16, "vcpu": 4},
    "g4dn.2xlarge": {"gpu_name": "NVIDIA T4", "gpu_count": 1, "gpu_vram_gb": 16, "vcpu": 8},
    "g4dn.4xlarge": {"gpu_name": "NVIDIA T4", "gpu_count": 1, "gpu_vram_gb": 16, "vcpu": 16},
    "g4dn.8xlarge": {"gpu_name": "NVIDIA T4", "gpu_count": 1, "gpu_vram_gb": 16, "vcpu": 32},
    "g4dn.16xlarge": {"gpu_name": "NVIDIA T4", "gpu_count": 1, "gpu_vram_gb": 16, "vcpu": 64},
    "g5.xlarge": {"gpu_name": "NVIDIA A10G", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 4},
    "g5.2xlarge": {"gpu_name": "NVIDIA A10G", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 8},
    "g5.4xlarge": {"gpu_name": "NVIDIA A10G", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 16},
    "g5.8xlarge": {"gpu_name": "NVIDIA A10G", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 32},
    "g5.16xlarge": {"gpu_name": "NVIDIA A10G", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 64},
    "g6.xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 4},
    "g6.2xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 8},
    "g6.4xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 16},
    "g6.8xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 32},
    "g6.16xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 64},
    "g6e.xlarge": {"gpu_name": "NVIDIA L40S", "gpu_count": 1, "gpu_vram_gb": 48, "vcpu": 4},
    "g6e.2xlarge": {"gpu_name": "NVIDIA L40S", "gpu_count": 1, "gpu_vram_gb": 48, "vcpu": 8},
    "g6e.4xlarge": {"gpu_name": "NVIDIA L40S", "gpu_count": 1, "gpu_vram_gb": 48, "vcpu": 16},
    "g6e.8xlarge": {"gpu_name": "NVIDIA L40S", "gpu_count": 1, "gpu_vram_gb": 48, "vcpu": 32},
    "g6e.16xlarge": {"gpu_name": "NVIDIA L40S", "gpu_count": 1, "gpu_vram_gb": 48, "vcpu": 64},
    "g7.2xlarge": {"gpu_name": "NVIDIA RTX PRO 4500 Blackwell Server Edition", "gpu_count": 1, "gpu_vram_gb": 32, "vcpu": 8},
    "g7e.2xlarge": {"gpu_name": "NVIDIA RTX PRO Server 6000", "gpu_count": 1, "gpu_vram_gb": 96, "vcpu": 8},
    "g7e.4xlarge": {"gpu_name": "NVIDIA RTX PRO Server 6000", "gpu_count": 1, "gpu_vram_gb": 96, "vcpu": 16},
    "g7e.8xlarge": {"gpu_name": "NVIDIA RTX PRO Server 6000", "gpu_count": 1, "gpu_vram_gb": 96, "vcpu": 32},
    "gr6.4xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 16},
    "gr6.8xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 32},
}

# Safety check: the GPU a target was calibrated on (its manifest's gpu_name)
# must be the family's GPU, so an instance can never be answered with another
# GPU's models. A family not listed here accepts the GPU recorded in its own
# validated manifest when its target arrives.
GPU_FAMILIES: dict[str, str] = {
    "g4dn": "Tesla T4",
    "g5": "NVIDIA A10G",
    "g6": "NVIDIA L4",
    "g6e": "NVIDIA L40S",
    "g7": "NVIDIA RTX PRO 4500 Blackwell Server Edition",
    "g7e": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
}


def _manifest(instance_type: str, runtime_root: Path) -> dict[str, Any] | None:
    path = runtime_root / "targets" / instance_type / "runtime_manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def instance_spec(instance_type: str, runtime_root: Path = RUNTIME_ROOT) -> dict[str, Any] | None:
    """Spec of a recommendable instance, or None if it is not listed, has no
    target of its own, or its target was calibrated on a different GPU."""
    listed = SUPPORTED_GPU_INSTANCES.get(instance_type)
    manifest = _manifest(instance_type, runtime_root)
    if listed is None or manifest is None:
        return None
    size_profile = manifest.get("size_profile") or {}
    family = size_profile.get("gpu_family") or instance_type.split(".", 1)[0]
    expected_gpu = GPU_FAMILIES.get(family)
    if expected_gpu is not None and manifest.get("gpu_name") != expected_gpu:
        return None
    return {**listed, "gpu_family": family, "calibrated_gpu": manifest.get("gpu_name")}


def supported_instances(
    engine_mode: str = "eager",
    runtime_root: Path = RUNTIME_ROOT,
) -> dict[str, dict[str, Any]]:
    """Every instance that can be recommended for this engine mode."""
    specs = {}
    for instance_type in available_targets(engine_mode=engine_mode):
        spec = instance_spec(instance_type, runtime_root)
        if spec is not None:
            specs[instance_type] = spec
    return specs


def recommendable_instance_types(runtime_root: Path = RUNTIME_ROOT) -> list[str]:
    """Every instance that can be recommended in at least one engine mode --
    the set aws_pricing.py fetches prices for."""
    names: set[str] = set()
    for mode in ENGINE_MODES:
        names |= set(supported_instances(mode, runtime_root))
    return sorted(names)


@dataclass(frozen=True)
class _CatalogEntry:
    instance_type: str
    aws_region: str
    price_usd_per_hour: float | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "_CatalogEntry":
        price = data.get("price_usd_per_hour")
        return cls(
            instance_type=str(data["instance_type"]),
            aws_region=str(data["aws_region"]),
            price_usd_per_hour=float(price) if price is not None else None,
        )


def filter_instances(
    required_vram_gb: float,
    aws_region: str,
    budget_usd_per_hour: float | None,
    instance_catalog: Sequence[Mapping[str, Any]],
    installed_targets: Iterable[str] | None = None,
    *,
    engine_mode: str = "eager",
    supported: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Candidates passing every condition:

      1. eager_runtime has a target for the instance itself in `engine_mode`
      2. gpu_vram_gb >= required_vram_gb
      3. listed as available in `aws_region` by instance_catalog
      4. price <= budget, when the price is known

    Sorted by ascending gpu_vram_gb (price may be partly unknown).
    """
    specs = dict(supported) if supported is not None else supported_instances(engine_mode)
    if installed_targets is not None:
        allowed = set(installed_targets)
        specs = {name: spec for name, spec in specs.items() if name in allowed}

    catalog_by_type: dict[str, _CatalogEntry] = {}
    for raw in instance_catalog:
        entry = _CatalogEntry.from_mapping(raw)
        if entry.aws_region == aws_region:
            catalog_by_type[entry.instance_type] = entry

    candidates: list[dict[str, Any]] = []
    for instance_type, spec in specs.items():
        if spec["gpu_vram_gb"] < required_vram_gb:
            continue
        entry = catalog_by_type.get(instance_type)
        if entry is None:
            continue
        price = entry.price_usd_per_hour
        if budget_usd_per_hour is not None and price is not None and price > budget_usd_per_hour:
            continue
        candidates.append(
            {
                "instance_type": instance_type,
                "gpu_name": spec["gpu_name"],
                "gpu_count": spec["gpu_count"],
                "gpu_vram_gb": spec["gpu_vram_gb"],
                "vcpu": spec["vcpu"],
                "price_usd_per_hour": price,
                "aws_region": aws_region,
                "profile_provenance": target_provenance(instance_type)["status"],
            }
        )
    candidates.sort(key=lambda item: (item["gpu_vram_gb"], item["instance_type"]))
    return candidates
