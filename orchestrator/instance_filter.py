"""AWS GPU instance candidate filter.

Turns an xMem VRAM requirement into a list of AWS single-GPU G-series
instance candidates that `eager_runtime` has a calibrated profile for.

Scope (프로토타입, CLAUDE_HANDOFF_CLOUDGPUFIT_EAGER_RUNTIME.md §1, §8):
  - 추론(inference) 전용이므로 g-시리즈만 대상으로 한다.
  - 프로토타입이므로 GPU가 1개뿐인 인스턴스만 지원한다
    (g4dn.xlarge / g5.xlarge / g6.xlarge / g6e.xlarge).
    g5.2xlarge, g5.12xlarge(멀티 GPU) 등은 이번 범위 밖이다.
  - eager_runtime이 실제로 캘리브레이션된 target과 1:1 대응해야 한다.

이 모듈은 AWS 가격을 하드코딩하지 않는다 (§8, §16: "AWS 가격을 출처 없이
하드코딩하지 않는다"). region별 가용성/가격은 호출자가 `instance_catalog`로
넘긴다 — 다른 팀원이 관리하는 인스턴스/가격 테이블이나 AWS Price List API
결과를 그대로 넣으면 된다. catalog가 비어 있으면(아직 준비되지 않았으면)
이 필터는 후보를 하나도 반환하지 않는다 — 없는 데이터를 추측해서 채우지
않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

# GPU VRAM 기준은 AWS 공식 인스턴스 스펙이며 CLAUDE_HANDOFF §8의 표와
# 동일하다. gpu_count == 1인 target만 등록한다 (프로토타입 범위).
SUPPORTED_GPU_INSTANCES: dict[str, dict[str, Any]] = {
    "g4dn.xlarge": {"gpu_name": "NVIDIA T4", "gpu_count": 1, "gpu_vram_gb": 16},
    "g5.xlarge": {"gpu_name": "NVIDIA A10G", "gpu_count": 1, "gpu_vram_gb": 24},
    "g6.xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24},
    "g6e.xlarge": {"gpu_name": "NVIDIA L40S", "gpu_count": 1, "gpu_vram_gb": 48},
}


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
    installed_targets: Iterable[str],
    *,
    supported_instances: Mapping[str, Mapping[str, Any]] = SUPPORTED_GPU_INSTANCES,
) -> list[dict[str, Any]]:
    """Return single-GPU G-series candidates passing every condition:

      1. gpu_vram_gb >= required_vram_gb
      2. instance is listed as available in `aws_region` by instance_catalog
      3. price_usd_per_hour <= budget_usd_per_hour, when a price is known
      4. eager_runtime has an installed profile for it (`installed_targets`)

    instance_catalog: list of {"instance_type", "aws_region",
    "price_usd_per_hour"} rows. A region+instance combination with no
    matching row is treated as "availability unknown" and is dropped —
    not guessed into or out of the result. A row that *is* present for the
    region but carries `price_usd_per_hour: None` means "available, price
    unknown": it is kept and the budget check is skipped for it rather than
    invented as 0.0 (CLAUDE_HANDOFF §8/§16).

    Returns candidates sorted by ascending gpu_vram_gb (cheapest-fit first
    in VRAM terms — not by price, since price may be partly unknown).
    """
    installed = set(installed_targets)

    catalog_by_type: dict[str, _CatalogEntry] = {}
    for raw in instance_catalog:
        entry = _CatalogEntry.from_mapping(raw)
        if entry.aws_region == aws_region:
            catalog_by_type[entry.instance_type] = entry

    candidates: list[dict[str, Any]] = []
    for instance_type, spec in supported_instances.items():
        if spec["gpu_vram_gb"] < required_vram_gb:
            continue
        if instance_type not in installed:
            continue

        catalog_entry = catalog_by_type.get(instance_type)
        if catalog_entry is None:
            # No catalog row for this instance in this region: availability
            # is unknown, so it is not offered as a candidate.
            continue

        price = catalog_entry.price_usd_per_hour
        if (
            budget_usd_per_hour is not None
            and price is not None
            and price > budget_usd_per_hour
        ):
            continue

        candidates.append(
            {
                "instance_type": instance_type,
                "gpu_name": spec["gpu_name"],
                "gpu_count": spec["gpu_count"],
                "gpu_vram_gb": spec["gpu_vram_gb"],
                "price_usd_per_hour": price,
                "aws_region": aws_region,
            }
        )

    candidates.sort(key=lambda item: item["gpu_vram_gb"])
    return candidates
