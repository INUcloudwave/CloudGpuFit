"""AWS GPU instance candidate filter.

Turns an xMem VRAM requirement into a list of AWS single-GPU G-series
instance candidates that `eager_runtime` has -- directly or via the
size-scaling proxy below -- a calibrated latency basis for.

Scope (프로토타입, CLAUDE_HANDOFF_CLOUDGPUFIT_EAGER_RUNTIME.md §1, §8):
  - 추론(inference) 전용이므로 g-시리즈만 대상으로 한다.
  - 프로토타입이므로 GPU가 1개뿐인 인스턴스만 지원한다. 같은 패밀리의
    멀티-GPU 크기(g5.12xlarge/.24xlarge/.48xlarge, g4dn.12xlarge,
    g6.12xlarge/.24xlarge/.48xlarge, g6e.12xlarge/.24xlarge/.48xlarge,
    g5g.16xlarge/.metal, g7e.12xlarge/.24xlarge/.48xlarge 등)는 이번
    범위 밖이다. AWS 실제 스펙(EC2 DescribeInstanceTypes, 이 저장소의
    aws_g_series_ap-northeast-2.json)으로 gpu_count==1만 추려서 아래
    29개를 등록해뒀다.
  - eager_runtime이 실제로 캘리브레이션한 target은 딱 4개
    (g4dn.xlarge/g5.xlarge/g6.xlarge/g6e.xlarge)뿐이다. 나머지 25개
    (같은 패밀리의 2xlarge~16xlarge, 그리고 g5g/g7e/gr6 패밀리 전체)는
    아래 `eager_runtime_profile`로 그 4개 중 하나를 "빌려 쓰고",
    vCPU 개수 차이만큼 `latency_scale_for_vcpu()`로 지연시간을
    조정한다 -- 실측 캘리브레이션이 아니라 근사치다. 자세한 내용과
    "나중에 반드시 갈아엎어야 하는 이유"는 그 함수의 docstring 참고.

이 모듈은 AWS 가격을 하드코딩하지 않는다 (§8, §16: "AWS 가격을 출처 없이
하드코딩하지 않는다"). region별 가용성/가격은 호출자가 `instance_catalog`로
넘긴다 — 다른 팀원이 관리하는 인스턴스/가격 테이블이나 AWS Price List API
결과를 그대로 넣으면 된다. catalog가 비어 있으면(아직 준비되지 않았으면)
이 필터는 후보를 하나도 반환하지 않는다 — 없는 데이터를 추측해서 채우지
않는다.

GPU 모델/VRAM/vCPU는 하드코딩이지만, AWS 가격과 달리 이건 자주 안 바뀌는
정적 스펙 데이터라서(그리고 이 저장소의 aws_g_series_ap-northeast-2.json,
EC2 DescribeInstanceTypes 실측 기준) §8/§16의 "가격 하드코딩 금지"와는
별개 사안이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Any, Iterable, Mapping, Sequence

# FIXME(latency-proxy): 아래 4개(.xlarge, vcpu=4)는 eager_runtime이 실제
# 캘리브레이션한 target이고, 나머지 25개는 전부 이 4개 중 하나의 latency/
# throughput 예측 결과를 "빌려서" vCPU 스케일만 적용한 것이다 (내일
# 영상 제출용으로 빠르게 채워넣은 자리채움 -- 나중에 실제 GPU에서
# 각 크기별로 다시 캘리브레이션해서 이 매핑 자체를 갈아엎어야 한다):
#   - g4dn/g5/g6/g6e의 2xlarge~16xlarge: 같은 패밀리의 .xlarge를 그대로 사용
#     (GPU 칩은 완전히 동일, vCPU/RAM만 다름 -- 그래도 실측은 아님).
#   - g5g(T4g, Graviton): g4dn.xlarge(T4)를 대신 사용. T4g는 T4의 Graviton
#     ARM 버전이라 GPU 성능 자체는 T4와 가장 가까울 것으로 "추정"만 함.
#   - g7e(NVIDIA RTX PRO Server 6000): g6e.xlarge(L40S)를 대신 사용.
#     현재 캘리브레이션된 것 중 가장 강력한 GPU라 골랐을 뿐, RTX PRO 6000
#     자체 성능 특성은 전혀 반영 안 됨.
#   - gr6(NVIDIA L4): g6.xlarge(L4)를 대신 사용. GPU 칩 자체는 동일.
# 스펙 출처: 이 저장소의 aws_g_series_ap-northeast-2.json
# (spec_source: "EC2 DescribeInstanceTypes").
SUPPORTED_GPU_INSTANCES: dict[str, dict[str, Any]] = {
    "g4dn.xlarge": {"gpu_name": "NVIDIA T4", "gpu_count": 1, "gpu_vram_gb": 16, "vcpu": 4, "eager_runtime_profile": "g4dn.xlarge"},
    "g4dn.2xlarge": {"gpu_name": "NVIDIA T4", "gpu_count": 1, "gpu_vram_gb": 16, "vcpu": 8, "eager_runtime_profile": "g4dn.xlarge"},
    "g4dn.4xlarge": {"gpu_name": "NVIDIA T4", "gpu_count": 1, "gpu_vram_gb": 16, "vcpu": 16, "eager_runtime_profile": "g4dn.xlarge"},
    "g4dn.8xlarge": {"gpu_name": "NVIDIA T4", "gpu_count": 1, "gpu_vram_gb": 16, "vcpu": 32, "eager_runtime_profile": "g4dn.xlarge"},
    "g4dn.16xlarge": {"gpu_name": "NVIDIA T4", "gpu_count": 1, "gpu_vram_gb": 16, "vcpu": 64, "eager_runtime_profile": "g4dn.xlarge"},
    "g5.xlarge": {"gpu_name": "NVIDIA A10G", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 4, "eager_runtime_profile": "g5.xlarge"},
    "g5.2xlarge": {"gpu_name": "NVIDIA A10G", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 8, "eager_runtime_profile": "g5.xlarge"},
    "g5.4xlarge": {"gpu_name": "NVIDIA A10G", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 16, "eager_runtime_profile": "g5.xlarge"},
    "g5.8xlarge": {"gpu_name": "NVIDIA A10G", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 32, "eager_runtime_profile": "g5.xlarge"},
    "g5.16xlarge": {"gpu_name": "NVIDIA A10G", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 64, "eager_runtime_profile": "g5.xlarge"},
    "g5g.xlarge": {"gpu_name": "NVIDIA T4G", "gpu_count": 1, "gpu_vram_gb": 16, "vcpu": 4, "eager_runtime_profile": "g4dn.xlarge"},
    "g5g.2xlarge": {"gpu_name": "NVIDIA T4G", "gpu_count": 1, "gpu_vram_gb": 16, "vcpu": 8, "eager_runtime_profile": "g4dn.xlarge"},
    "g5g.4xlarge": {"gpu_name": "NVIDIA T4G", "gpu_count": 1, "gpu_vram_gb": 16, "vcpu": 16, "eager_runtime_profile": "g4dn.xlarge"},
    "g5g.8xlarge": {"gpu_name": "NVIDIA T4G", "gpu_count": 1, "gpu_vram_gb": 16, "vcpu": 32, "eager_runtime_profile": "g4dn.xlarge"},
    "g6.xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 4, "eager_runtime_profile": "g6.xlarge"},
    "g6.2xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 8, "eager_runtime_profile": "g6.xlarge"},
    "g6.4xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 16, "eager_runtime_profile": "g6.xlarge"},
    "g6.8xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 32, "eager_runtime_profile": "g6.xlarge"},
    "g6.16xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 64, "eager_runtime_profile": "g6.xlarge"},
    "g6e.xlarge": {"gpu_name": "NVIDIA L40S", "gpu_count": 1, "gpu_vram_gb": 48, "vcpu": 4, "eager_runtime_profile": "g6e.xlarge"},
    "g6e.2xlarge": {"gpu_name": "NVIDIA L40S", "gpu_count": 1, "gpu_vram_gb": 48, "vcpu": 8, "eager_runtime_profile": "g6e.xlarge"},
    "g6e.4xlarge": {"gpu_name": "NVIDIA L40S", "gpu_count": 1, "gpu_vram_gb": 48, "vcpu": 16, "eager_runtime_profile": "g6e.xlarge"},
    "g6e.8xlarge": {"gpu_name": "NVIDIA L40S", "gpu_count": 1, "gpu_vram_gb": 48, "vcpu": 32, "eager_runtime_profile": "g6e.xlarge"},
    "g6e.16xlarge": {"gpu_name": "NVIDIA L40S", "gpu_count": 1, "gpu_vram_gb": 48, "vcpu": 64, "eager_runtime_profile": "g6e.xlarge"},
    "g7e.2xlarge": {"gpu_name": "NVIDIA RTX PRO Server 6000", "gpu_count": 1, "gpu_vram_gb": 96, "vcpu": 8, "eager_runtime_profile": "g6e.xlarge"},
    "g7e.4xlarge": {"gpu_name": "NVIDIA RTX PRO Server 6000", "gpu_count": 1, "gpu_vram_gb": 96, "vcpu": 16, "eager_runtime_profile": "g6e.xlarge"},
    "g7e.8xlarge": {"gpu_name": "NVIDIA RTX PRO Server 6000", "gpu_count": 1, "gpu_vram_gb": 96, "vcpu": 32, "eager_runtime_profile": "g6e.xlarge"},
    "gr6.4xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 16, "eager_runtime_profile": "g6.xlarge"},
    "gr6.8xlarge": {"gpu_name": "NVIDIA L4", "gpu_count": 1, "gpu_vram_gb": 24, "vcpu": 32, "eager_runtime_profile": "g6.xlarge"},
}

# FIXME(latency-proxy): 순전히 "vCPU가 늘어나면 호스트 쪽 오버헤드(커널 launch,
# 입출력 전처리 등)가 조금 줄어들 것"이라는 손짐작 곡선이다. 실측 데이터
# 없음. GPU가 1개뿐인 인스턴스에서는 추론 자체가 GPU-bound라 vCPU 개수가
# 지연시간에 미치는 실제 영향은 이보다 훨씬 작을 수도, 워크로드에 따라
# 아예 다른 패턴일 수도 있다. vCPU가 2배씩 늘어날 때마다 3%씩 빨라지고,
# 최대 15%(64 vCPU, base 4 vCPU 대비 16배)까지만 빨라지는 것으로 캡을 씌워둔
# 것도 임의의 값이다. 나중에 실제로 각 크기별 GPU 인스턴스를 캘리브레이션
# 하게 되면 이 함수 자체를 통째로 지우고 진짜 계측값으로 교체해야 한다.
_LATENCY_SCALE_PER_VCPU_DOUBLING = 0.03
_LATENCY_SCALE_FLOOR = 0.85
_BASELINE_VCPU = 4


def latency_scale_for_vcpu(vcpu: int) -> float:
    """placeholder 전용: baseline(4 vCPU, 실제 캘리브레이션된 .xlarge 크기)
    대비 예상 지연시간 배율(<=1.0, 작을수록 더 빠름). vCPU가 baseline보다
    적거나 같으면 1.0(스케일 없음)."""
    if vcpu <= _BASELINE_VCPU:
        return 1.0
    doublings = log2(vcpu / _BASELINE_VCPU)
    return max(_LATENCY_SCALE_FLOOR, 1.0 - _LATENCY_SCALE_PER_VCPU_DOUBLING * doublings)


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
      4. eager_runtime has an installed profile for its `eager_runtime_profile`
         (`installed_targets`) -- not necessarily for the instance's own
         name; see the FIXME(latency-proxy) notes above SUPPORTED_GPU_INSTANCES.

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
        if spec["eager_runtime_profile"] not in installed:
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
