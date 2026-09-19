"""Fetches AWS on-demand Linux/Shared EC2 pricing for our known single-GPU
instance types, straight from the AWS Price List Bulk API -- no AWS
credentials, no boto3, since that endpoint is a plain public HTTPS GET.

CLAUDE_HANDOFF §8/§16: never invent a price. This module is the actual price
source instance_catalog.py was waiting on; GPU specs (name/VRAM) don't come
from here -- they're already correct, static, and public in
instance_filter.SUPPORTED_GPU_INSTANCES, and unlike price they don't vary by
region, so there is no reason to fetch them at all.

The Bulk API has no server-side filtering: one region's index.json lists
every EC2 product for every OS/tenancy/instance type in that region (~200MB,
~65k products for ap-northeast-2, confirmed against a real downloaded copy).
There is no way to ask it for just g5.xlarge. A genuinely small per-instance
response needs the separate, authenticated AWS Price List *Query* API
(boto3's `pricing` client) instead -- deliberately not used here, so this
never requires AWS credentials just to look up a price. The cost of that
choice is paid once per region via the on-disk cache below, not on every job.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

from instance_filter import SUPPORTED_GPU_INSTANCES

logger = logging.getLogger("orchestrator")

PRICE_URL_TEMPLATE = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/{region}/index.json"
FETCH_TIMEOUT_SECONDS = 300

CACHE_DIR = Path(os.environ.get("AWS_PRICE_CACHE_DIR", Path(__file__).parent / ".aws_pricing_cache")).resolve()
# Prices don't change minute to minute; a day amortizes the ~200MB fetch
# without going so stale that a real price change takes too long to show up.
CACHE_TTL_SECONDS = int(os.environ.get("AWS_PRICE_CACHE_TTL_SECONDS", 24 * 3600))


class PriceFetchError(Exception):
    """Raised on a network/parse failure -- callers turn this into an
    honest 'lookup failed' state, never a guessed price."""


def _cache_path(region: str) -> Path:
    return CACHE_DIR / f"{region}.json"


def _fetch_region_prices(region: str) -> dict[str, float]:
    """instance_type -> hourly USD, for exactly the types in
    SUPPORTED_GPU_INSTANCES that this region's Linux/OnDemand/Shared-tenancy
    pricing actually lists. A type missing from the result means "not sold
    in this region", not "free" or "unavailable due to an error"."""
    url = PRICE_URL_TEMPLATE.format(region=region)
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        price_index = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise PriceFetchError(f"'{region}' 가격 데이터를 받아오지 못했습니다: {exc}") from exc

    wanted = set(SUPPORTED_GPU_INSTANCES)
    products = price_index.get("products", {})
    terms = price_index.get("terms", {}).get("OnDemand", {})

    prices: dict[str, float] = {}
    for sku, product in products.items():
        if product.get("productFamily") != "Compute Instance":
            continue
        attrs = product.get("attributes", {})
        instance_type = attrs.get("instanceType")
        if instance_type not in wanted or instance_type in prices:
            continue
        if attrs.get("operatingSystem") != "Linux":
            continue
        if attrs.get("tenancy") != "Shared":
            continue
        if attrs.get("preInstalledSw") != "NA":
            continue
        if attrs.get("capacitystatus") != "Used":
            continue
        license_model = attrs.get("licenseModel")
        if license_model not in (None, "No License required"):
            continue

        for term in terms.get(sku, {}).values():
            for dim in term.get("priceDimensions", {}).values():
                if dim.get("unit") == "Hrs":
                    usd = dim.get("pricePerUnit", {}).get("USD")
                    if usd not in (None, ""):
                        prices[instance_type] = float(usd)
                        break
            if instance_type in prices:
                break

    return prices


def _read_fresh_cache(region: str) -> dict[str, Any] | None:
    path = _cache_path(region)
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if time.time() - cached.get("fetched_at", 0) > CACHE_TTL_SECONDS:
        return None
    return cached


def _write_cache(region: str, prices: dict[str, float]) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = {
        "region": region,
        "fetched_at": time.time(),
        "source": PRICE_URL_TEMPLATE.format(region=region),
        "prices": prices,
    }
    _cache_path(region).write_text(json.dumps(cached), encoding="utf-8")
    return cached


def load_region_catalog(region: str) -> list[dict[str, Any]]:
    """[{"instance_type", "aws_region", "price_usd_per_hour"}, ...] for our
    known instance types. Uses the on-disk cache when it's still fresh;
    raises PriceFetchError (never guesses a price) if there's no usable
    cache and the live fetch also fails."""
    cached = _read_fresh_cache(region) or _write_cache(region, _fetch_region_prices(region))
    return [
        {"instance_type": instance_type, "aws_region": region, "price_usd_per_hour": price}
        for instance_type, price in cached["prices"].items()
    ]


def catalog_fetched_at(region: str) -> float | None:
    """None means "no fresh cache for this region yet" -- used by
    cache_key.py so a stale-vs-refreshed price catalog changes the pipeline
    cache key instead of silently reusing an old job's result."""
    cached = _read_fresh_cache(region)
    return cached["fetched_at"] if cached else None
