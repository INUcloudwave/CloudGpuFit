"""Loads the AWS instance/price catalog `instance_filter.filter_instances()`
needs, for whichever region the user picked.

Default path is aws_pricing.py -- a live fetch (cached on disk) against the
public AWS Price List Bulk API, per instance type/region, never a guessed
price (CLAUDE_HANDOFF §8/§16). INSTANCE_CATALOG_PATH remains as a manual
override for a hand-curated or offline catalog file, e.g.:

    [
      {"instance_type": "g6.xlarge",  "aws_region": "ap-northeast-2", "price_usd_per_hour": 1.006},
      {"instance_type": "g6e.xlarge", "aws_region": "ap-northeast-2", "price_usd_per_hour": null}
    ]

`price_usd_per_hour: null` means "available in this region, price unknown"
(instance_filter keeps it as a candidate and skips the budget check for it,
rather than treating an unknown price as either free or unaffordable).
An instance/region combination absent from the catalog means "availability
unknown" and is never offered as a candidate -- whether that catalog came
from the override file or the live fetch.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from aws_pricing import PriceFetchError, catalog_fetched_at, load_region_catalog

logger = logging.getLogger("orchestrator")

CATALOG_PATH_ENV = "INSTANCE_CATALOG_PATH"
# Optional manual override for the cache-key ingredient (see cache_key.py) --
# useful if the catalog is fetched from a live API rather than a file, where
# mtime isn't meaningful. If unset, the catalog file's mtime is used instead.
CATALOG_VERSION_ENV = "INSTANCE_CATALOG_VERSION"


def load_instance_catalog(aws_region: str | None = None) -> list[dict[str, Any]]:
    raw_path = os.environ.get(CATALOG_PATH_ENV)
    if raw_path:
        return _load_override_file(raw_path)

    if not aws_region:
        return []

    try:
        return load_region_catalog(aws_region)
    except PriceFetchError as exc:
        logger.warning("AWS 가격 조회 실패: %s", exc)
        raise


def _load_override_file(raw_path: str) -> list[dict[str, Any]]:
    path = Path(raw_path)
    if not path.is_file():
        logger.warning("%s points at a missing file: %s", CATALOG_PATH_ENV, path)
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read/parse %s (%s): %s", CATALOG_PATH_ENV, path, exc)
        return []

    if not isinstance(data, list):
        logger.warning(
            "%s must contain a JSON array of catalog rows, got %s",
            CATALOG_PATH_ENV,
            type(data).__name__,
        )
        return []

    return data


def catalog_version(aws_region: str | None = None) -> str:
    """A stable identifier for "which catalog data produced this result",
    used as one ingredient of the pipeline cache key (CLAUDE_HANDOFF §18) so
    that updating prices/availability invalidates old cached job results."""
    override = os.environ.get(CATALOG_VERSION_ENV)
    if override:
        return override

    raw_path = os.environ.get(CATALOG_PATH_ENV)
    if raw_path:
        try:
            return f"mtime:{Path(raw_path).stat().st_mtime_ns}"
        except OSError:
            return "missing"

    if not aws_region:
        return "none"
    fetched_at = catalog_fetched_at(aws_region)
    return f"{aws_region}:{fetched_at}" if fetched_at is not None else f"{aws_region}:not-cached-yet"
