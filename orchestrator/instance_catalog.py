"""Loads the AWS instance/price catalog `instance_filter.filter_instances()`
needs, from an external JSON file this orchestrator does not own or invent.

CLAUDE_HANDOFF_CLOUDGPUFIT_EAGER_RUNTIME.md §8/§16 explicitly forbid
hardcoding an AWS price with no source. Until a teammate (or a real AWS
Price List API pull) supplies one, INSTANCE_CATALOG_PATH is unset, and
`load_instance_catalog()` returns `[]` -- `filter_instances()` then honestly
returns zero candidates instead of a guessed recommendation.

Input contract (INSTANCE_CATALOG_PATH points at a JSON file like this):

    [
      {"instance_type": "g6.xlarge",  "aws_region": "ap-northeast-2", "price_usd_per_hour": 1.006},
      {"instance_type": "g6e.xlarge", "aws_region": "ap-northeast-2", "price_usd_per_hour": null}
    ]

`price_usd_per_hour: null` means "available in this region, price unknown"
(instance_filter keeps it as a candidate and skips the budget check for it,
rather than treating an unknown price as either free or unaffordable).
An instance/region combination absent from the file means "availability
unknown" and is never offered as a candidate.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator")

CATALOG_PATH_ENV = "INSTANCE_CATALOG_PATH"
# Optional manual override for the cache-key ingredient (see cache_key.py) --
# useful if the catalog is fetched from a live API rather than a file, where
# mtime isn't meaningful. If unset, the catalog file's mtime is used instead.
CATALOG_VERSION_ENV = "INSTANCE_CATALOG_VERSION"


def load_instance_catalog() -> list[dict[str, Any]]:
    raw_path = os.environ.get(CATALOG_PATH_ENV)
    if not raw_path:
        logger.warning(
            "%s is not set -- no AWS instance/price catalog is available yet, "
            "so instance_filter will return zero candidates for every job "
            "until one is provided. This is the expected state until the "
            "team supplies real region/pricing data (see CLAUDE_HANDOFF "
            "§8: prices are never invented).",
            CATALOG_PATH_ENV,
        )
        return []

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


def catalog_version() -> str:
    """A stable identifier for "which catalog data produced this result",
    used as one ingredient of the pipeline cache key (CLAUDE_HANDOFF §18) so
    that updating prices/availability invalidates old cached job results."""
    override = os.environ.get(CATALOG_VERSION_ENV)
    if override:
        return override

    raw_path = os.environ.get(CATALOG_PATH_ENV)
    if not raw_path:
        return "none"

    path = Path(raw_path)
    try:
        return f"mtime:{path.stat().st_mtime_ns}"
    except OSError:
        return "missing"
