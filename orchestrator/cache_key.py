"""Cache key for the combined VRAM + instance-filter + Runtime pipeline.

CLAUDE_HANDOFF_CLOUDGPUFIT_EAGER_RUNTIME.md §18: the old cache key
(xmem_runner.cache_key_for) hashed only the xMem input args. That is no
longer enough now that a cached "succeeded" job also carries `candidates`
and `runtime` -- if predict_vram.py, an eager_runtime target's artifacts, or
the instance catalog change, a stale VRAM-only-era result must never be
served back as if it already included those.

pipeline_cache_key_for() folds in every ingredient CLAUDE_HANDOFF §18 lists:
  - VRAM predictor version
  - Runtime predictor version
  - installed target/manifest identifiers (fingerprinted, not hand-versioned)
  - instance catalog / price data version
  - the input payload
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Bump by hand whenever predict_vram.py's output shape/semantics change in a
# way that should invalidate previously cached jobs.
VRAM_PREDICTOR_VERSION = "1"

# Bump by hand whenever runtime_adapter.py / instance_filter.py's own logic
# changes in a way that should invalidate previously cached jobs. This is
# for OUR adapter code -- changes to the target artifacts themselves are
# already covered by eager_runtime_manifest_fingerprint() below, with no
# manual bump needed.
#
# "2" reflects the 29-instance vCPU-scaling proxy added to
# instance_filter.py/runtime_adapter.py -- bumped after the fact because the
# cache-hit path (main.py's create_job()) was, at the time, still serving
# pre-existing cached rows computed by the OLD adapter logic even after that
# logic changed. The cache-hit lookup itself is now disabled in
# create_job() for the same reason (every job re-runs the full pipeline), so
# this version number is currently inert, but keep bumping it on adapter
# changes anyway in case caching is ever turned back on.
RUNTIME_PIPELINE_VERSION = "2"


def eager_runtime_manifest_fingerprint(eager_runtime_root: Path) -> str:
    """Fingerprints exactly which target manifests are installed and their
    contents, so retraining/replacing a target's PKL or calibration factor
    invalidates old cached results automatically."""
    targets_root = eager_runtime_root / "targets"
    if not targets_root.is_dir():
        return "no-targets"

    fingerprints = []
    for manifest_path in sorted(targets_root.glob("*/runtime_manifest.json")):
        try:
            digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
        except OSError:
            digest = "unreadable"
        fingerprints.append(f"{manifest_path.parent.name}:{digest}")
    return ",".join(fingerprints) or "no-manifests"


def pipeline_cache_key_for(
    xmem_args: dict[str, Any],
    payload: dict[str, Any],
    *,
    eager_runtime_root: Path,
    catalog_version: str,
) -> str:
    material = {
        "vram_predictor_version": VRAM_PREDICTOR_VERSION,
        "runtime_pipeline_version": RUNTIME_PIPELINE_VERSION,
        "eager_runtime_manifests": eager_runtime_manifest_fingerprint(eager_runtime_root),
        "instance_catalog_version": catalog_version,
        "xmem_args": xmem_args,
        # xmem_args already carries model/batch_size/params; deployment and
        # type feed instance_filter/runtime_adapter but not xmem, so they're
        # folded in separately rather than invisibly reusing the old key.
        "type": payload.get("type"),
        "deployment": payload.get("deployment"),
        # xmem_args already includes this too (build_xmem_args copies it in),
        # but folded in again explicitly so a config change is never missed
        # even if that copy step changes later.
        "model_config_snapshot": payload.get("model_config_snapshot"),
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
