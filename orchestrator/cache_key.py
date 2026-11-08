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
# "2": remove the fixed 2 GiB margin; invalidate old result semantics.
VRAM_PREDICTOR_VERSION = "4-native-inference-single-request-outside-bd902817a45c2b20"


def vram_code_fingerprint():
    root=Path(__file__).resolve().parents[1]
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ["predict_vram.py","cpu_tensor_trace.py","inference_trace_runtime.py","vram_policy.py","gpu_overheads.json"]}

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
# "3": engine_mode became a real request field (Compiled/Inductor predictor
# added), and eager_runtime_manifest_fingerprint()'s output string shape
# changed (now includes the manifest filename, and covers each target's
# Compiled manifests too, not just runtime_manifest.json) -- both change
# what a cache-hit would represent, so bump even though caching is inert.
# "4": candidates are only instances with their own target (the 25-size vCPU
# proxy and latency_scale_for_vcpu are gone), audio_encoder accepted, and
# compiled_context_calibration.json is fingerprinted.
# "5": Generate Request E2E (Prefill + cache-aware Decode; mode default
# generate for decoder_llm/text_encdec), Generate artifacts + code +
# torch/transformers versions fingerprinted.
RUNTIME_PIPELINE_VERSION = "5"


def eager_runtime_manifest_fingerprint(eager_runtime_root: Path) -> str:
    """Fingerprints exactly which target manifests are installed and their
    contents, so retraining/replacing a target's PKL or calibration factor
    invalidates old cached results automatically."""
    targets_root = eager_runtime_root / "targets"
    if not targets_root.is_dir():
        return "no-targets"

    # Eager manifests (targets/<instance>/runtime_manifest.json) plus, when
    # present, each target's Compiled manifests -- so a Compiled-mode job's
    # cache key also changes if its LightGBM external/fusion models are
    # retrained, exactly like the pre-existing Eager-only coverage below.
    # compiled_context_calibration.json (g7, g7e) is not referenced by any
    # manifest, so it has to be hashed here explicitly or a recalibration
    # would leave old cached Compiled results valid.
    manifest_globs = (
        "*/runtime_manifest.json",
        "*/compiled_external_model/compiled_external_manifest.json",
        "*/compiled_fusion_model/compiled_fusion_manifest.json",
        "*/compiled_context_calibration.json",
        # Generate artifacts (generate_specialist_profile.json and the
        # Generate host model), each pinned by sha256 in this manifest.
        "*/generate_manifest.json",
    )
    fingerprints = []
    for pattern in manifest_globs:
        for manifest_path in sorted(targets_root.glob(pattern)):
            try:
                digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
            except OSError:
                digest = "unreadable"
            relative = manifest_path.relative_to(targets_root).as_posix()
            fingerprints.append(f"{relative}:{digest}")
    return ",".join(fingerprints) or "no-manifests"


def generate_code_fingerprint(eager_runtime_root: Path) -> str:
    """eager_runtime/generate/*.py contents: a Generate code change changes
    predictions without touching any target manifest."""
    parts = []
    for path in sorted((eager_runtime_root / "generate").glob("*.py")):
        parts.append(f"{path.name}:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}")
    return ",".join(parts) or "no-generate"


def runtime_library_versions() -> dict[str, str]:
    """Generate graphs are torch.export'ed with the installed torch/transformers."""
    try:
        import torch
        import transformers

        return {"torch": torch.__version__, "transformers": transformers.__version__}
    except ImportError:
        return {"torch": "missing", "transformers": "missing"}


def pipeline_cache_key_for(
    xmem_args: dict[str, Any],
    payload: dict[str, Any],
    *,
    eager_runtime_root: Path,
    catalog_version: str,
) -> str:
    material = {
        "vram_predictor_version": VRAM_PREDICTOR_VERSION,
        "vram_code": vram_code_fingerprint(),
        "runtime_pipeline_version": RUNTIME_PIPELINE_VERSION,
        "eager_runtime_manifests": eager_runtime_manifest_fingerprint(eager_runtime_root),
        "generate_code": generate_code_fingerprint(eager_runtime_root),
        "runtime_libraries": runtime_library_versions(),
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
        # Eager vs. Compiled predicts a genuinely different result for the
        # same inputs -- folded in explicitly rather than assuming xmem_args
        # or the other payload fields above happen to change too.
        "engine_mode": payload.get("engine_mode") or "eager",
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
