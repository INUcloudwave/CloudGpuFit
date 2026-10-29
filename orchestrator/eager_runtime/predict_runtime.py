#!/usr/bin/env python3
"""Unified public entry point for eager_runtime.

This module is the package's real front door (see ``__init__.py``): it wraps
both the Eager (``predict_eager_runtime``) and Compiled
(``predict_compiled_runtime``) predictors behind one ``engine_mode``-aware
API, while staying call-compatible with the pre-Compiled public surface that
CloudGPUFit's ``orchestrator/runtime_adapter.py`` and ``main.py`` already use
(``predict_runtime(request, target=...)``, ``predict_runtime_targets(request,
targets=[...])``, ``available_targets()`` -- all with zero knowledge of
``engine_mode``, which defaults to ``"eager"`` so that existing calls are
byte-for-byte unaffected).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

# Sibling modules in this package (predict_eager_runtime, predict_compiled_runtime,
# compiled_*) import each other with bare, non-relative names (e.g. `import
# runtime_manifest`), the same convention this whole codebase already uses for
# its CLI entry points. That only resolves when eager_runtime/'s own directory
# is on sys.path. predict_eager_runtime.py bootstraps this for itself before
# its own sibling imports, but this module is the package's __init__.py entry
# point and gets imported *first* when someone does `import eager_runtime` (or
# `from eager_runtime import ...`) from outside this directory -- at that
# point nothing has inserted eager_runtime/ into sys.path yet, so the bare
# imports below would fail with ModuleNotFoundError. Do the same bootstrap
# here, before importing any sibling module, so package-style import works
# regardless of import order or the caller's own working directory.
_MODULE_ROOT = Path(__file__).resolve().parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

# The predictors print progress lines with emoji/Korean. On a Windows console
# stdout defaults to the ANSI codepage (cp949 on Korean Windows), where those
# prints raise UnicodeEncodeError and every prediction comes back as an error.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import predict_compiled_runtime
import predict_eager_runtime
from compiled_external_model import external_model_root
from compiled_fusion_model import fusion_model_root


DEFAULT_ENGINE_MODE = "eager"
SUPPORTED_ENGINE_MODES = {"eager", "compiled"}
DEFAULT_TARGET = "g6.xlarge"

# Provenance of a target's CPU/memory/dispatch overhead profile. This is
# metadata only (how trustworthy the numbers are), never a separate
# prediction path: every target -- full or size-profile-built -- is predicted
# by exactly the same code from its own validated runtime_manifest.json.
PROVENANCE_STATUSES = ("measured", "interpolated", "extrapolated")

# What the predicted latency covers, per workload. Generative models stop at
# the logits of the first generated token (prefill + LM head on the last
# position; no decode loop); the others are one full forward pass.
PREDICTION_SCOPES = {
    "decoder_llm": "first_token_logits",
    "text_encdec": "first_token_logits",
    "vision_cls": "classification_forward",
    "text_encoder": "encoder_forward",
    "audio_encoder": "audio_encoder_forward",
}

# Keys every successful result carries, in both engine modes, so a caller can
# read Eager and Compiled results with the same code (requirement: same
# interface and same result format for both modes).
_P90_STATUS = {
    # Eager applies a per-target global conformal P90 factor
    # (p90_calibration_factor.json).
    "eager": "conformal_calibrated",
    # Compiled P90 is still diagnostic only -- median is the official number.
    "compiled": "diagnostic",
}


GENERATE_WORKLOAD_TYPES = {"decoder_llm", "text_encdec"}


def _is_generate_request(payload: Mapping[str, Any]) -> bool:
    """Generate only on an explicit params.mode == "generate" for text
    workloads. The service default (mode missing -> Generate for
    decoder_llm/text_encdec) is applied by the caller's adapter, so a request
    without mode keeps this module's previous behaviour."""
    params = payload.get("params") or {}
    return (
        payload.get("type") in GENERATE_WORKLOAD_TYPES
        and str(params.get("mode") or "").lower() == "generate"
    )


def _with_generate_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Request with model_config filled in.

    model_config_snapshot / model_config given -> used as is. Missing -> the
    existing Prefill loader (predict_eager_runtime.load_config, i.e.
    AutoConfig.from_pretrained(model, trust_remote_code=False)), called once;
    the dict is then shared by every target, the Graph extraction and the
    Prefill fallback (which therefore does not fetch it again)."""
    request = dict(payload)
    if request.get("model_config") is None:
        config = predict_eager_runtime.load_config(str(request.get("model", "")), request)
        request["model_config"] = config.to_dict()
        request["model_config_source"] = "autoconfig"
    return request


def _predict_generate(
    payload: Mapping[str, Any],
    targets: list[str],
    engine_mode: str,
    raise_on_error: bool,
) -> list[dict[str, Any]]:
    """Generate Request E2E for several targets (graphs extracted once)."""
    from generate import GenerateInputError, predict_generate_targets

    def prefill(request: Mapping[str, Any], target: str, mode: str) -> dict[str, Any]:
        return predict_runtime(request, target_instance=target, engine_mode=mode, raise_on_error=raise_on_error)

    model_name = str(payload.get("model", ""))
    try:
        request = _with_generate_config(payload)
    except Exception as exc:  # noqa: BLE001 - network / gated / unknown model
        if raise_on_error:
            raise
        return [
            {
                **_error_result(error=exc, model=model_name, instance_type=target, engine_mode=engine_mode),
                "error_code": "model_config_unavailable",
            }
            for target in targets
        ]
    config_source = request.get("model_config_source", "model_config_snapshot")
    try:
        results = predict_generate_targets(
            request, targets, engine_mode=engine_mode, runtime_root=_MODULE_ROOT, prefill_predict=prefill
        )
    except GenerateInputError as exc:
        if raise_on_error:
            raise
        return [
            {
                **_error_result(error=exc, model=model_name, instance_type=target, engine_mode=engine_mode),
                "error_code": exc.error_code,
            }
            for target in targets
        ]
    for result in results:
        target = result["target_instance"]
        result["engine_mode"] = engine_mode
        result["instance_type"] = target
        result.setdefault("model", model_name)
        result.setdefault("workload_type", payload.get("type"))
        result.setdefault("dtype", payload.get("dtype"))
        result["model_config_source"] = config_source
        if result.get("status") == "ok":
            result["profile_provenance"] = target_provenance(target)
    return results


def _compiled_artifacts_present(runtime_root: Path, target: str) -> bool:
    """Cheap presence check (file existence only, no hash/size verification --
    that's what ``verify_runtime_artifacts.py`` is for) used by
    ``available_targets(engine_mode="compiled")``. Mirrors the same
    lightweight style as ``predict_eager_runtime.available_targets()``, which
    only checks for ``runtime_manifest.json``.
    """
    external_manifest = external_model_root(runtime_root, target) / "compiled_external_manifest.json"
    fusion_manifest = fusion_model_root(runtime_root, target) / "compiled_fusion_manifest.json"
    return external_manifest.is_file() and fusion_manifest.is_file()


def available_targets(
    engine_mode: str | None = None,
    runtime_root: str | Path | None = None,
) -> list[str]:
    """List target instances with an installed profile for ``engine_mode``.

    ``engine_mode=None`` (the default) preserves the pre-Compiled behavior
    that CloudGPUFit's ``main.py``/``instance_filter.py`` already call
    (``available_targets()`` with no arguments) -- it defers entirely to
    ``predict_eager_runtime.available_targets()``. Passing
    ``engine_mode="compiled"`` instead lists only targets that have both a
    ``compiled_external_model`` and ``compiled_fusion_model`` manifest. Adding
    a new ``targets/{instance_type}/`` directory (full artifacts, or a thin one
    generated by ``build_size_target.py``) is picked up automatically in both
    modes -- no code change needed.
    """
    selected_mode = str(engine_mode or DEFAULT_ENGINE_MODE).lower()
    if selected_mode not in SUPPORTED_ENGINE_MODES:
        raise ValueError(f"engine_mode must be one of {sorted(SUPPORTED_ENGINE_MODES)}")

    if selected_mode == "eager":
        return predict_eager_runtime.available_targets(runtime_root)

    root = Path(runtime_root).expanduser().resolve() if runtime_root else _MODULE_ROOT
    targets_root = root / "targets"
    if not targets_root.is_dir():
        return []
    return sorted(
        path.name
        for path in targets_root.iterdir()
        if path.is_dir() and _compiled_artifacts_present(root, path.name)
    )


def target_provenance(
    instance_type: str,
    runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    """Where a target's CPU/memory/dispatch overhead numbers come from.

    A target built from a size profile (``build_size_target.py``) records a
    ``size_profile`` block in its runtime_manifest.json; that block is
    returned as-is. Every other target is a full, directly calibrated
    artifact set, i.e. ``measured`` on that instance itself.
    """
    root = Path(runtime_root).expanduser().resolve() if runtime_root else _MODULE_ROOT
    manifest_path = root / "targets" / instance_type / "runtime_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "unknown", "source": "missing_runtime_manifest"}
    size_profile = manifest.get("size_profile")
    if isinstance(size_profile, Mapping):
        return {
            "status": str(size_profile.get("provenance_status", "unknown")),
            "gpu_base_target": size_profile.get("gpu_base_target"),
            "source": "size_profile",
            "size_profile": size_profile.get("profile_path"),
            "fields": size_profile.get("field_provenance", {}),
        }
    return {
        "status": "measured",
        "gpu_base_target": instance_type,
        "source": "full_target_artifacts",
    }


def _p90_status(engine_mode: str, instance_type: str) -> str:
    if engine_mode == "compiled":
        return _P90_STATUS["compiled"]
    path = _MODULE_ROOT / "targets" / instance_type / "p90_calibration_factor.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    if (
        payload.get("method") == "global_conformal_ratio"
        and int(payload.get("calibration_rows") or 0) > 0
    ):
        return _P90_STATUS["eager"]
    return "uncalibrated"


def _error_result(
    *,
    error: Exception,
    model: str,
    instance_type: str,
    engine_mode: str,
) -> dict[str, Any]:
    return {
        "status": "error",
        "model": model,
        "instance_type": instance_type,
        "target_instance": instance_type,
        "engine_mode": engine_mode,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _request_from_keywords(
    *,
    model: str | None,
    model_type: str | None,
    model_config: Mapping[str, Any] | None,
    dtype: str | None,
    batch_size: int | None,
    sequence_length: int | None,
    decoder_sequence_length: int | None,
    resolution: int | None,
    audio_samples: int | None,
) -> dict[str, Any]:
    """Build the canonical request dict from the keyword-style call
    ``predict_runtime(model=..., instance_type=..., batch_size=...,
    sequence_length=...)``. Only fields the caller actually passed are set;
    everything else keeps the predictors' own defaults."""
    request: dict[str, Any] = {}
    if model is not None:
        request["model"] = model
    if model_type is not None:
        request["type"] = model_type
    if model_config is not None:
        request["model_config"] = dict(model_config)
    if dtype is not None:
        request["dtype"] = dtype
    if batch_size is not None:
        request["batch_size"] = batch_size
    params: dict[str, Any] = {}
    if sequence_length is not None:
        # text_encdec reads encoder_seq_len and falls back to seq_len, so one
        # key serves every text workload.
        params["seq_len"] = sequence_length
    if decoder_sequence_length is not None:
        params["decoder_seq_len"] = decoder_sequence_length
    if resolution is not None:
        params["resolution"] = resolution
    if audio_samples is not None:
        params["audio_samples"] = audio_samples
    if params:
        request["params"] = params
    return request


def _normalize_result(
    result: dict[str, Any],
    *,
    engine_mode: str,
    instance_type: str,
) -> dict[str, Any]:
    """Stamp the fields shared by both engine modes so Eager and Compiled
    results can be consumed by the same code."""
    result["engine_mode"] = engine_mode
    result["instance_type"] = instance_type
    result["target_instance"] = instance_type
    if result.get("status") == "ok":
        result.setdefault(
            "raw_predicted_median_ms", result.get("predicted_median_ms")
        )
        result["official_estimate"] = "predicted_median_ms"
        result["p90_status"] = _p90_status(engine_mode, instance_type)
        if result["p90_status"] == "uncalibrated":
            result.setdefault("warnings", []).append(
                "Eager P90 for this target has no E2E calibration data yet "
                "(identity factor); use predicted_median_ms."
            )
        result["profile_provenance"] = target_provenance(instance_type)
        result["prediction_scope"] = PREDICTION_SCOPES.get(
            str(result.get("workload_type")), "unknown"
        )
    return result


def predict_runtime(
    request: Mapping[str, Any] | None = None,
    target: str = DEFAULT_TARGET,
    *,
    engine_mode: str | None = None,
    instance_type: str | None = None,
    target_instance: str | None = None,
    model: str | None = None,
    model_type: str | None = None,
    model_config: Mapping[str, Any] | None = None,
    dtype: str | None = None,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    decoder_sequence_length: int | None = None,
    resolution: int | None = None,
    audio_samples: int | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Predict runtime for one target instance under one engine mode.

    Two equivalent call styles::

        predict_runtime(request, "g6.xlarge")                     # request dict
        predict_runtime(model="facebook/opt-125m",                 # keywords
                        instance_type="g6.xlarge",
                        engine_mode="compiled",
                        batch_size=8, sequence_length=128)

    Target selection, highest priority first: ``instance_type=`` /
    ``target_instance=`` keyword, the request's own ``instance_type`` /
    ``target_instance`` key, then the positional ``target`` (kept for exact
    call-compatibility with ``predict_eager_runtime.predict_runtime(request,
    target=...)``, which CloudGPUFit's orchestrator uses). ``engine_mode``
    likewise falls back to the request's key and finally to ``"eager"``, so an
    unmodified pre-Compiled call behaves exactly as before.

    A target is usable only if ``targets/<instance_type>/`` holds a valid
    artifact set for that mode; anything else is rejected up front with the
    list of installed targets instead of silently falling back to another
    GPU's artifacts.
    """
    payload = dict(request or {})
    keyword_request = _request_from_keywords(
        model=model,
        model_type=model_type,
        model_config=model_config,
        dtype=dtype,
        batch_size=batch_size,
        sequence_length=sequence_length,
        decoder_sequence_length=decoder_sequence_length,
        resolution=resolution,
        audio_samples=audio_samples,
    )
    keyword_params = keyword_request.pop("params", None)
    payload.update(keyword_request)
    if keyword_params:
        payload["params"] = {**dict(payload.get("params") or {}), **keyword_params}

    request_mode = payload.pop("engine_mode", None)
    request_instance_type = payload.pop("instance_type", None)
    request_target_instance = payload.pop("target_instance", None)
    selected_mode = str(
        engine_mode or request_mode or DEFAULT_ENGINE_MODE
    ).lower()
    selected_target = str(
        instance_type
        or target_instance
        or request_instance_type
        or request_target_instance
        or target
    )
    model_name = str(payload.get("model", ""))

    if selected_mode not in SUPPORTED_ENGINE_MODES:
        raise ValueError(
            f"engine_mode must be one of {sorted(SUPPORTED_ENGINE_MODES)}"
        )

    installed = available_targets(engine_mode=selected_mode)
    if selected_target not in installed:
        error = LookupError(
            f"instance_type={selected_target!r} has no installed "
            f"{selected_mode} artifacts under targets/. "
            f"Installed for {selected_mode}: {installed}"
        )
        if raise_on_error:
            raise error
        return _error_result(
            error=error,
            model=model_name,
            instance_type=selected_target,
            engine_mode=selected_mode,
        )

    if _is_generate_request(payload):
        return _predict_generate(payload, [selected_target], selected_mode, raise_on_error)[0]

    predictor = (
        predict_eager_runtime.predict_runtime
        if selected_mode == "eager"
        else predict_compiled_runtime.predict_runtime
    )
    # Graph tracing (FX for Prefill/forward, torch.export for Generate) patches
    # process-global torch state; concurrent traces corrupt it for the rest of
    # the process. Every graph-building prediction in the process is therefore
    # serialized on one RLock -- the one predict_eager_runtime already holds
    # for its target environment, and generate.export_graph.EXPORT_LOCK.
    # Values are unchanged; only concurrent calls run one at a time.
    with predict_eager_runtime._RUNTIME_LOCK:
        result = predictor(
            payload,
            target=selected_target,
            raise_on_error=raise_on_error,
        )
    return _normalize_result(
        result,
        engine_mode=selected_mode,
        instance_type=selected_target,
    )


def predict_runtime_targets(
    request: Mapping[str, Any],
    targets: Iterable[str],
    *,
    engine_mode: str | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Predict the same request across several target instances, one engine
    mode for all of them.

    Kept call-compatible with CloudGPUFit's real usage in
    ``runtime_adapter.py``: ``predict_runtime_targets(request,
    targets=unique_profiles)``, no ``engine_mode`` passed -> defaults to
    Eager, identical to the pre-Compiled function it replaces. Mirrors
    ``predict_eager_runtime.predict_runtime_targets()``'s response shape
    (``status``/``model``/``predictions``, with ``"partial"`` when only some
    targets succeed) so callers do not need to special-case the dispatcher.
    """
    if not isinstance(request, Mapping):
        error = TypeError("request must be a mapping")
        if raise_on_error:
            raise error
        return {
            "status": "error",
            "model": "",
            "predictions": [],
            "error_type": type(error).__name__,
            "error": str(error),
        }

    selected_targets = list(dict.fromkeys(str(t) for t in targets))
    if not selected_targets:
        error = ValueError("targets must contain at least one target instance")
        if raise_on_error:
            raise error
        return {
            "status": "error",
            "model": str(request.get("model", "")),
            "predictions": [],
            "error_type": type(error).__name__,
            "error": str(error),
        }

    selected_mode = str(engine_mode or request.get("engine_mode") or DEFAULT_ENGINE_MODE).lower()
    if _is_generate_request(request) and selected_mode in SUPPORTED_ENGINE_MODES:
        installed = set(available_targets(engine_mode=selected_mode))
        runnable = [t for t in selected_targets if t in installed]
        generated = iter(_predict_generate(request, runnable, selected_mode, raise_on_error) if runnable else [])
        predictions = [
            next(generated) if target in installed
            else predict_runtime(request, target_instance=target, engine_mode=engine_mode, raise_on_error=raise_on_error)
            for target in selected_targets
        ]
    else:
        predictions = [
            predict_runtime(
                request,
                target_instance=target,
                engine_mode=engine_mode,
                raise_on_error=raise_on_error,
            )
            for target in selected_targets
        ]
    succeeded = sum(item.get("status") == "ok" for item in predictions)
    if succeeded == len(predictions):
        status = "ok"
    elif succeeded:
        status = "partial"
    else:
        status = "error"
    return {
        "status": status,
        "model": str(request.get("model", "")),
        "predictions": predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path)
    source.add_argument("--json")
    parser.add_argument(
        "--engine-mode",
        choices=sorted(SUPPORTED_ENGINE_MODES),
        default=None,
    )
    parser.add_argument(
        "--instance-type",
        "--target",
        dest="instance_type",
        default=None,
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = (
        args.request.read_text(encoding="utf-8")
        if args.request is not None
        else args.json
    )
    request = json.loads(raw)
    if not isinstance(request, Mapping):
        raise ValueError("request must be a JSON object")
    result = predict_runtime(
        request,
        engine_mode=args.engine_mode,
        instance_type=args.instance_type,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            default=str,
        )
    )
    raise SystemExit(0 if result.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
