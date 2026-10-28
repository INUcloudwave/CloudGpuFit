"""CloudGPUFit payload -> eager_runtime adapter.

Validates a CloudGPUFit `PredictPayload`-shaped dict, then asks
eager_runtime for Eager or Compiled latency on the candidate instances.

Rules:
  * An instance with its own installed target is predicted directly. A size
    without one borrows its same-family target's prediction scaled by vCPU
    count (instance_filter FIXME(latency-proxy)); anything with neither is
    reported in `not_installed_targets`.
  * `engine_mode` ("eager" default, or "compiled") is passed to eager_runtime
    as its own argument, never smuggled inside the request dict.
  * audio_encoder (Wav2Vec2/HuBERT-style encoder-only) is supported.
    audio_encdec (Whisper-style encoder-decoder) is still refused: it is a
    different workload that eager_runtime has not been calibrated for.
  * mode: missing -> Generate for decoder_llm / text_encdec (Request E2E:
    Prefill + cache-aware Decode, max_new_tokens default 128), prefill/forward
    for vision_cls, audio_encoder, text_encoder. An explicit mode=prefill keeps
    the previous first-token / forward prediction (mode/max_new_tokens
    stripped, vCPU proxy unchanged).
  * Generate on a proxy size (no target of its own): the same-GPU base
    target's full Generate result is returned unscaled -- the Prefill vCPU
    ratio is not applied -- marked generate_proxy_status
    "unvalidated_same_gpu_base_target" with a warning.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping

from eager_runtime import available_targets, predict_runtime_targets
from instance_filter import supported_instances

SUPPORTED_WORKLOAD_TYPES = {"decoder_llm", "vision_cls", "text_encdec", "audio_encoder"}
SUPPORTED_ENGINE_MODES = {"eager", "compiled"}
GENERATE_WORKLOAD_TYPES = {"decoder_llm", "text_encdec"}
GENERATE_PROXY_STATUS = "unvalidated_same_gpu_base_target"
GENERATE_PROXY_WARNING = (
    "Generate latency uses the measured same-GPU base target without vCPU scaling. "
    "Host-size differences for this instance have not been validated."
)
_EXPLICITLY_UNSUPPORTED_TYPES = {
    "audio_encdec": (
        "audio_encdec (Whisper-style encoder-decoder) is not calibrated. Only "
        "audio_encoder (Wav2Vec2/HuBERT-style encoder-only) is supported, and "
        "this adapter does not silently remap one onto the other."
    ),
}


class UnsupportedRuntimeRequest(ValueError):
    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


def validate_engine_mode(engine_mode: str) -> None:
    if engine_mode not in SUPPORTED_ENGINE_MODES:
        raise UnsupportedRuntimeRequest(
            "unsupported_engine_mode",
            f"engine_mode must be one of {sorted(SUPPORTED_ENGINE_MODES)}, got {engine_mode!r}.",
        )


def validate_payload(payload: Mapping[str, Any]) -> None:
    dtype = str(payload.get("dtype", "")).lower()
    if dtype != "fp16":
        raise UnsupportedRuntimeRequest(
            "unsupported_dtype",
            f"Runtime prediction only runs fp16 requests, got dtype={dtype!r}.",
        )
    workload_type = payload.get("type")
    if effective_mode(payload) == "generate" and workload_type not in GENERATE_WORKLOAD_TYPES:
        raise UnsupportedRuntimeRequest(
            "unsupported_mode",
            f"mode=generate is only supported for {sorted(GENERATE_WORKLOAD_TYPES)}, "
            f"got type={workload_type!r}.",
        )
    if workload_type in _EXPLICITLY_UNSUPPORTED_TYPES:
        raise UnsupportedRuntimeRequest(
            "unsupported_workload_type", _EXPLICITLY_UNSUPPORTED_TYPES[workload_type]
        )
    if workload_type is not None and workload_type not in SUPPORTED_WORKLOAD_TYPES:
        raise UnsupportedRuntimeRequest(
            "unsupported_workload_type",
            f"type={workload_type!r} is not one of {sorted(SUPPORTED_WORKLOAD_TYPES)}.",
        )


def effective_mode(payload: Mapping[str, Any]) -> str:
    """Service policy: mode missing -> Generate for decoder_llm/text_encdec,
    forward/prefill for every other type; an explicit mode is kept."""
    params = payload.get("params") or {}
    mode = params.get("mode")
    if not mode:
        return "generate" if payload.get("type") in GENERATE_WORKLOAD_TYPES else "prefill"
    return str(mode).lower()


def validate_generate_inputs(payload: Mapping[str, Any]) -> None:
    """Generate input limits (batch 128, seq_len 1024, max_new_tokens 1024,
    default 128) -> 4xx-style rejection, never clamped."""
    from generate import GenerateInputError, resolve_generate_inputs

    try:
        resolve_generate_inputs(_runtime_request(payload))
    except GenerateInputError as exc:
        raise UnsupportedRuntimeRequest(exc.error_code, exc.message) from exc


def _runtime_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    # engine_mode is deliberately not part of the request: it is passed to
    # predict_runtime_targets() as its own argument, so there is exactly one
    # place that decides the mode.
    if effective_mode(payload) == "generate":
        # Generate Request E2E: mode/max_new_tokens are kept for eager_runtime.
        params = dict(payload.get("params") or {})
        params["mode"] = "generate"
    else:
        # mode/max_new_tokens are stripped: the prefill/forward path predicts
        # the prefill forward pass only.
        params = {
            key: value
            for key, value in (payload.get("params") or {}).items()
            if key not in {"mode", "max_new_tokens"}
        }
    return {
        "model": payload["model"],
        "type": payload.get("type"),
        "dtype": payload["dtype"],
        "batch_size": payload["batch_size"],
        "params": params,
        "model_config": payload.get("model_config_snapshot") or payload.get("model_config"),
    }


def predict_candidate_runtimes(
    payload: Mapping[str, Any],
    candidate_targets: Iterable[str],
    engine_mode: str = "eager",
) -> dict[str, Any]:
    """Never raises: always returns a structured dict."""
    candidate_list = list(dict.fromkeys(str(t) for t in candidate_targets))
    engine_mode = str(engine_mode or "eager")
    try:
        validate_engine_mode(engine_mode)
        validate_payload(payload)
        if effective_mode(payload) == "generate":
            validate_generate_inputs(payload)
    except UnsupportedRuntimeRequest as exc:
        return {
            "status": "skipped",
            "error_code": exc.error_code,
            "error_message": exc.message,
            "engine_mode": engine_mode,
            "requested_targets": candidate_list,
            "succeeded": 0,
            "failed": 0,
            "predictions": [],
        }

    installed = set(available_targets(engine_mode=engine_mode))
    specs = supported_instances(engine_mode)
    # candidate -> the target whose prediction answers it (itself, or the
    # same-family target it borrows).
    profile_for: dict[str, str] = {}
    for t in candidate_list:
        profile = t if t in installed else (specs.get(t) or {}).get("latency_proxy_of")
        if profile in installed:
            profile_for[t] = profile
    runnable = list(profile_for)
    not_installed = [t for t in candidate_list if t not in profile_for]
    if not runnable:
        code = "no_installed_targets" if candidate_list else "no_candidates"
        return {
            "status": code,
            "error_code": code,
            "error_message": (
                f"No candidate has an installed {engine_mode} target (missing: {not_installed})."
                if candidate_list
                else "The instance filter returned no candidates -- a normal result, not a runtime error."
            ),
            "engine_mode": engine_mode,
            "requested_targets": candidate_list,
            "not_installed_targets": not_installed,
            "succeeded": 0,
            "failed": 0,
            "predictions": [],
        }

    unique_profiles = list(dict.fromkeys(profile_for.values()))
    result = predict_runtime_targets(
        _runtime_request(payload), targets=unique_profiles, engine_mode=engine_mode
    )
    base_predictions = result.get("predictions", [])
    input_errors = {p.get("error_code") for p in base_predictions if p.get("error_code")}
    if base_predictions and len(input_errors) == 1 and all(p.get("error_code") for p in base_predictions):
        # Generate input rejected after the config was read (position limit,
        # model config unavailable) -> same 4xx-style skip as validate_payload
        first = base_predictions[0]
        return {
            "status": "skipped",
            "error_code": first["error_code"],
            "error_message": first.get("error"),
            "engine_mode": engine_mode,
            "requested_targets": candidate_list,
            "succeeded": 0,
            "failed": 0,
            "predictions": [],
        }
    by_profile = {p.get("target_instance"): p for p in base_predictions}
    predictions = []
    for target, profile in profile_for.items():
        base = by_profile.get(profile)
        if base is None:
            continue
        prediction = copy.deepcopy(base)
        if profile != target:
            prediction.update(instance_type=target, target_instance=target, latency_proxy_of=profile)
            # Generate Request E2E on a proxy size: the base target's full
            # result, unscaled (the Prefill vCPU ratio is not applied), so
            # median / P90 / throughput / breakdown stay consistent.
            generate_proxy = (
                prediction.get("status") == "ok"
                and prediction.get("prediction_scope") == "generate_request_e2e"
            )
            if generate_proxy:
                prediction["generate_proxy_status"] = GENERATE_PROXY_STATUS
                prediction["warnings"] = [*(prediction.get("warnings") or []), GENERATE_PROXY_WARNING]
            if prediction.get("status") == "ok" and not generate_proxy:
                scale = specs[target]["latency_scale"]
                for key in ("predicted_median_ms", "raw_predicted_median_ms", "predicted_p90_ms"):
                    if prediction.get(key) is not None:
                        prediction[key] *= scale
                # throughput = batch_size / median, so it scales inversely.
                if prediction.get("predicted_throughput_rps") is not None:
                    prediction["predicted_throughput_rps"] /= scale
        predictions.append(prediction)
    succeeded = sum(p.get("status") == "ok" for p in predictions)
    failed = len(predictions) - succeeded
    status = "ok" if succeeded == len(predictions) else ("partial" if succeeded else "error")
    return {
        "status": status,
        "model": result.get("model", ""),
        "engine_mode": engine_mode,
        "requested_targets": candidate_list,
        "not_installed_targets": not_installed,
        "succeeded": succeeded,
        "failed": failed,
        "predictions": predictions,
    }
