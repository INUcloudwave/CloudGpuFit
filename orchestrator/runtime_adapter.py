"""CloudGPUFit payload -> eager_runtime adapter.

Validates a CloudGPUFit `PredictPayload`-shaped dict, then asks
eager_runtime for Eager or Compiled latency on the candidate instances.

Rules:
  * Only instances with their own installed target are predicted. There is
    no proxy and no vCPU scaling -- a candidate without a target is reported
    in `not_installed_targets`, never answered with another instance's numbers.
  * `engine_mode` ("eager" default, or "compiled") is passed to eager_runtime
    as its own argument, never smuggled inside the request dict.
  * audio_encoder (Wav2Vec2/HuBERT-style encoder-only) is supported.
    audio_encdec (Whisper-style encoder-decoder) is still refused: it is a
    different workload that eager_runtime has not been calibrated for.
  * mode=generate (decoder_llm / text_encdec) is accepted, but latency is
    predicted for the prefill pass only (up to the first token's logits) --
    eager_runtime has no decode-loop model. mode only changes VRAM (KV cache).
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from eager_runtime import available_targets, predict_runtime_targets

SUPPORTED_WORKLOAD_TYPES = {"decoder_llm", "vision_cls", "text_encdec", "audio_encoder"}
SUPPORTED_ENGINE_MODES = {"eager", "compiled"}
GENERATE_WORKLOAD_TYPES = {"decoder_llm", "text_encdec"}
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
    if _mode(payload) == "generate" and workload_type not in GENERATE_WORKLOAD_TYPES:
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


def _mode(payload: Mapping[str, Any]) -> str:
    params = payload.get("params") or {}
    return str(params.get("mode") or "prefill").lower()


def _runtime_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    # engine_mode is deliberately not part of the request: it is passed to
    # predict_runtime_targets() as its own argument, so there is exactly one
    # place that decides the mode.
    # mode/max_new_tokens are stripped: eager_runtime only ever predicts the
    # prefill forward pass, and refuses mode=generate outright.
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
    runnable = [t for t in candidate_list if t in installed]
    not_installed = [t for t in candidate_list if t not in installed]
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

    result = predict_runtime_targets(
        _runtime_request(payload), targets=runnable, engine_mode=engine_mode
    )
    predictions = result.get("predictions", [])
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
