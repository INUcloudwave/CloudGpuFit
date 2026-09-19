"""CloudGPUFit -> eager_runtime adapter.

Validates/normalizes a CloudGPUFit `PredictPayload`-shaped dict against the
Eager FP16 runtime predictor's supported range (CLAUDE_HANDOFF
§3, §4, §14), then calls `eager_runtime.predict_runtime_targets()` for the
intersection of the caller's candidate targets (usually
`instance_filter.filter_instances()`'s output) and the targets
`eager_runtime` actually has an installed profile for.

Runtime 전용 HTTP 서버는 만들지 않는다 (§1) — 같은 프로세스에서 eager_runtime을
직접 import한다. 이 모듈이 하는 일은 오직: (1) CloudGPUFit payload를
Runtime이 받는 형태로 정규화, (2) Runtime이 실행할 수 없는 조합을 미리
걸러서 명확한 이유와 함께 거부, (3) 설치된 target과의 교집합 계산,
(4) predict_runtime_targets() 호출과 결과를 그대로 전달.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from eager_runtime import available_targets, predict_runtime_targets

# eager_runtime이 실제로 지원 확정한 workload (§4). audio_encdec
# (Whisper 계열 encoder-decoder)은 eager_runtime의 audio_encoder
# (Wav2Vec2/HuBERT 계열 encoder-only)와 절대 같은 것으로 취급하지 않고
# 명시적으로 거부한다 — 조용히 잘못된 어댑터로 매핑하지 않는다.
_SUPPORTED_CLOUDGPUFIT_TYPES = {"decoder_llm", "vision_cls", "text_encdec"}
_EXPLICITLY_UNSUPPORTED_TYPES = {
    "audio_encdec": (
        "audio_encdec (e.g. Whisper-style encoder-decoder) is not supported by "
        "the Eager runtime predictor. Only audio_encoder (Wav2Vec2/HuBERT-style "
        "encoder-only) is calibrated, and this adapter refuses to silently "
        "remap one workload onto the other."
    ),
}


class UnsupportedRuntimeRequest(ValueError):
    """Raised (internally) for a payload runtime_adapter refuses outright,
    before ever calling eager_runtime -- wrong dtype/mode/type. Never
    escapes predict_candidate_runtimes(); it is turned into the same
    structured-error shape eager_runtime itself returns."""

    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


def _validate_payload(payload: Mapping[str, Any]) -> None:
    dtype = str(payload.get("dtype", "")).lower()
    if dtype != "fp16":
        raise UnsupportedRuntimeRequest(
            "unsupported_dtype",
            f"Eager runtime predictor only runs fp16 requests, got dtype={dtype!r}. "
            "fp32/bf16 requests get a VRAM-only result, no runtime estimate.",
        )

    params = payload.get("params") or {}
    mode = str(params.get("mode") or "prefill").lower()
    if mode != "prefill":
        raise UnsupportedRuntimeRequest(
            "unsupported_mode",
            f"Eager runtime predictor only runs mode=prefill, got mode={mode!r}. "
            "generate is out of scope for Runtime prediction here.",
        )

    workload_type = payload.get("type")
    if workload_type in _EXPLICITLY_UNSUPPORTED_TYPES:
        raise UnsupportedRuntimeRequest(
            "unsupported_workload_type",
            _EXPLICITLY_UNSUPPORTED_TYPES[workload_type],
        )
    if workload_type is not None and workload_type not in _SUPPORTED_CLOUDGPUFIT_TYPES:
        raise UnsupportedRuntimeRequest(
            "unsupported_workload_type",
            f"type={workload_type!r} is not one of {sorted(_SUPPORTED_CLOUDGPUFIT_TYPES)}.",
        )


def _runtime_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    # eager_runtime only reads model/type/dtype/batch_size/params/model_config.
    # deployment and warnings are dropped here explicitly rather than relying
    # on eager_runtime to ignore unknown keys.
    return {
        "model": payload["model"],
        "type": payload.get("type"),
        "dtype": payload["dtype"],
        "batch_size": payload["batch_size"],
        "params": payload.get("params") or {},
        # load_config() reads this key name and skips its own
        # AutoConfig.from_pretrained() fetch when it's present.
        "model_config": payload.get("model_config_snapshot"),
    }


def predict_candidate_runtimes(
    payload: Mapping[str, Any],
    candidate_targets: Iterable[str],
) -> dict[str, Any]:
    """Run the Eager FP16 runtime predictor for every candidate target that
    `eager_runtime` has an installed profile for.

    Never raises: always returns an orchestrator-shaped dict, including for
    a refusal, so `process_job()` can store `result["runtime"]` unconditionally.
    """
    candidate_list = list(dict.fromkeys(str(t) for t in candidate_targets))

    try:
        _validate_payload(payload)
    except UnsupportedRuntimeRequest as exc:
        return {
            "status": "skipped",
            "error_code": exc.error_code,
            "error_message": exc.message,
            "requested_targets": candidate_list,
            "succeeded": 0,
            "failed": 0,
            "predictions": [],
        }

    installed = set(available_targets())
    runnable_targets = [t for t in candidate_list if t in installed]
    not_installed = [t for t in candidate_list if t not in installed]

    if not runnable_targets:
        return {
            "status": "no_installed_targets" if candidate_list else "no_candidates",
            "error_code": "no_installed_targets" if candidate_list else "no_candidates",
            "error_message": (
                f"None of the candidate targets have an installed eager_runtime "
                f"profile (missing: {not_installed})."
                if candidate_list
                else "instance_filter returned no candidate instances -- "
                "this is a normal filter result, not a Runtime error."
            ),
            "requested_targets": candidate_list,
            "not_installed_targets": not_installed,
            "succeeded": 0,
            "failed": 0,
            "predictions": [],
        }

    request = _runtime_request(payload)
    result = dict(predict_runtime_targets(request, targets=runnable_targets))
    result["not_installed_targets"] = not_installed
    return result
