#!/usr/bin/env python3
"""Predict target-GPU Eager latency from a CloudGPUFit-style JSON request."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

_MODULE_ROOT = Path(__file__).resolve().parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

import torch
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

import fxgraph_conv1d_v10 as v10
import fxgraph_eager_final as final
import run_eager_local as local
from runtime_manifest import (
    DEFAULT_TARGET,
    load_manifest,
    manifest_for_target,
    validate_manifest,
)


RUNTIME_ROOT = _MODULE_ROOT
SUPPORTED_BATCH_SIZES = {1, 8, 16, 32, 64, 128}
TOKEN_BYTES = 4
AUDIO_SAMPLES_PER_CHUNK = 30 * 16000
_RUNTIME_LOCK = RLock()
_TARGET_ENVIRONMENT_NAMES = {
    "EAGER_TARGET_ROOT",
    "EAGER_TARGET_INSTANCE",
    "EAGER_P90_CALIBRATION_ACTIVE",
    "EAGER_HOST_SCHEDULE_SCALE",
    "EAGER_HOST_SCHEDULE_POLICY",
}


def available_targets(runtime_root: str | Path | None = None) -> list[str]:
    root = Path(runtime_root).expanduser().resolve() if runtime_root else RUNTIME_ROOT
    targets_root = root / "targets"
    if not targets_root.is_dir():
        return []
    return sorted(
        path.name
        for path in targets_root.iterdir()
        if path.is_dir() and (path / "runtime_manifest.json").is_file()
    )


def resolve_manifest_path(
    target: str,
    manifest_path: str | Path | None,
) -> Path:
    if manifest_path is not None:
        return Path(manifest_path).expanduser().resolve()
    return manifest_for_target(RUNTIME_ROOT, target)


@contextmanager
def runtime_context(manifest_path: Path):
    manifest_preview = load_manifest(manifest_path)
    environment_names = _TARGET_ENVIRONMENT_NAMES | set(
        manifest_preview.get("environment", {})
    )

    with _RUNTIME_LOCK:
        previous_environment = {
            name: os.environ.get(name)
            for name in environment_names
        }
        original_cwd = Path.cwd()
        try:
            for name in environment_names:
                os.environ.pop(name, None)
            os.chdir(RUNTIME_ROOT)
            yield
        finally:
            os.chdir(original_cwd)
            for name, value in previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def config_int(config: Any, *names: str, default: int | None = None) -> int:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    if default is not None:
        return default
    raise ValueError(f"Config is missing a positive value for: {', '.join(names)}")


def load_config(model_id: str, payload: Mapping[str, Any]):
    raw = payload.get("model_config")
    if raw is None:
        return AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    if not isinstance(raw, Mapping):
        raise ValueError("model_config must be a full config.json object")
    model_type = raw.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("model_config.model_type is required")
    try:
        config_class = CONFIG_MAPPING[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported model_type in model_config: {model_type}") from exc
    return config_class.from_dict(dict(raw))


def workload_type_for(
    config: Any,
    model: torch.nn.Module,
    requested: str | None,
) -> str:
    architectures = [
        str(value).lower()
        for value in (getattr(config, "architectures", None) or [])
    ]
    architecture_text = " ".join(architectures)
    input_names = set(inspect.signature(model.forward).parameters)

    if "pixel_values" in input_names:
        inferred = "vision_cls"
    elif "input_values" in input_names:
        inferred = "audio_encoder"
    elif "input_features" in input_names:
        if getattr(config, "is_encoder_decoder", False) or "decoder_input_ids" in input_names:
            raise ValueError(
                "Audio encoder-decoder models are not supported yet; "
                "the current audio adapter supports encoder-only input_values models"
            )
        inferred = "audio_encoder"
    elif getattr(config, "is_encoder_decoder", False) or "decoder_input_ids" in input_names:
        inferred = "text_encdec"
    elif "input_ids" in input_names:
        causal_hint = any(
            token in architecture_text
            for token in ("forcausallm", "causallm", "lmheadmodel")
        )
        inferred = (
            "decoder_llm"
            if causal_hint or getattr(config, "is_decoder", False)
            else "text_encoder"
        )
    else:
        model_type = str(getattr(config, "model_type", "unknown"))
        names = ", ".join(sorted(input_names)) or "none"
        raise ValueError(
            f"Cannot infer workload adapter for model_type={model_type!r}; "
            f"forward inputs: {names}"
        )

    if requested and requested != inferred:
        model_type = str(getattr(config, "model_type", "unknown"))
        raise ValueError(
            f"Request type={requested!r} conflicts with model_type={model_type!r} "
            f"({inferred!r})"
        )
    return inferred


def token_input(batch_size: int, sequence: int, vocab_size: int) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.zeros(
            batch_size,
            sequence,
            dtype=torch.long,
            device=local.ANALYSIS_DEVICE,
        )
    }


def decoder_input(
    batch_size: int,
    encoder_sequence: int,
    decoder_sequence: int,
    vocab_size: int,
) -> dict[str, torch.Tensor]:
    return {
        "input_ids": token_input(batch_size, encoder_sequence, vocab_size)["input_ids"],
        "decoder_input_ids": token_input(batch_size, decoder_sequence, vocab_size)["input_ids"],
    }


def audio_output_length(config: Any, input_samples: int) -> int:
    strides = getattr(config, "conv_stride", None)
    if not isinstance(strides, (list, tuple)) or not strides:
        return max(1, input_samples // 320)
    total_stride = 1
    for stride in strides:
        total_stride *= int(stride)
    return max(1, input_samples // total_stride)


def build_dynamic_model_config(
    payload: Mapping[str, Any],
) -> tuple[str, str, dict[str, object], list[str], dict[str, int]]:
    model_id = str(payload.get("model", "")).strip()
    if not model_id:
        raise ValueError("model is required")

    dtype = str(payload.get("dtype", "fp16")).lower()
    if dtype != "fp16":
        raise ValueError("The current Eager runtime profiles are calibrated only for fp16")

    batch_size = positive_int(payload.get("batch_size", 1), "batch_size")
    params = payload.get("params") or {}
    if not isinstance(params, Mapping):
        raise ValueError("params must be an object")
    mode = str(params.get("mode") or "prefill").lower()
    if mode != "prefill":
        raise ValueError("generate mode is not supported by the Eager runtime predictor")

    config = load_config(model_id, payload)
    config.use_cache = False
    config.torch_dtype = torch.float16
    with torch.device(local.ANALYSIS_DEVICE):
        base_model = AutoModel.from_config(
            config,
            trust_remote_code=False,
        ).half()
    requested_type = payload.get("type")
    requested = str(requested_type) if requested_type else None
    workload_type = workload_type_for(config, base_model, requested)
    model_type = str(config.model_type).lower()
    warnings: list[str] = []
    normalized: dict[str, int] = {"batch_size": batch_size}

    if batch_size not in SUPPORTED_BATCH_SIZES:
        warnings.append(
            "batch_size is outside the validated set {1, 8, 16, 32, 64, 128}"
        )

    model_factory: Callable[[], torch.nn.Module]
    dummy_fn: Callable[[int], torch.Tensor | dict[str, torch.Tensor]]
    bytes_fn: Callable[[int, int], tuple[int, int]]

    if workload_type in {"text_encoder", "decoder_llm"}:
        vocab_size = config_int(config, "vocab_size", default=32000)
        hidden_size = config_int(config, "hidden_size", "d_model", "n_embd")
        sequence = positive_int(params.get("seq_len", 128), "params.seq_len")
        normalized["seq_len"] = sequence
        if sequence != 128:
            warnings.append("seq_len differs from the validated value 128")
        model_factory = lambda: base_model
        dummy_fn = lambda bs: token_input(bs, sequence, vocab_size)
        bytes_fn = lambda bs, precision: (
            bs * sequence * TOKEN_BYTES,
            bs * sequence * hidden_size * precision,
        )
    elif workload_type == "text_encdec":
        vocab_size = config_int(config, "vocab_size", default=32000)
        hidden_size = config_int(config, "hidden_size", "d_model", "n_embd")
        encoder_sequence = positive_int(
            params.get("encoder_seq_len", params.get("seq_len", 128)),
            "params.encoder_seq_len",
        )
        decoder_sequence = positive_int(
            params.get("decoder_seq_len", 128),
            "params.decoder_seq_len",
        )
        normalized.update(
            encoder_seq_len=encoder_sequence,
            decoder_seq_len=decoder_sequence,
        )
        if encoder_sequence != 128 or decoder_sequence != 128:
            warnings.append(
                "encoder_seq_len or decoder_seq_len differs from the validated value 128"
            )
        model_factory = lambda: base_model
        dummy_fn = lambda bs: decoder_input(
            bs,
            encoder_sequence,
            decoder_sequence,
            vocab_size,
        )
        bytes_fn = lambda bs, precision: (
            bs * (encoder_sequence + decoder_sequence) * TOKEN_BYTES,
            bs * decoder_sequence * hidden_size * precision,
        )
    elif workload_type == "audio_encoder":
        hidden_size = config_int(config, "hidden_size", "d_model")
        if params.get("audio_samples") is not None:
            audio_samples = positive_int(params["audio_samples"], "params.audio_samples")
        else:
            audio_chunks = params.get("audio_chunks")
            audio_samples = (
                positive_int(audio_chunks, "params.audio_chunks") * AUDIO_SAMPLES_PER_CHUNK
                if audio_chunks is not None
                else 32000
            )
        normalized["audio_samples"] = audio_samples
        if audio_samples != 32000:
            warnings.append("audio_samples differs from the validated value 32000")
        output_length = audio_output_length(config, audio_samples)
        model_factory = lambda: base_model
        dummy_fn = lambda bs: {
            "input_values": torch.empty(
                bs,
                audio_samples,
                dtype=torch.float16,
                device=local.ANALYSIS_DEVICE,
            )
        }
        bytes_fn = lambda bs, precision: (
            bs * audio_samples * precision,
            bs * output_length * hidden_size * precision,
        )
    elif workload_type == "vision_cls":
        resolution = positive_int(
            params.get("resolution", getattr(config, "image_size", 224)),
            "params.resolution",
        )
        normalized["resolution"] = resolution
        if resolution != 224:
            warnings.append("resolution differs from the validated value 224")
        num_labels = config_int(config, "num_labels", default=1000)
        model_factory = lambda: AutoModelForImageClassification.from_config(
            config,
            trust_remote_code=False,
        ).half()
        dummy_fn = lambda bs: {
            "pixel_values": torch.empty(
                bs,
                3,
                resolution,
                resolution,
                dtype=torch.float16,
                device=local.ANALYSIS_DEVICE,
            )
        }
        bytes_fn = lambda bs, precision: (
            bs * 3 * resolution * resolution * precision,
            bs * num_labels * precision,
        )
    else:
        raise AssertionError(f"Unhandled workload type: {workload_type}")

    trace_name = "Wav2Vec2" if workload_type == "audio_encoder" else model_id
    runtime_config: dict[str, object] = {
        "is_hf": True,
        "load_fn": model_factory,
        "dummy_fn": dummy_fn,
        "bytes_fn": bytes_fn,
    }
    runtime_config["hf_model_type"] = model_type
    return trace_name, workload_type, runtime_config, warnings, normalized


def _predict_or_raise(
    payload: Mapping[str, Any],
    manifest_path: Path,
    target: str,
) -> dict[str, Any]:
    trace_name, workload_type, runtime_config, warnings, normalized = (
        build_dynamic_model_config(payload)
    )
    model_id = str(payload["model"])
    batch_size = normalized["batch_size"]

    manifest = validate_manifest(
        manifest_path,
        expected_target=target,
        check_gpu=False,
    )
    os.environ["EAGER_P90_CALIBRATION_ACTIVE"] = "1"
    local.enable_unicode_safe_lightgbm_loading()
    parser = v10.Conv1dContextFXGraphParser()
    router = final.CalibratedEagerSimulatorRouter(engine_mode="eager")
    rows = local.predict_model(
        model_name=trace_name,
        config=runtime_config,
        batch_sizes=[batch_size],
        parser=parser,
        router=router,
        construction_device="meta",
    )
    row = json_safe(rows[0])
    row["model"] = model_id
    hf_model_type = str(runtime_config["hf_model_type"])
    predicted_median_ms = float(row["predicted_e2e_ms"])

    # TODO(throughput): naive single-stream estimate -- one batch of
    # `batch_size` samples completes every `predicted_median_ms`, back-to-back,
    # with nothing else sharing the GPU. Ignores request queuing/batch-formation
    # wait time, multiple concurrent streams, and steady-state GPU utilization
    # effects (e.g. thermal/clock throttling under sustained load), so it's an
    # upper bound, not a measured number. Replace with a real concurrent-load
    # benchmark once one exists.
    predicted_throughput_rps = batch_size / (predicted_median_ms / 1000.0)

    return {
        "status": "ok",
        "model": model_id,
        "model_type": hf_model_type,
        "workload_type": workload_type,
        "target_instance": manifest["target_instance"],
        "target_gpu": manifest["gpu_name"],
        "engine_mode": "eager",
        "dtype": "fp16",
        "batch_size": batch_size,
        "params": {key: value for key, value in normalized.items() if key != "batch_size"},
        "predicted_median_ms": predicted_median_ms,
        "predicted_p90_ms": float(row["predicted_p90_ms"]),
        "predicted_throughput_rps": predicted_throughput_rps,
        "warnings": warnings,
        "breakdown": row,
    }


def predict_runtime(
    request: Mapping[str, Any],
    target: str = DEFAULT_TARGET,
    *,
    manifest_path: str | Path | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Predict Eager latency for one target instance."""
    if not isinstance(request, Mapping):
        error = TypeError("request must be a mapping")
        if raise_on_error:
            raise error
        return {
            "status": "error",
            "target_instance": target,
            "error_type": type(error).__name__,
            "error": str(error),
        }

    selected_manifest = resolve_manifest_path(target, manifest_path)
    try:
        with runtime_context(selected_manifest):
            return _predict_or_raise(request, selected_manifest, target)
    except Exception as exc:
        if raise_on_error:
            raise
        return {
            "status": "error",
            "model": str(request.get("model", "")),
            "target_instance": target,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def predict_runtime_targets(
    request: Mapping[str, Any],
    targets: Iterable[str],
    *,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Predict the same request for multiple target instances."""
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

    selected_targets = list(dict.fromkeys(str(target) for target in targets))
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

    predictions = [
        predict_runtime(
            request,
            target,
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
        "requested_targets": selected_targets,
        "succeeded": succeeded,
        "failed": len(predictions) - succeeded,
        "predictions": predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--json")
    source.add_argument("--request", type=Path)
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Target profile directory under targets/ (default: g6.xlarge)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional explicit manifest path; normally selected by --target",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_payload(args: argparse.Namespace) -> Mapping[str, Any]:
    raw = args.json if args.json is not None else args.request.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("The request JSON must be an object")
    return payload


def main() -> None:
    args = parse_args()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else None
    )
    try:
        payload = load_payload(args)
        result = predict_runtime(
            payload,
            args.target,
            manifest_path=args.manifest,
        )
        exit_code = 0 if result.get("status") == "ok" else 1
    except Exception as exc:
        result = {
            "status": "error",
            "target_instance": args.target,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        exit_code = 1

    output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if output_path:
        output_path.write_text(output + "\n", encoding="utf-8")
    print(output)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
