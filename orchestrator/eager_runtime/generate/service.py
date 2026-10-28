"""Generate Request E2E prediction service (decoder_llm / text_encdec).

Flow per request (shared by every target):
  1. validate the Generate inputs (4xx on violation, never clamped)
  2. build the real config + meta model from the request's model_config
     (model_config_snapshot, or -- when the caller sent none --
     AutoConfig.from_pretrained resolved once by predict_runtime._with_generate_config)
  3. torch.export the Prefill graph, then the Decode graphs at the minimal
     canonical cache anchors (see select_anchors)
  4. per target: route each phase graph through the target's existing Eager or
     Compiled predictors (median, then the existing P90 models for the
     diagnostic P90)
  5. aggregate with the bundle v35 Request E2E formula

Graph extraction or routing failure -> explicit Prefill fallback (caller's
existing Prefill predictor), with the failure reason. A target without Generate
artifacts -> the same explicit Prefill fallback (generate_target_status =
validation_pending_forward_graph). All 7 installed targets now have them
(gr6.4xlarge since its v42 checkpoint validation).
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import transformers

from . import export_graph as eg
from . import routing

CANONICAL_ANCHORS = (1, 128, 512, 2048, 4095)
# Eager: opt-in batch API of the common routers (Generate only). Set
# GENERATE_EAGER_BATCH=0 to use the scalar simulate_fx_graph path.
EAGER_BATCH_ENABLED = os.environ.get("GENERATE_EAGER_BATCH", "1") != "0"
INPUT_LIMITS = {"batch_size": (1, 128), "seq_len": (1, 1024), "max_new_tokens": (1, 1024)}
DEFAULT_MAX_NEW_TOKENS = 128
GENERATE_WORKLOAD_TYPES = {eg.CAUSAL_WORKLOAD, eg.SEQ2SEQ_WORKLOAD}
GENERATE_ARTIFACTS = ("generate_specialist_profile.json", "generate_eager_host_model")
GENERATE_MANIFEST = "generate_manifest.json"

P90_STATUS = "diagnostic_uncalibrated"
P90_WARNING = (
    "Generate P90 is a diagnostic value from the existing per-operation P90 models "
    "(Generate host time uses the median host model); its 90% coverage has not been validated."
)
EXPERIMENTAL_WARNING = (
    "Generate prediction succeeded, but this architecture has not been validated against measured runtime."
)
FALLBACK_WARNING = (
    "Generate graph extraction failed; only Prefill latency is predicted "
    "(Decode and max_new_tokens are not reflected)."
)
ROUTING_FALLBACK_WARNING = (
    "Generate graph routing failed; only Prefill latency is predicted "
    "(Decode and max_new_tokens are not reflected)."
)
GR6_STYLE_WARNING = (
    "{target} Generate is temporarily estimated with the existing Prefill predictor; "
    "Decode and max_new_tokens are not reflected."
)
DECODER_SEQ_LEN_WARNING = (
    "decoder_seq_len is kept for API compatibility but not used by Generate: the decoder starts "
    "from one start token and runs max_new_tokens steps."
)
PENDING_TARGET_STATUS = "validation_pending_forward_graph"
# metadata: graphs are extracted with config._attn_implementation = "eager"
# (export_graph.build_config), and `validated` means "same architecture
# signature as an E2E-measured reference", not a per-model measurement.
ATTENTION_IMPLEMENTATION = eg.ATTENTION_IMPLEMENTATION
VALIDATION_SCOPE = "architecture_signature"


class GenerateInputError(ValueError):
    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


def runtime_versions() -> dict[str, str]:
    return {"torch": torch.__version__, "transformers": transformers.__version__}


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def resolve_generate_inputs(request: dict[str, Any]) -> dict[str, Any]:
    """Validated (batch, seq_len, max_new_tokens). Raises GenerateInputError."""
    workload = request.get("type")
    if workload not in GENERATE_WORKLOAD_TYPES:
        raise GenerateInputError("unsupported_mode", f"Generate is only for {sorted(GENERATE_WORKLOAD_TYPES)}")
    if str(request.get("dtype", "")).lower() != "fp16":
        raise GenerateInputError("unsupported_dtype", "Generate prediction only runs fp16 requests")
    params = dict(request.get("params") or {})
    key = "encoder_seq_len" if workload == eg.SEQ2SEQ_WORKLOAD else "seq_len"
    values = {
        "batch_size": request.get("batch_size"),
        "seq_len": params.get(key),
        "max_new_tokens": params.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
    }
    if values["max_new_tokens"] is None:
        values["max_new_tokens"] = DEFAULT_MAX_NEW_TOKENS
    for name, value in values.items():
        low, high = INPUT_LIMITS[name]
        label = key if name == "seq_len" else name
        if isinstance(value, bool) or not isinstance(value, int):
            raise GenerateInputError("invalid_generate_input", f"{label} must be an integer, got {value!r}")
        if not low <= value <= high:
            raise GenerateInputError("generate_input_out_of_range", f"{label}={value} is outside [{low}, {high}]")
    values["decoder_seq_len_ignored"] = workload == eg.SEQ2SEQ_WORKLOAD and params.get("decoder_seq_len") is not None
    return values


def check_position_limits(config: Any, workload: str, seq_len: int, max_new_tokens: int) -> None:
    limit = getattr(config, "max_position_embeddings", None)
    if limit is None:
        return
    limit = int(limit)
    if workload == eg.CAUSAL_WORKLOAD:
        if seq_len + max_new_tokens - 1 > limit:
            raise GenerateInputError(
                "position_limit_exceeded",
                f"seq_len + max_new_tokens - 1 = {seq_len + max_new_tokens - 1} > max_position_embeddings={limit}",
            )
        return
    if seq_len > limit:
        raise GenerateInputError("position_limit_exceeded", f"encoder_seq_len={seq_len} > max_position_embeddings={limit}")
    if max_new_tokens > limit:
        raise GenerateInputError("position_limit_exceeded", f"decoder length {max_new_tokens} > max_position_embeddings={limit}")


def decode_cache_range(workload: str, seq_len: int, max_new_tokens: int) -> tuple[int, int] | None:
    """First/last Decode cache length; None when max_new_tokens == 1."""
    if max_new_tokens <= 1:
        return None
    start = 1 if workload == eg.SEQ2SEQ_WORKLOAD else seq_len
    return start, start + max_new_tokens - 2


def select_anchors(cache_range: tuple[int, int] | None) -> list[int]:
    """Minimal canonical anchors that give the same piecewise-linear sum as all
    five: the anchor at/below the first length, the anchor at/above the last,
    and every canonical anchor in between. No extrapolation."""
    if cache_range is None:
        return []
    first, last = cache_range
    if first < CANONICAL_ANCHORS[0] or last > CANONICAL_ANCHORS[-1]:
        raise GenerateInputError(
            "cache_anchor_out_of_range", f"decode cache range {first}..{last} is outside {CANONICAL_ANCHORS[0]}..{CANONICAL_ANCHORS[-1]}"
        )
    low = bisect.bisect_right(CANONICAL_ANCHORS, first) - 1
    high = bisect.bisect_left(CANONICAL_ANCHORS, last)
    return list(CANONICAL_ANCHORS[low:high + 1])


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------
_VALIDATED_TARGETS: dict[str, bool] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_generate_artifacts(runtime_root: Path, target: str) -> bool:
    root = runtime_root / "targets" / target
    return all((root / name).exists() for name in GENERATE_ARTIFACTS) and (root / GENERATE_MANIFEST).is_file()


def validate_generate_target(runtime_root: Path, target: str) -> None:
    if _VALIDATED_TARGETS.get(target):
        return
    root = runtime_root / "targets" / target
    manifest = json.loads((root / GENERATE_MANIFEST).read_text(encoding="utf-8"))
    errors = []
    for item in manifest.get("artifacts", []):
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"missing {item['path']}")
        elif path.stat().st_size != int(item["size_bytes"]) or _sha256(path) != item["sha256"]:
            errors.append(f"hash/size mismatch {item['path']}")
    if manifest.get("target_instance") != target:
        errors.append(f"manifest target {manifest.get('target_instance')!r} != {target!r}")
    if errors:
        raise RuntimeError(f"Generate artifacts for {target} failed validation: {errors}")
    _VALIDATED_TARGETS[target] = True


# --------------------------------------------------------------------------
# caches (GUIDE section 11)
# --------------------------------------------------------------------------
class _LRU(OrderedDict):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def put(self, key, value):
        """Insert (or refresh) with eviction; the only way entries are added."""
        if key in self:
            self.move_to_end(key)
        self[key] = value
        while len(self) > self.size:
            self.popitem(last=False)
        return value

    def get_or(self, key, factory):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return self.put(key, factory())


_GRAPH_CACHE = _LRU(24)        # (fingerprint, workload, batch, seq_len, phase, anchor, dtype, torch, transformers)
_PHASE_CACHE = _LRU(4096)      # + engine_mode, target, prediction_mode
_FAILURE_CACHE = _LRU(256)     # graph extraction failures (fingerprint, workload, batch, seq_len)
_EAGER_ROUTERS: dict[str, Any] = {}
_COMPILED_ROUTERS: dict[str, Any] = {}


def _graph_key(fingerprint, workload, batch, seq_len, phase, anchor):
    versions = runtime_versions()
    return (fingerprint, workload, batch, seq_len, phase, anchor, "fp16", versions["torch"], versions["transformers"])


def clear_caches() -> None:
    for cache in (_GRAPH_CACHE, _PHASE_CACHE, _FAILURE_CACHE):
        cache.clear()


_UNICODE_LGB_LOCK = threading.Lock()
_UNICODE_LGB_ENABLED = False


def _unicode_safe_lightgbm() -> None:
    """Same loader the existing Prefill/Compiled paths enable before building
    routers (run_eager_local.enable_unicode_safe_lightgbm_loading): the model
    text is read by Python and passed as model_str, because LightGBM's C
    loader cannot open model_file paths with non-ASCII characters on Windows.
    Booster contents are identical either way. Applied once per process
    (lock + flag); the existing function itself is reused unchanged."""
    global _UNICODE_LGB_ENABLED
    if _UNICODE_LGB_ENABLED:
        return
    with _UNICODE_LGB_LOCK:
        if _UNICODE_LGB_ENABLED:
            return
        import run_eager_local

        run_eager_local.enable_unicode_safe_lightgbm_loading()
        _UNICODE_LGB_ENABLED = True


def _eager_router(runtime_root: Path, target: str):
    if target not in _EAGER_ROUTERS:
        _unicode_safe_lightgbm()
        with routing.eager_target_environment(runtime_root, target):
            _EAGER_ROUTERS[target] = routing.ExportGenerateEagerRouter()
    return _EAGER_ROUTERS[target]


def _compiled_router(runtime_root: Path, target: str):
    if target not in _COMPILED_ROUTERS:
        _unicode_safe_lightgbm()
        _COMPILED_ROUTERS[target] = routing.CompiledGenerateRouter(runtime_root=runtime_root, target=target)
    return _COMPILED_ROUTERS[target]


# --------------------------------------------------------------------------
# graph extraction (target independent)
# --------------------------------------------------------------------------
class ExtractedRequest:
    def __init__(self, extracted: eg.ExtractedModel, decode: dict[int, eg.PhaseGraph], fingerprint: str):
        self.extracted = extracted
        self.decode = decode
        self.fingerprint = fingerprint


def extract_request_graphs(snapshot: dict, workload: str, batch: int, seq_len: int, anchors: list[int]) -> ExtractedRequest:
    fingerprint = eg.config_fingerprint(snapshot)
    failure_key = (fingerprint, workload, batch, seq_len, tuple(anchors))
    if failure_key in _FAILURE_CACHE:
        raise _FAILURE_CACHE[failure_key]
    try:
        extracted = _GRAPH_CACHE.get_or(
            _graph_key(fingerprint, workload, batch, seq_len, "prefill", 0),
            lambda: eg.extract_prefill(snapshot, workload, batch, seq_len),
        )
        decode = {
            anchor: _GRAPH_CACHE.get_or(
                _graph_key(fingerprint, workload, batch, seq_len, "decode", anchor),
                lambda anchor=anchor: eg.extract_decode(extracted, anchor),
            )
            for anchor in anchors
        }
    except eg.GenerateGraphError as exc:
        _FAILURE_CACHE.get_or(failure_key, lambda: exc)
        raise
    return ExtractedRequest(extracted, decode, fingerprint)


# --------------------------------------------------------------------------
# per-target phase prediction
# --------------------------------------------------------------------------
def _predict_phase(runtime_root, target, engine, graphs: ExtractedRequest, phase: eg.PhaseGraph, mode: str):
    key = (
        graphs.fingerprint, graphs.extracted.workload_type, graphs.extracted.batch, graphs.extracted.seq_len,
        phase.phase, phase.cache_len, engine, target, mode, "fp16", *runtime_versions().values(),
    )

    def run():
        if engine == "eager":
            router = _eager_router(runtime_root, target)
            with routing.eager_target_environment(runtime_root, target):
                both = routing.predict_eager_modes(
                    router, phase.graph_module, graphs.extracted.config, graphs.extracted.batch,
                    phase=phase.phase, seq_len=graphs.extracted.seq_len, cache_len=phase.cache_len,
                    prediction_modes=("median", "90th"), lifted_constants=phase.lifted_constants,
                    use_batch=EAGER_BATCH_ENABLED,
                )
            # median and P90 share the routed graph / feature rows; cache both
            for other_mode, other_key_mode in (("median", "median"), ("90th", "p90")):
                other_key = key[:8] + (other_key_mode,) + key[9:]
                if other_key != key and other_key not in _PHASE_CACHE:
                    _PHASE_CACHE.put(other_key, both[other_mode])
            return both["median" if mode == "median" else "90th"]
        router = _compiled_router(runtime_root, target)
        return routing.predict_compiled(
            router, phase.graph_module, "median" if mode == "median" else "p90",
            lifted_constants=phase.lifted_constants,
        )

    result = _PHASE_CACHE.get_or(key, run)
    unsupported = result.get("unsupported_operations") or []
    if unsupported:
        raise eg.GenerateGraphError("routing", f"{engine} unsupported operations at {phase.phase}:{phase.cache_len}: {unsupported}")
    if engine == "eager" and int(result.get("host_model_applied", 0)) != 1:
        raise eg.GenerateGraphError("routing", f"Generate Eager host model not applied for {target}")
    return result


def _interpolate(cache_len: int, anchors: list[tuple[int, float]]) -> float:
    # identical to bundle aggregate_generate_request_e2e_v35.interpolate
    from .aggregate_generate_request_e2e_v35 import interpolate

    return interpolate(cache_len, anchors)


def aggregate_request(prefill_ms, decode_anchor_ms, *, workload, batch, seq_len, max_new_tokens, hardware):
    """Bundle v35 aggregate() formula for one engine and one max_new_tokens;
    first decode cache length from the workload (Seq2Seq = 1)."""
    cache_range = decode_cache_range(workload, seq_len, max_new_tokens)
    anchors = sorted(decode_anchor_ms.items())
    decode_values = [] if cache_range is None else [
        _interpolate(length, anchors) for length in range(cache_range[0], cache_range[1] + 1)
    ]
    decode_ms = float(sum(decode_values))
    input_bytes = batch * 2 * seq_len * 8
    output_bytes = batch * max_new_tokens * 8
    h2d_ms = float(hardware["t_h2d_base"]) + input_bytes / (float(hardware["bw_h2d_GBs"]) * 1_000_000_000) * 1000
    d2h_ms = float(hardware["t_d2h_base"]) + output_bytes / (float(hardware["bw_d2h_GBs"]) * 1_000_000_000) * 1000
    sync_ms = float(hardware["t_sync"])
    generate_ms = float(prefill_ms) + decode_ms
    return {
        "prefill_ms": float(prefill_ms),
        "decode_ms": decode_ms,
        "decode_steps": max_new_tokens - 1,
        "kernel_and_host_generate_ms": generate_ms,
        "h2d_ms": h2d_ms,
        "d2h_ms": d2h_ms,
        "final_sync_ms": sync_ms,
        "predicted_request_inference_ms": generate_ms + h2d_ms + d2h_ms + sync_ms,
        "first_decode_cache_length": cache_range[0] if cache_range else None,
        "last_decode_cache_length": cache_range[1] if cache_range else None,
        "cache_anchors_used": [a for a, _ in anchors],
    }


def _hardware(runtime_root: Path, target: str) -> dict[str, Any]:
    return json.loads((runtime_root / "targets" / target / "hardware_profile.json").read_text(encoding="utf-8"))


def _validation_status(extracted: eg.ExtractedModel) -> tuple[str, dict]:
    from .architecture import validation_status

    layout = extracted.prefill.layout
    return validation_status(extracted.config, cache_tensors_per_layer=layout[0] if layout else None,
                             adaptations=extracted.adaptations)


def predict_target(runtime_root, target, engine, graphs: ExtractedRequest, inputs: dict, anchors: list[int]) -> dict:
    extracted = graphs.extracted
    phases = {}
    for mode in ("median", "p90"):
        prefill = _predict_phase(runtime_root, target, engine, graphs, extracted.prefill, mode)
        decode = {a: _predict_phase(runtime_root, target, engine, graphs, graphs.decode[a], mode) for a in anchors}
        phases[mode] = aggregate_request(
            prefill["scheduled_ms"], {a: r["scheduled_ms"] for a, r in decode.items()},
            workload=extracted.workload_type, batch=inputs["batch_size"], seq_len=inputs["seq_len"],
            max_new_tokens=inputs["max_new_tokens"], hardware=_hardware(runtime_root, target),
        )
        if mode == "median":
            prefill_detail, decode_detail = prefill, decode
    median = phases["median"]
    p90_calc = phases["p90"]["predicted_request_inference_ms"]
    median_ms = median["predicted_request_inference_ms"]
    status, signature = _validation_status(extracted)
    warnings = [P90_WARNING]
    if status != "validated":
        warnings.insert(0, EXPERIMENTAL_WARNING)
    if inputs.get("decoder_seq_len_ignored"):
        warnings.append(DECODER_SEQ_LEN_WARNING)
    return {
        "status": "ok",
        "target_instance": target,
        "engine_mode": engine,
        "mode": "generate",
        "prediction_scope": "generate_request_e2e",
        "generation_coverage": "prefill_and_decode",
        "decode_included": True,
        "max_new_tokens_applied": True,
        "architecture_validation_status": status,
        "validation_scope": VALIDATION_SCOPE,
        "architecture_signature": signature,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "batch_size": inputs["batch_size"],
        "seq_len": inputs["seq_len"],
        "max_new_tokens": inputs["max_new_tokens"],
        "predicted_median_ms": median_ms,
        "raw_predicted_median_ms": median_ms,
        "predicted_p90_ms": max(p90_calc, median_ms),
        "p90_status": P90_STATUS,
        # existing formula of predict_eager_runtime / predict_compiled_runtime,
        # applied to the Request E2E median
        "predicted_throughput_rps": inputs["batch_size"] / (median_ms / 1000.0),
        "official_estimate": "predicted_median_ms",
        "breakdown": {
            "device_resident_ms": median["kernel_and_host_generate_ms"],
            "prefill_ms": median["prefill_ms"],
            "decode_ms": median["decode_ms"],
            "h2d_ms": median["h2d_ms"],
            "d2h_ms": median["d2h_ms"],
            "final_sync_ms": median["final_sync_ms"],
            "decode_steps": median["decode_steps"],
            "first_decode_cache_length": median["first_decode_cache_length"],
            "last_decode_cache_length": median["last_decode_cache_length"],
            "cache_anchors_used": median["cache_anchors_used"],
            "p90_request_e2e_calculated_ms": p90_calc,
            "prefill_phase": _phase_summary(prefill_detail),
            "decode_anchor_phases": {str(a): _phase_summary(r) for a, r in decode_detail.items()},
        },
        "graph_extraction": {
            "method": "torch.export(strict=False) on meta/FakeTensor, use_cache=True",
            "wrapper": "model.forward",
            "graph_breaks": 0,  # torch.export yields one whole graph or raises
            "lm_head_module": extracted.lm_head_module,
            "num_logits_to_keep_applied": eg.supports_num_logits_to_keep(extracted.model),
            "lm_head_linear_nodes": {"prefill": extracted.prefill.lm_head_linear_nodes,
                                     **{f"decode@{a}": graphs.decode[a].lm_head_linear_nodes for a in anchors}},
            "prefill_graph_nodes": extracted.prefill.graph_node_count,
            "decode_graph_nodes": {str(a): graphs.decode[a].graph_node_count for a in anchors},
            "attempts": extracted.attempts,
        },
        "compiled_compatibility_adaptations": list(extracted.adaptations),
        "runtime_versions": runtime_versions(),
        "warnings": warnings,
    }


def _phase_summary(result: dict) -> dict:
    keys = ("scheduled_ms", "kernel_sum_ms", "routed_nodes", "predicted_framework_host_ms",
            "external_kernel_count", "fusion_group_count")
    return {k: result[k] for k in keys if k in result}


# --------------------------------------------------------------------------
# fallback
# --------------------------------------------------------------------------
def prefill_request(request: dict[str, Any]) -> dict[str, Any]:
    params = {k: v for k, v in (request.get("params") or {}).items() if k not in {"mode", "max_new_tokens"}}
    return {**request, "params": params}


def fallback_result(prefill_result: dict, *, target: str, reason: dict, warning: str) -> dict:
    result = dict(prefill_result)
    result.update(
        {
            "target_instance": target,
            "mode": "generate",
            "prediction_scope": "generate_prefill_fallback",
            "generation_coverage": "prefill_only",
            "decode_included": False,
            "max_new_tokens_applied": False,
            "generate_fallback_reason": reason,
            "runtime_versions": runtime_versions(),
        }
    )
    result["warnings"] = [warning, *list(prefill_result.get("warnings") or [])]
    return result


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def predict_generate_targets(
    request: dict[str, Any],
    targets: Iterable[str],
    *,
    engine_mode: str,
    runtime_root: Path,
    prefill_predict: Callable[[dict, str, str], dict],
) -> list[dict[str, Any]]:
    """One result per target. Raises GenerateInputError for 4xx inputs.

    prefill_predict(request, target, engine_mode) is the caller's existing
    Prefill predictor (used only for the explicit fallback)."""
    runtime_root = Path(runtime_root).resolve()
    inputs = resolve_generate_inputs(request)
    workload = request["type"]
    snapshot = request.get("model_config") or {}
    anchors = select_anchors(decode_cache_range(workload, inputs["seq_len"], inputs["max_new_tokens"]))
    targets = list(dict.fromkeys(targets))

    graphs = None
    graph_error: eg.GenerateGraphError | None = None
    try:
        config = eg.build_config(snapshot)
        check_position_limits(config, workload, inputs["seq_len"], inputs["max_new_tokens"])
        graphs = extract_request_graphs(snapshot, workload, inputs["batch_size"], inputs["seq_len"], anchors)
    except eg.GenerateGraphError as exc:
        graph_error = exc

    results = []
    for target in targets:
        reason = None
        warning = FALLBACK_WARNING
        target_status = None
        if not has_generate_artifacts(runtime_root, target):
            reason = {"stage": "target", "message": f"{target} has no validated forward-graph Generate artifacts"}
            warning = GR6_STYLE_WARNING.format(target=target)
            target_status = PENDING_TARGET_STATUS
        elif graph_error is not None:
            reason = {"stage": graph_error.stage, "message": graph_error.message, "attempts": graph_error.attempts}
        else:
            try:
                validate_generate_target(runtime_root, target)
                results.append(predict_target(runtime_root, target, engine_mode, graphs, inputs, anchors))
                continue
            except eg.GenerateGraphError as exc:
                reason = {"stage": exc.stage, "message": exc.message}
                warning = ROUTING_FALLBACK_WARNING if exc.stage == "routing" else FALLBACK_WARNING
            except Exception as exc:  # noqa: BLE001 - routing failure -> explicit fallback
                reason = {"stage": "routing", "message": f"{type(exc).__name__}: {str(exc)[:400]}"}
                warning = ROUTING_FALLBACK_WARNING
        prefill = prefill_predict(prefill_request(request), target, engine_mode)
        result = fallback_result(prefill, target=target, reason=reason, warning=warning)
        if target_status:
            result["generate_target_status"] = target_status
        results.append(result)
    return results
