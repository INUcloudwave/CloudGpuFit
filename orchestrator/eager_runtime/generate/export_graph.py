"""Generate Prefill/Decode Graph extraction with torch.export on meta tensors.

Only torch.export (non-strict, FakeTensor/meta) is used. No weights are
downloaded and no real forward runs: the model is built from the request's
model_config_snapshot with AutoConfig + AutoModelFor{CausalLM,Seq2SeqLM}
.from_config() on the meta device in fp16.

Wrappers call the real model.forward() and only adapt inputs/outputs for
torch.export: Tensor arguments, use_cache=True, flattened KV cache <-> tuple,
selection of the last logits position + argmax, and (Seq2Seq) handing the
Prefill encoder output to Decode through the model's own `encoder_outputs`
argument. The model itself runs its Transformer, LM head, T5 scaling, BART
final_logits_bias and decides how many positions the LM head covers.

Decode inputs are built from the Prefill export's actual cache outputs; only
the self-attention length axis is set to the anchor length, cross-attention
cache keeps the encoder length.

Export-time aids (each applied only while torch.export runs, then restored):
  * meta_safe_autocast        - torch.autocast("meta") becomes a no-op context
  * qwen2/llama logger        - warning_once ignored by dynamo (logging only)
  * num_logits_to_keep=1    - passed only when model.forward() declares it
    (what HF generate() passes); recorded in the result metadata
  * fp16 inf/nan clamp branch - BartEncoderLayer only, and only when the
    Prefill export fails on exactly that data-dependent branch. Attention /
    MLP / LayerNorm steps unchanged. Recorded in compiled_compatibility_adaptations.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import json
import re
import textwrap
import threading
from functools import wraps
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Iterable

import torch
from torch import nn

# One process-wide lock for every graph trace. torch.export, FakeTensor/meta
# model construction (torch.set_default_dtype) and the temporary export-time
# patches all touch process-global state; two exports at once corrupt torch's
# tracing mode stack ("Trying to pop empty mode stack"), and an export running
# next to the existing Prefill FX trace fails too ("module is not installed as
# a submodule") -- in both cases leaving later traces in the process broken.
# It is the same RLock the existing Prefill path holds
# (predict_eager_runtime._RUNTIME_LOCK, also taken around the Prefill/forward
# predictors in predict_runtime), so Prefill traces and Generate exports never
# overlap. RLock: nested acquisition by the same thread cannot deadlock.
try:
    from predict_eager_runtime import _RUNTIME_LOCK as EXPORT_LOCK
except ImportError:  # export_graph used on its own (tests): a private lock
    EXPORT_LOCK = threading.RLock()


def _serialized(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with EXPORT_LOCK:
            return fn(*args, **kwargs)
    return wrapper


# graphs are always extracted with eager attention (matmul-softmax-matmul);
# reported as result metadata attention_implementation
ATTENTION_IMPLEMENTATION = "eager"

CAUSAL_WORKLOAD = "decoder_llm"
SEQ2SEQ_WORKLOAD = "text_encdec"


class GenerateGraphError(RuntimeError):
    """Graph extraction failed -- the caller falls back to Prefill."""

    def __init__(self, stage: str, message: str, attempts: list[dict] | None = None):
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.message = message
        self.attempts = attempts or []


# --------------------------------------------------------------------------
# meta model
# --------------------------------------------------------------------------
def config_fingerprint(snapshot: dict[str, Any]) -> str:
    normalized = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_config(snapshot: dict[str, Any]):
    from transformers import CONFIG_MAPPING

    if not isinstance(snapshot, dict) or not snapshot.get("model_type"):
        raise GenerateGraphError("config", "model_config_snapshot.model_type is required")
    model_type = str(snapshot["model_type"])
    if model_type not in CONFIG_MAPPING:
        raise GenerateGraphError(
            "config", f"unsupported_model_architecture: model_type={model_type!r} is not a transformers config"
        )
    config_class = CONFIG_MAPPING[model_type]
    try:
        config = config_class.from_dict(dict(snapshot))
    except Exception as exc:  # noqa: BLE001 - report, caller decides fallback
        raise GenerateGraphError("config", f"{type(exc).__name__}: {exc}") from exc
    config.use_cache = True
    config.return_dict = True
    config._attn_implementation = ATTENTION_IMPLEMENTATION
    if getattr(config, "is_encoder_decoder", False) and getattr(config, "decoder_start_token_id", None) is None:
        # token id value is irrelevant on meta tensors; same default as bundle v21
        config.decoder_start_token_id = 0
    return config


def build_meta_model(config, workload_type: str) -> nn.Module:
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

    factory = AutoModelForSeq2SeqLM if workload_type == SEQ2SEQ_WORKLOAD else AutoModelForCausalLM
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with torch.device("meta"):
            model = factory.from_config(config, trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001
        raise GenerateGraphError("model", f"{factory.__name__}.from_config failed: {type(exc).__name__}: {exc}") from exc
    finally:
        torch.set_default_dtype(previous)
    if any(not p.is_meta for p in model.parameters()):
        raise GenerateGraphError("model", "all parameters must stay on the meta device")
    return model.eval()


# --------------------------------------------------------------------------
# cache helpers (layout only, no shape guessing)
# --------------------------------------------------------------------------
def legacy_cache(value: Any) -> tuple[tuple[torch.Tensor, ...], ...]:
    converter = getattr(value, "to_legacy_cache", None)
    converted = converter() if callable(converter) else value
    if not isinstance(converted, (tuple, list)):
        raise TypeError(f"cache is not convertible to a legacy tuple: {type(value).__name__}")
    layers = []
    for layer in converted:
        if not isinstance(layer, (tuple, list)) or not all(torch.is_tensor(t) for t in layer):
            raise TypeError("cache layer is not a tuple of tensors")
        layers.append(tuple(layer))
    return tuple(layers)


def rebuild_cache(flat: Iterable[torch.Tensor], layout: tuple[int, ...]):
    values = tuple(flat)
    if len(values) != sum(layout):
        raise ValueError(f"expected {sum(layout)} cache tensors, received {len(values)}")
    out, index = [], 0
    for count in layout:
        out.append(tuple(values[index:index + count]))
        index += count
    return tuple(out)


class _Recorder:
    """Python-side record of the cache layout produced while tracing."""

    def __init__(self):
        self.layout: tuple[int, ...] | None = None

    def flatten(self, cache: Any) -> tuple[torch.Tensor, ...]:
        layers = legacy_cache(cache)
        self.layout = tuple(len(layer) for layer in layers)
        return tuple(t for layer in layers for t in layer)


# --------------------------------------------------------------------------
# wrappers: real model.forward(), input/output adaptation only
# --------------------------------------------------------------------------
def _last_token(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def supports_num_logits_to_keep(model: nn.Module) -> bool:
    """transformers 4.45.1 GenerationMixin passes num_logits_to_keep=1 only
    when the model's forward() declares that parameter explicitly. Decided
    from the signature (a bare **kwargs does not count), never by name."""
    parameter = inspect.signature(model.forward).parameters.get("num_logits_to_keep")
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY
    )


def _generate_kwargs(model: nn.Module) -> dict[str, Any]:
    # same extra forward argument HF generate() sends for these models
    return {"num_logits_to_keep": 1} if supports_num_logits_to_keep(model) else {}


class CausalPrefillForward(nn.Module):
    def __init__(self, model: nn.Module, recorder: _Recorder):
        super().__init__()
        self.model = model
        self.recorder = recorder

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True,
                         **_generate_kwargs(self.model))
        return (_last_token(out.logits), *self.recorder.flatten(out.past_key_values))


class CausalDecodeForward(nn.Module):
    def __init__(self, model: nn.Module, recorder: _Recorder, layout: tuple[int, ...]):
        super().__init__()
        self.model = model
        self.recorder = recorder
        self.layout = layout

    def forward(self, input_ids, attention_mask, *flat_cache):
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=rebuild_cache(flat_cache, self.layout), use_cache=True, return_dict=True,
            **_generate_kwargs(self.model),
        )
        return (_last_token(out.logits), *self.recorder.flatten(out.past_key_values))


class Seq2SeqPrefillForward(nn.Module):
    def __init__(self, model: nn.Module, recorder: _Recorder):
        super().__init__()
        self.model = model
        self.recorder = recorder

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
            use_cache=True, return_dict=True, **_generate_kwargs(self.model),
        )
        return (_last_token(out.logits), out.encoder_last_hidden_state, *self.recorder.flatten(out.past_key_values))


class Seq2SeqDecodeForward(nn.Module):
    """Encoder is not re-run: the Prefill encoder output goes in through the
    model's own encoder_outputs argument."""

    def __init__(self, model: nn.Module, recorder: _Recorder, layout: tuple[int, ...]):
        super().__init__()
        self.model = model
        self.recorder = recorder
        self.layout = layout

    def forward(self, decoder_input_ids, encoder_hidden_states, attention_mask, *flat_cache):
        out = self.model(
            encoder_outputs=(encoder_hidden_states,), attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids, past_key_values=rebuild_cache(flat_cache, self.layout),
            use_cache=True, return_dict=True, **_generate_kwargs(self.model),
        )
        return (_last_token(out.logits), *self.recorder.flatten(out.past_key_values))


# --------------------------------------------------------------------------
# export-time patches (always restored)
# --------------------------------------------------------------------------
@contextlib.contextmanager
def meta_safe_autocast():
    original = torch.autocast

    def factory(device_type, *args, **kwargs):
        if device_type == "meta":
            return contextlib.nullcontext()
        return original(device_type, *args, **kwargs)

    torch.autocast = factory
    try:
        yield
    finally:
        torch.autocast = original


@contextlib.contextmanager
def ignore_qwen_llama_warning_once(model: nn.Module):
    """Logging-only: keep warning_once out of the traced graph (bundle v21)."""
    module_name = type(model).__module__
    added = None
    if module_name.endswith(("modeling_qwen2", "modeling_llama")):
        try:
            import importlib

            logger_fn = importlib.import_module(module_name).logger.warning_once
            funcs = torch._dynamo.config.ignore_logging_functions
            if logger_fn not in funcs:
                funcs.add(logger_fn)
                added = logger_fn
        except (AttributeError, ImportError):
            added = None
    try:
        yield
    finally:
        if added is not None:
            torch._dynamo.config.ignore_logging_functions.discard(added)


_CLAMP_PATTERN = "torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()"


def _normalized_source(fn) -> str:
    source = textwrap.dedent(inspect.getsource(fn))
    source = re.sub(r'"""[\s\S]*?"""', "", source)
    source = re.sub(r"#[^\n]*", "", source)
    return re.sub(r"\s+", " ", source).strip()


# BartEncoderLayer.forward as reviewed for the clamp patch: transformers
# 4.45.1, SHA-256 of _normalized_source() (docstrings, comments and
# whitespace removed). bart_encoder_layer_forward_for_inference below is that
# source without the fp16 inf/nan clamp branch, so the patch is allowed only
# when the installed version and source hash both match.
BART_CLAMP_REVIEWED_TRANSFORMERS = "4.45.1"
BART_CLAMP_REVIEWED_SOURCE_SHA256 = "a5b03f6b5e5e4805675da31941470b20fed6153332667f208f8f270df81b1c31"


def bart_clamp_source_guard() -> tuple[bool, str]:
    """(allowed, reason) for the BartEncoderLayer clamp patch."""
    import transformers

    try:
        from transformers.models.bart.modeling_bart import BartEncoderLayer

        digest = hashlib.sha256(_normalized_source(BartEncoderLayer.forward).encode("utf-8")).hexdigest()
    except (ImportError, OSError, TypeError) as exc:
        return False, f"BartEncoderLayer.forward source unavailable: {type(exc).__name__}"
    if transformers.__version__ != BART_CLAMP_REVIEWED_TRANSFORMERS:
        return False, (f"transformers {transformers.__version__} != reviewed "
                       f"{BART_CLAMP_REVIEWED_TRANSFORMERS}")
    if digest != BART_CLAMP_REVIEWED_SOURCE_SHA256:
        return False, f"BartEncoderLayer.forward source sha256 {digest[:16]}... != reviewed"
    return True, "reviewed source"


def bart_encoder_layer_forward_for_inference(self, hidden_states, attention_mask, layer_head_mask, output_attentions=False):
    """BartEncoderLayer.forward (transformers 4.45.1) without the fp16
    inf/nan clamp branch. Attention / MLP / LayerNorm steps are unchanged."""
    residual = hidden_states
    hidden_states, attn_weights, _ = self.self_attn(
        hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
    )
    hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = self.self_attn_layer_norm(residual + hidden_states)
    residual = hidden_states
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = torch.nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states = self.fc2(hidden_states)
    hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = self.final_layer_norm(residual + hidden_states)
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)
    return outputs


def clamp_patch_candidates(model: nn.Module) -> list[nn.Module]:
    """BartEncoderLayer instances only (exact class), and only if the
    installed transformers version and forward source hash match the
    reviewed ones (bart_clamp_source_guard). Otherwise [] -> no patch, the
    export failure stands and the caller falls back to Prefill."""
    allowed, _ = bart_clamp_source_guard()
    if not allowed:
        return []
    from transformers.models.bart.modeling_bart import BartEncoderLayer

    return [m for m in model.modules() if type(m) is BartEncoderLayer]


def _has_bart_encoder_layers(model: nn.Module) -> bool:
    try:
        from transformers.models.bart.modeling_bart import BartEncoderLayer
    except ImportError:
        return False
    return any(type(m) is BartEncoderLayer for m in model.modules())


def is_clamp_branch_failure(exc: BaseException) -> bool:
    """True only for torch.export's data-dependent guard raised on the fp16
    inf/nan clamp line of an encoder layer."""
    text = f"{type(exc).__name__}: {exc}"
    data_dependent = "data-dependent" in text.lower() or "GuardOnDataDependentSymNode" in text or "DataDependent" in type(exc).__name__
    if not data_dependent:
        return False
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_name == "forward":
            condition = _enclosing_if_condition(tb.tb_frame.f_code, tb.tb_lineno)
            if condition is not None and "torch.float16" in condition and (
                "isinf(hidden_states)" in condition or "isnan(hidden_states)" in condition
            ):
                return True
        tb = tb.tb_next
    return _CLAMP_PATTERN in text


def _enclosing_if_condition(code, lineno: int) -> str | None:
    """Full source of the innermost `if` condition spanning `lineno`.

    The reported line of a multi-line condition differs by Python version
    (3.11: the `if ... and (` line, 3.13: the `torch.isinf(...)` line), so
    the whole condition is checked instead of the single reported line."""
    try:
        lines, start = inspect.getsourcelines(code)
        tree = ast.parse(textwrap.dedent("".join(lines)))
    except (OSError, TypeError, SyntaxError):
        return None
    line = lineno - start + 1
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and node.test.lineno <= line <= (node.test.end_lineno or node.test.lineno):
            span = (node.test.end_lineno or node.test.lineno) - node.test.lineno
            if best is None or span < best[0]:
                best = (span, node.test)
    return ast.unparse(best[1]) if best else None


@contextlib.contextmanager
def clamp_branch_patch(layers: list[nn.Module]):
    for layer in layers:
        layer.forward = MethodType(bart_encoder_layer_forward_for_inference, layer)
    try:
        yield
    finally:
        for layer in layers:
            layer.__dict__.pop("forward", None)


# --------------------------------------------------------------------------
# export + validation
# --------------------------------------------------------------------------
@dataclass
class TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class PhaseGraph:
    phase: str
    cache_len: int
    graph_module: torch.fx.GraphModule
    outputs: list[TensorSpec]
    layout: tuple[int, ...]
    graph_node_count: int
    lm_head_linear_nodes: int
    lifted_constants: frozenset[str]


@dataclass
class ExtractedModel:
    workload_type: str
    model_type: str
    config: Any
    model: nn.Module
    batch: int
    seq_len: int
    prefill: PhaseGraph
    cache_specs: list[TensorSpec]       # Prefill cache outputs
    self_mask: list[bool]               # True = self-attention entry (grows on decode)
    encoder_hidden: TensorSpec | None
    lm_head_module: str
    adaptations: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)


def _output_specs(gm: torch.fx.GraphModule) -> list[TensorSpec]:
    output = next(node for node in gm.graph.nodes if node.op == "output")
    values = output.args[0]
    values = values if isinstance(values, (tuple, list)) else (values,)
    specs = []
    for value in values:
        val = value.meta.get("val") if isinstance(value, torch.fx.Node) else None
        if not torch.is_tensor(val):
            raise GenerateGraphError("shape", "export output is not a tensor")
        specs.append(TensorSpec(tuple(int(s) for s in val.shape), val.dtype))
    return specs


class _ConcreteTensorMeta:
    def __init__(self, value):
        self.shape = torch.Size(int(s) for s in value.shape)
        self.dtype = value.dtype
        self.stride = tuple(int(s) for s in value.stride())


def _normalize_metadata(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        value = node.meta.get("val")
        if torch.is_tensor(value):
            node.meta["tensor_meta"] = _ConcreteTensorMeta(value)
    return gm


def _module_path(node: torch.fx.Node) -> str:
    stack = node.meta.get("nn_module_stack")
    if not isinstance(stack, dict) or not stack:
        return ""
    value = list(stack.values())[-1]
    return str(value[0]) if isinstance(value, tuple) and value else ""


def lm_head_module_name(model: nn.Module) -> str:
    head = model.get_output_embeddings()
    for name, module in model.named_modules():
        if module is head and name:
            return name
    raise GenerateGraphError("lm_head", "output embeddings module not found in the model")


def count_lm_head_linears(gm: torch.fx.GraphModule, head_name: str) -> int:
    """aten linear nodes executed inside the model's output-embedding module
    (wrapper attribute `model.` prefix included)."""
    target_path = f"model.{head_name}"
    count = 0
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        name = str(getattr(node.target, "__name__", node.target)).split(".")[0]
        if name in {"linear", "mm", "addmm", "matmul"} and _module_path(node) == target_path:
            count += 1
    return count


def _export(module: nn.Module, args: tuple, model: nn.Module, patches: list[nn.Module]):
    with contextlib.ExitStack() as stack:
        stack.enter_context(meta_safe_autocast())
        stack.enter_context(ignore_qwen_llama_warning_once(model))
        if patches:
            stack.enter_context(clamp_branch_patch(patches))
        exported = torch.export.export(module, args, strict=False)
    lifted = frozenset(getattr(exported.graph_signature, "inputs_to_lifted_tensor_constants", {}) or {})
    return _normalize_metadata(exported.graph_module), lifted


def _meta_long(shape, value):
    return torch.full(shape, value, device="meta", dtype=torch.long)


def _check_prefill(specs, layout, batch, prefill_len, seq2seq, encoder_len):
    if layout is None or not layout:
        raise GenerateGraphError("cache", "Prefill produced no KV cache (use_cache=True not honoured)")
    if len(set(layout)) != 1 or layout[0] not in ((4,) if seq2seq else (2,)):
        raise GenerateGraphError("cache", f"unexpected cache layout per layer: {sorted(set(layout))}")
    if specs[0].shape != (batch, 1):
        raise GenerateGraphError("shape", f"next-token shape {specs[0].shape} != {(batch, 1)}")
    cache = specs[2 if seq2seq else 1:]
    if len(cache) != sum(layout):
        raise GenerateGraphError("cache", f"cache tensor count {len(cache)} != layout {sum(layout)}")
    self_mask = []
    for index, spec in enumerate(cache):
        is_self = index % layout[0] < 2
        self_mask.append(is_self)
        if len(spec.shape) != 4 or spec.shape[0] != batch:
            raise GenerateGraphError("cache", f"cache tensor {index} has unexpected shape {spec.shape}")
        expected = prefill_len if is_self else encoder_len
        if spec.shape[-2] != expected:
            raise GenerateGraphError("cache", f"cache tensor {index} length {spec.shape[-2]} != {expected}")
    return cache, self_mask


def _check_lm_head(gm, head_name, phase):
    count = count_lm_head_linears(gm, head_name)
    if count != 1:
        raise GenerateGraphError("lm_head", f"{phase}: {count} LM head linear nodes in '{head_name}' (expected exactly 1)")
    return count


def _assert_meta_only(model: nn.Module) -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        raise GenerateGraphError("device", "CUDA was initialised during Generate graph extraction")
    if next(model.parameters()).device.type != "meta":
        raise GenerateGraphError("device", "model parameters are not on the meta device")


@_serialized
def extract_prefill(snapshot, workload_type, batch, seq_len) -> ExtractedModel:
    config = build_config(snapshot)
    model = build_meta_model(config, workload_type)
    seq2seq = workload_type == SEQ2SEQ_WORKLOAD
    head_name = lm_head_module_name(model)
    input_ids = _meta_long((batch, seq_len), 3)
    attention_mask = torch.ones((batch, seq_len), device="meta", dtype=torch.long)
    args = (input_ids, attention_mask)
    if seq2seq:
        args += (_meta_long((batch, 1), int(config.decoder_start_token_id)),)
    wrapper_cls = Seq2SeqPrefillForward if seq2seq else CausalPrefillForward
    attempts: list[dict[str, Any]] = []
    adaptations: list[dict[str, Any]] = []
    recorder = _Recorder()
    try:
        gm, lifted = _export(wrapper_cls(model, recorder), args, model, [])
    except Exception as exc:  # noqa: BLE001
        patches = clamp_patch_candidates(model) if seq2seq else []
        if not (patches and is_clamp_branch_failure(exc)):
            attempt = {"wrapper": "model.forward", "stage": "prefill_export", "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            if seq2seq and not patches and _has_bart_encoder_layers(model) and is_clamp_branch_failure(exc):
                attempt["clamp_patch_refused"] = bart_clamp_source_guard()[1]
            attempts.append(attempt)
            raise GenerateGraphError("prefill_export", f"{type(exc).__name__}: {str(exc)[:400]}", attempts) from exc
        adaptations.append({
            "type": "fp16_inf_nan_clamp_branch_removed",
            "reason": "torch.export cannot guard the data-dependent fp16 inf/nan clamp branch on meta tensors",
            "layer_classes": sorted({type(layer).__name__ for layer in patches}),
            "layers": len(patches),
            "reference": "transformers.models.bart.modeling_bart.BartEncoderLayer.forward",
            "reviewed_transformers": BART_CLAMP_REVIEWED_TRANSFORMERS,
            "reviewed_source_sha256": BART_CLAMP_REVIEWED_SOURCE_SHA256,
            "scope": "export_only_restored",
        })
        recorder = _Recorder()
        try:
            gm, lifted = _export(wrapper_cls(model, recorder), args, model, patches)
        except Exception as exc2:  # noqa: BLE001
            attempts.append({"wrapper": "model.forward", "stage": "prefill_export_patched", "error": f"{type(exc2).__name__}: {str(exc2)[:400]}"})
            raise GenerateGraphError("prefill_export", f"{type(exc2).__name__}: {str(exc2)[:400]}", attempts) from exc2
    _assert_meta_only(model)
    try:
        specs = _output_specs(gm)
        encoder_hidden = specs[1] if seq2seq else None
        if seq2seq and (len(encoder_hidden.shape) != 3 or encoder_hidden.shape[:2] != (batch, seq_len)):
            raise GenerateGraphError("shape", f"encoder hidden shape {encoder_hidden.shape}")
        cache, self_mask = _check_prefill(specs, recorder.layout, batch, 1 if seq2seq else seq_len, seq2seq, seq_len)
        head_count = _check_lm_head(gm, head_name, "prefill")
    except (GenerateGraphError, TypeError) as exc:
        stage = getattr(exc, "stage", "prefill_check")
        attempts.append({"wrapper": "model.forward", "stage": stage, "error": str(exc)})
        raise GenerateGraphError(stage, str(exc), attempts) from exc
    attempts.append({"wrapper": "model.forward", "stage": "prefill", "status": "ok"})
    return ExtractedModel(
        workload_type=workload_type, model_type=str(config.model_type), config=config, model=model,
        batch=batch, seq_len=seq_len,
        prefill=PhaseGraph("prefill", 0, gm, specs, recorder.layout, sum(1 for _ in gm.graph.nodes), head_count, lifted),
        cache_specs=cache, self_mask=self_mask, encoder_hidden=encoder_hidden, lm_head_module=head_name,
        adaptations=adaptations, attempts=attempts,
    )


@_serialized
def extract_decode(extracted: ExtractedModel, cache_len: int) -> PhaseGraph:
    seq2seq = extracted.workload_type == SEQ2SEQ_WORKLOAD
    batch, seq_len = extracted.batch, extracted.seq_len
    flat_cache = []
    for spec, is_self in zip(extracted.cache_specs, extracted.self_mask):
        shape = list(spec.shape)
        if is_self:
            shape[-2] = cache_len
        flat_cache.append(torch.empty(tuple(shape), device="meta", dtype=spec.dtype))
    recorder = _Recorder()
    layout = extracted.prefill.layout
    if seq2seq:
        hidden = extracted.encoder_hidden
        wrapper = Seq2SeqDecodeForward(extracted.model, recorder, layout)
        args = (
            _meta_long((batch, 1), int(extracted.config.decoder_start_token_id)),
            torch.empty(hidden.shape, device="meta", dtype=hidden.dtype),
            torch.ones((batch, seq_len), device="meta", dtype=torch.long), *flat_cache,
        )
    else:
        wrapper = CausalDecodeForward(extracted.model, recorder, layout)
        args = (_meta_long((batch, 1), 3), torch.ones((batch, cache_len + 1), device="meta", dtype=torch.long), *flat_cache)
    try:
        gm, lifted = _export(wrapper, args, extracted.model, [])
    except Exception as exc:  # noqa: BLE001
        raise GenerateGraphError("decode_export", f"cache_len={cache_len}: {type(exc).__name__}: {str(exc)[:400]}") from exc
    _assert_meta_only(extracted.model)
    specs = _output_specs(gm)
    if recorder.layout != layout:
        raise GenerateGraphError("cache", f"decode cache layout {recorder.layout} != prefill {layout}")
    if specs[0].shape != (batch, 1):
        raise GenerateGraphError("shape", f"decode next-token shape {specs[0].shape}")
    out_cache = specs[1:]
    if len(out_cache) != len(extracted.cache_specs):
        raise GenerateGraphError("cache", "decode cache tensor count changed")
    for index, (before, after, is_self) in enumerate(zip(extracted.cache_specs, out_cache, extracted.self_mask)):
        expected = list(before.shape)
        expected[-2] = cache_len + 1 if is_self else before.shape[-2]
        if list(after.shape) != expected or after.dtype != before.dtype:
            raise GenerateGraphError("cache", f"decode cache tensor {index}: {after.shape} != {tuple(expected)} (self={is_self})")
    head_count = _check_lm_head(gm, extracted.lm_head_module, f"decode@{cache_len}")
    return PhaseGraph("decode", cache_len, gm, specs, layout, sum(1 for _ in gm.graph.nodes), head_count, lifted)


def math_prod(values) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out
