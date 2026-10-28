"""`validated` vs `experimental_unvalidated` for Generate results.

`validated` needs both: an E2E-measured Generate reference for the
architecture, and the request config matching that reference on the
structural features that change the graph (not on sizes):

  attention (MHA / GQA), cache layout, activation / feed_forward_proj,
  dense vs MoE, local / sparse attention, LM head tie, encoder-decoder,
  attention / MLP bias switches, RoPE scaling type, plus architecture-specific structural switches (OPT pre-LN and
  word_embed_proj_dim == hidden_size, Qwen2 sliding window).

References = the configs the bundle's Generate checkpoints were measured
with (bundle runtime/generate_model_configs.py: opt, qwen2, llama, tinyllama,
t5, bart). Layer count, hidden size, heads and vocab may differ.
"""
from __future__ import annotations

from typing import Any

# Bundle generate_model_configs.py values (measured Generate references).
REFERENCE_SNAPSHOTS: dict[str, list[dict[str, Any]]] = {
    "opt": [dict(model_type="opt", vocab_size=50272, hidden_size=768, num_hidden_layers=12, ffn_dim=3072,
                 num_attention_heads=12, max_position_embeddings=2048)],
    "qwen2": [dict(model_type="qwen2", vocab_size=151936, hidden_size=1536, intermediate_size=8960,
                   num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2,
                   max_position_embeddings=131072, rope_theta=1_000_000.0, tie_word_embeddings=True,
                   bos_token_id=151643, eos_token_id=151643)],
    "llama": [
        dict(model_type="llama", vocab_size=32000, hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
             num_attention_heads=32, num_key_value_heads=8, max_position_embeddings=4096, attention_bias=False,
             mlp_bias=False, tie_word_embeddings=False, pad_token_id=0, bos_token_id=1, eos_token_id=2),
        dict(model_type="llama", vocab_size=32000, hidden_size=2048, intermediate_size=5632, num_hidden_layers=22,
             num_attention_heads=32, num_key_value_heads=4, max_position_embeddings=2048, attention_bias=False,
             mlp_bias=False, tie_word_embeddings=False, pad_token_id=0, bos_token_id=1, eos_token_id=2,
             rms_norm_eps=1e-5, rope_theta=10000.0),
    ],
    "t5": [dict(model_type="t5", vocab_size=32128, d_model=512, d_kv=64, d_ff=2048, num_layers=6,
                num_decoder_layers=6, num_heads=8)],
    "bart": [dict(model_type="bart", vocab_size=50265, d_model=768, encoder_layers=6, decoder_layers=6,
                  encoder_attention_heads=12, decoder_attention_heads=12, encoder_ffn_dim=3072,
                  decoder_ffn_dim=3072, max_position_embeddings=1024, activation_function="gelu",
                  dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, classifier_dropout=0.0,
                  forced_bos_token_id=None, forced_eos_token_id=None, pad_token_id=1, bos_token_id=0,
                  eos_token_id=2, decoder_start_token_id=2)],
}
REFERENCE_CACHE_TENSORS = {"opt": 2, "qwen2": 2, "llama": 2, "t5": 4, "bart": 4}
MOE_FIELDS = ("num_experts", "num_local_experts", "moe_num_experts", "n_routed_experts", "num_experts_per_tok")
LOCAL_SPARSE_FIELDS = ("attention_window", "local_radius", "block_size", "num_random_blocks", "global_block_size")


def _first(config: Any, *names: str):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def _flag(value: Any) -> bool | None:
    return None if value is None else bool(value)


def signature(config: Any) -> dict[str, Any]:
    heads = _first(config, "num_attention_heads", "decoder_attention_heads", "num_heads")
    kv_heads = _first(config, "num_key_value_heads")
    attention_type = _first(config, "attention_type", "encoder_attention_type")
    sliding = bool(getattr(config, "use_sliding_window", False)) or (
        getattr(config, "sliding_window", None) is not None and config.model_type not in {"qwen2"}
    )
    local_sparse = (
        (attention_type is not None and str(attention_type) not in {"original_full", "full"})
        or sliding
        or any(getattr(config, name, None) not in (None, 0) for name in LOCAL_SPARSE_FIELDS)
    )
    sig = {
        "model_type": config.model_type,
        "encoder_decoder": bool(getattr(config, "is_encoder_decoder", False)),
        "attention": "gqa" if kv_heads is not None and heads is not None and int(kv_heads) < int(heads) else "mha",
        "activation": str(_first(config, "feed_forward_proj", "hidden_act", "activation_function")),
        "moe": any(getattr(config, name, None) not in (None, 0, 1) for name in MOE_FIELDS),
        "local_or_sparse_attention": bool(local_sparse),
        "lm_head_tied": bool(getattr(config, "tie_word_embeddings", False)),
    }
    # graph-changing switches read from the config as written (None = the
    # architecture has no such config switch and uses its fixed default)
    sig["attention_bias"] = _flag(_first(config, "attention_bias", "enable_bias", "use_bias", "bias"))
    sig["mlp_bias"] = _flag(_first(config, "mlp_bias", "enable_bias", "use_bias", "bias"))
    rope_scaling = getattr(config, "rope_scaling", None)
    sig["rope_scaling"] = (
        None if not rope_scaling
        else str(rope_scaling.get("rope_type") or rope_scaling.get("type") or "unknown") if isinstance(rope_scaling, dict)
        else str(rope_scaling)
    )
    if config.model_type == "opt":
        sig["opt_pre_layernorm"] = bool(getattr(config, "do_layer_norm_before", True))
        sig["opt_projection"] = int(getattr(config, "word_embed_proj_dim", config.hidden_size)) != int(config.hidden_size)
    return sig


def _reference_signatures(model_type: str) -> list[dict[str, Any]]:
    from transformers import CONFIG_MAPPING

    out = []
    for snapshot in REFERENCE_SNAPSHOTS.get(model_type, []):
        config = CONFIG_MAPPING[model_type].from_dict(dict(snapshot))
        out.append(signature(config))
    return out


def validation_status(config: Any, *, cache_tensors_per_layer: int | None, adaptations: list[dict]) -> tuple[str, dict]:
    sig = signature(config)
    references = _reference_signatures(config.model_type)
    reasons = []
    if not references:
        reasons.append("no E2E-measured Generate reference for this model_type")
    elif sig not in references:
        diffs = {k: (sig.get(k), [r.get(k) for r in references]) for k in sig if all(sig.get(k) != r.get(k) for r in references)}
        reasons.append(f"structure differs from the measured reference: {diffs}")
    if references and cache_tensors_per_layer != REFERENCE_CACHE_TENSORS.get(config.model_type):
        reasons.append(f"cache layout {cache_tensors_per_layer} tensors/layer differs from reference")
    if adaptations and config.model_type != "bart":
        reasons.append("export adaptation applied to a non-reference architecture")
    status = "validated" if not reasons else "experimental_unvalidated"
    return status, {"features": sig, "reasons": reasons}
