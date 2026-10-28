#!/usr/bin/env python3
"""Model-independent Eager host enqueue model and CPU/GPU overlap scheduler.

This module deliberately does not contain model-specific scale factors or
end-to-end latency constants.  It predicts independently measured framework
enqueue work from graph/configuration features and places that work on the CPU
timeline before each routed ATen operation.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd


HEAVY_TYPES = {"Linear", "BMM", "BMM_SDPA"}
TYPE_FEATURES = (
    "Linear",
    "BMM",
    "BMM_SDPA",
    "Embedding",
    "LayerNorm",
    "Softmax",
    "Reduction",
    "MemoryCopy",
    "Pointwise",
)


def config_value(config: Any, *names: str, default: int = 0) -> int:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return int(default)


def graph_elements(gm: Any) -> tuple[int, int, int]:
    total = 0
    maximum = 0
    tensors = 0
    for node in gm.graph.nodes:
        meta = node.meta.get("tensor_meta")
        shape = getattr(meta, "shape", None)
        if shape is None:
            continue
        elements = int(math.prod(int(value) for value in shape))
        total += elements
        maximum = max(maximum, elements)
        tensors += 1
    return total, maximum, tensors


def layer_counts(log_df: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for value in log_df["Layer_Type"]:
        name = str(value)
        if name.startswith("Pointwise_"):
            name = "Pointwise"
        counts[name] += 1
    return counts


def feature_row(
    gm: Any,
    log_df: Any,
    config: Any,
    router: Any,
    *,
    phase: str,
    batch_size: int,
    seq_len: int,
    cache_len: int,
) -> dict[str, float | int]:
    counts = layer_counts(log_df)
    heavy_count = sum(counts.get(name, 0) for name in HEAVY_TYPES)
    routed_nodes = int(len(log_df))
    light_count = routed_nodes - heavy_count
    base_dispatch = (
        heavy_count * float(router.t_dispatch_heavy)
        + light_count * float(router.t_dispatch_light)
        + routed_nodes * float(router.t_python_loop)
        + float(router.t_pytorch_overhead)
    )
    total_elements, max_elements, tensor_nodes = graph_elements(gm)
    result: dict[str, float | int] = {
        "phase_decode": int(phase == "decode"),
        "family_seq2seq": int(bool(getattr(config, "is_encoder_decoder", False))),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "cache_len": int(cache_len),
        "hidden_size": config_value(config, "hidden_size", "d_model"),
        "intermediate_size": config_value(
            config, "intermediate_size", "ffn_dim", "d_ff"
        ),
        "num_layers": config_value(
            config,
            "num_hidden_layers",
            "decoder_layers",
            "num_decoder_layers",
        ),
        "num_heads": config_value(
            config, "num_attention_heads", "decoder_attention_heads", "num_heads"
        ),
        "num_kv_heads": config_value(
            config,
            "num_key_value_heads",
            "num_attention_heads",
            "decoder_attention_heads",
            "num_heads",
        ),
        "vocab_size": config_value(config, "vocab_size"),
        "routed_nodes": routed_nodes,
        "heavy_nodes": int(heavy_count),
        "light_nodes": int(light_count),
        "graph_tensor_nodes": int(tensor_nodes),
        "graph_total_elements": int(total_elements),
        "graph_max_elements": int(max_elements),
        "predicted_kernel_sum_ms": float(log_df["Predicted_Actual_ms"].sum()),
        "base_dispatch_ms": float(base_dispatch),
    }
    for name in TYPE_FEATURES:
        result[f"count_{name.lower()}"] = int(counts.get(name, 0))
    return result


def schedule_with_overlap(log_df: Any, router: Any, framework_extra_ms: float) -> float:
    """Replay one CPU producer and one ordered CUDA stream.

    The predicted framework work is distributed over the graph's routed ATen
    operations.  This is a scheduling assumption, not an additive correction:
    CPU work is exposed only when it outruns the GPU queue.
    """

    count = max(int(len(log_df)), 1)
    extra_per_node = max(float(framework_extra_ms), 0.0) / count
    cpu_time = float(router.t_pytorch_overhead)
    gpu_time = 0.0
    for _, row in log_df.iterrows():
        heavy = str(row["Layer_Type"]) in HEAVY_TYPES
        dispatch = router.t_dispatch_heavy if heavy else router.t_dispatch_light
        cpu_time += float(router.t_python_loop) + float(dispatch) + extra_per_node
        gpu_time = max(gpu_time, cpu_time) + float(row["Predicted_Actual_ms"])
    return float(gpu_time)


class GenerateEagerHostModel:
    def __init__(self, directory: Path):
        manifest_path = directory / "generate_eager_host_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not self.manifest.get("validation_gate_passed", False):
            raise RuntimeError(f"Unvalidated Eager host model: {manifest_path}")
        if self.manifest.get("model_name_is_feature", True):
            raise RuntimeError("Model name must not be an Eager host-model feature")
        if self.manifest.get("physical_dispatch_modified", True):
            raise RuntimeError("Eager host model requires unchanged physical dispatch")
        model_path = directory / str(self.manifest["model_file"])
        self.features = list(self.manifest["feature_names"])
        self.booster = lgb.Booster(model_file=str(model_path))

    def predict(self, features: dict[str, float | int]) -> float:
        missing = sorted(set(self.features) - set(features))
        if missing:
            raise KeyError(f"Missing Eager host-model features: {missing}")
        frame = pd.DataFrame([[features[name] for name in self.features]], columns=self.features)
        # The model target is log1p(milliseconds).
        import numpy as np

        value = float(np.expm1(self.booster.predict(frame)[0]))
        return max(value, 0.0)


def optional_host_model(target_root: Path) -> GenerateEagerHostModel | None:
    directory = target_root / "generate_eager_host_model"
    manifest = directory / "generate_eager_host_manifest.json"
    if not manifest.is_file():
        return None
    return GenerateEagerHostModel(directory)
