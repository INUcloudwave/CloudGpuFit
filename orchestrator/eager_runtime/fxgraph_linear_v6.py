#!/usr/bin/env python3
"""Hybrid eager simulator with Transformer-Linear context routing.

Required beside this file:
  fxgraph_reduction_v2.py
  fxgraph_cache_v3.py
  fxgraph_memory_v4.py
  fxgraph_sdpa_v5.py

Required model directories:
  cache_model/, memory_model/, sdpa_model/, linear_context_model_deep_v2/
"""

from __future__ import annotations

import math
import os
import re
from collections import OrderedDict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import fxgraph_memory_v4 as memory_v4
import fxgraph_reduction_v2 as base
import fxgraph_sdpa_v5 as sdpa_v5


PROJECTIONS = ("Q", "K", "V", "AttentionOut", "FFNUp", "FFNDown")
PRECISION_BYTES = 2
L2_CACHE_BYTES = 48 * 1024 * 1024


def _iter_base_nodes(fused_graph):
    for group in fused_graph:
        if group.get("type") == "FusionGroup":
            yield group["base_op"]


def _input_batch(value):
    if isinstance(value, torch.Tensor) and value.ndim:
        return int(value.shape[0])
    if isinstance(value, dict):
        for item in value.values():
            batch = _input_batch(item)
            if batch is not None:
                return batch
    if isinstance(value, (tuple, list)):
        for item in value:
            batch = _input_batch(item)
            if batch is not None:
                return batch
    return None


def _projection(target):
    name = target.lower()
    tail = name.rsplit(".", 1)[-1]

    if tail == "c_attn":
        return "Q"
    if tail in {"query", "q", "q_proj"}:
        return "Q"
    if tail in {"key", "k", "k_proj"}:
        return "K"
    if tail in {"value", "v", "v_proj"}:
        return "V"
    if tail in {"out_proj", "o_proj", "o"}:
        return "AttentionOut"
    if "attention.output.dense" in name:
        return "AttentionOut"
    if tail == "c_proj" and ".attn." in name:
        return "AttentionOut"

    if tail in {
        "fc1",
        "c_fc",
        "wi",
        "wi_0",
        "wi_1",
        "gate_proj",
        "up_proj",
        "intermediate_dense",
    }:
        return "FFNUp"
    if "intermediate.dense" in name:
        return "FFNUp"

    if tail == "c_proj" and ".mlp." in name:
        return "FFNDown"
    if tail in {"fc2", "wo", "down_proj", "output_dense"}:
        return "FFNDown"
    if "attention.output.dense" not in name and name.endswith("output.dense"):
        return "FFNDown"
    return None


def _block_key(target):
    name = target.lower()
    patterns = (
        r"(?:^|\.)(encoder|decoder)\.layer\.(\d+)(?:\.|$)",
        r"(?:^|\.)(encoder|decoder)\.layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)(encoder|decoder)\.block\.(\d+)(?:\.|$)",
        r"(?:^|\.)(layers)\.(\d+)(?:\.|$)",
        r"(?:^|\.)(h)\.(\d+)(?:\.|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return match.group(1), int(match.group(2))
    return None


def _context_working_set(batch, sequence, hidden, intermediate, depth, bias):
    m = batch * sequence
    weights_per_layer = 4 * hidden**2 + 2 * hidden * intermediate
    activations = m * (6 * hidden + 2 * intermediate)
    bias_elements = (5 * hidden + intermediate if bias else 0) * depth
    return (weights_per_layer * depth + activations + bias_elements) * PRECISION_BYTES


def _previous_weight_bytes(projection, hidden, intermediate):
    if projection == "FFNDown":
        return hidden * intermediate * PRECISION_BYTES
    return hidden * hidden * PRECISION_BYTES


def _previous_gap_bytes(projection, m, hidden, intermediate):
    if projection in {"K", "V"}:
        return m * hidden * PRECISION_BYTES
    if projection == "AttentionOut":
        return 4 * m * hidden * PRECISION_BYTES
    if projection == "FFNUp":
        return 2 * m * hidden * PRECISION_BYTES
    if projection == "FFNDown":
        return 2 * m * intermediate * PRECISION_BYTES
    return 2 * m * hidden * PRECISION_BYTES


def _node_shape(node):
    meta = getattr(node, "meta", {}).get("tensor_meta")
    if meta is None or not hasattr(meta, "shape"):
        return None
    return tuple(int(value) for value in meta.shape)


def _module_stack_target(node):
    stack = getattr(node, "meta", {}).get("nn_module_stack")
    if not isinstance(stack, dict):
        return None
    for key, value in reversed(list(stack.items())):
        if isinstance(value, tuple) and value:
            path = str(value[0])
        else:
            path = str(key)
        tail = path.lower().rsplit(".", 1)[-1]
        if tail in {"c_attn", "c_fc", "c_proj"}:
            return path
    return None


def _linear_graph_info(node, modules):
    if node.op == "call_module":
        module = modules.get(str(node.target))
        if not isinstance(module, nn.Linear):
            return None
        return {
            "target": str(node.target),
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "has_bias": int(module.bias is not None),
            "rhs_is_transposed": 1,
        }

    if node.op != "call_function" or node.target is not torch.addmm:
        return None
    if len(node.args) < 3:
        return None
    input_shape = _node_shape(node.args[1])
    weight_shape = _node_shape(node.args[2])
    target = _module_stack_target(node)
    if (
        target is None
        or input_shape is None
        or weight_shape is None
        or len(input_shape) != 2
        or len(weight_shape) != 2
        or input_shape[1] != weight_shape[0]
    ):
        return None
    return {
        "target": target,
        "in_features": input_shape[1],
        "out_features": weight_shape[1],
        "has_bias": 1,
        "rhs_is_transposed": 0,
    }


class LinearContextFXGraphParser(sdpa_v5.SDPAAwareFXGraphParser):
    def extract_nodes(self, gm, dummy_input, engine_mode="eager"):
        fused_graph = super().extract_nodes(gm, dummy_input, engine_mode)
        modules = dict(gm.named_modules())
        input_batch = _input_batch(dummy_input) or 1

        graph_linears = []
        block_order = OrderedDict()
        for fx_node in gm.graph.nodes:
            info = _linear_graph_info(fx_node, modules)
            if info is None:
                continue
            target = info["target"]
            projection = _projection(target)
            block = _block_key(target)
            info.update({"projection": projection, "block": block})
            graph_linears.append(info)
            if projection is not None and block is not None:
                block_order.setdefault(block, len(block_order))

        parsed_linears = [
            node for node in _iter_base_nodes(fused_graph)
            if node.get("type") == "Linear"
        ]
        if len(parsed_linears) != len(graph_linears):
            print(
                "Transformer Linear routing skipped: FX/parsed Linear count "
                f"mismatch ({len(graph_linears)} vs {len(parsed_linears)})"
            )
            return fused_graph

        block_dimensions = {}
        model_has_bias = 0
        for info in graph_linears:
            projection = info["projection"]
            block = info["block"]
            if projection is None or block is None:
                continue
            dims = block_dimensions.setdefault(
                block, {"hidden": 0, "intermediate": 0}
            )
            if projection in {"Q", "K", "V"}:
                dims["hidden"] = max(
                    dims["hidden"], info["in_features"]
                )
            elif projection == "AttentionOut":
                dims["hidden"] = max(
                    dims["hidden"], info["out_features"]
                )
            elif projection == "FFNUp":
                dims["hidden"] = max(
                    dims["hidden"], info["in_features"]
                )
                dims["intermediate"] = max(
                    dims["intermediate"], info["out_features"]
                )
            elif projection == "FFNDown":
                dims["hidden"] = max(
                    dims["hidden"], info["out_features"]
                )
                dims["intermediate"] = max(
                    dims["intermediate"], info["in_features"]
                )
            model_has_bias = max(model_has_bias, info["has_bias"])

        depth = len(block_order)
        routed = 0
        for parsed, info in zip(parsed_linears, graph_linears):
            target = info["target"]
            projection = info["projection"]
            block = info["block"]
            parsed["FX_Target"] = target
            if projection is None or block not in block_order:
                parsed["Is_Transformer_Context"] = 0
                continue

            dims = block_dimensions[block]
            hidden = dims["hidden"] or info["in_features"]
            intermediate = dims["intermediate"] or hidden
            layer_index = block_order[block]
            m = int(parsed["M"])
            batch = input_batch
            sequence = max(int(round(m / max(batch, 1))), 1)
            context_bytes = _context_working_set(
                batch,
                sequence,
                hidden,
                intermediate,
                max(depth, 1),
                model_has_bias,
            )
            weight_bytes = int(parsed["K"]) * int(parsed["N"]) * PRECISION_BYTES
            reuse_distance = max(context_bytes - weight_bytes, 0)

            parsed.update(
                {
                    "Use_Transformer_Linear_Context": 1,
                    "Is_Transformer_Context": 1,
                    "Sequence_Length": sequence,
                    "Hidden_Size": hidden,
                    "Intermediate_Size": intermediate,
                    "Block_Depth": max(depth, 1),
                    "Layer_Position": layer_index / max(depth - 1, 1),
                    "Projection_Index": PROJECTIONS.index(projection),
                    "Is_QKV": int(projection in {"Q", "K", "V"}),
                    "Is_AttentionOut": int(projection == "AttentionOut"),
                    "Is_FFNUp": int(projection == "FFNUp"),
                    "Is_FFNDown": int(projection == "FFNDown"),
                    "Has_Bias": info["has_bias"],
                    "Previous_Linear_Weight_Bytes": _previous_weight_bytes(
                        projection, hidden, intermediate
                    ),
                    "Previous_Op_Gap_Bytes": _previous_gap_bytes(
                        projection, m, hidden, intermediate
                    ),
                    "Weight_Bytes": weight_bytes,
                    "RHS_Is_Transposed": info["rhs_is_transposed"],
                    "Context_Working_Set_Bytes": context_bytes,
                    "Reuse_Distance_Bytes": reuse_distance,
                    "Estimated_Weight_L2_Resident": int(
                        weight_bytes + reuse_distance <= L2_CACHE_BYTES
                    ),
                }
            )
            routed += 1

        if routed:
            print(
                f"  Transformer Linear routing: {routed}/{len(parsed_linears)} "
                f"nodes, {depth} blocks"
            )
        return fused_graph


class LinearContextSimulatorRouter(sdpa_v5.SDPAAwareSimulatorRouter):
    def __init__(self, engine_mode="eager"):
        super().__init__(engine_mode=engine_mode)
        model_dir = Path(
            os.environ.get(
                "LINEAR_CONTEXT_MODEL_DIR",
                "linear_context_model_deep_v2",
            )
        )
        self.linear_context_median = lgb.Booster(
            model_file=str(
                model_dir / "gemm_context_global_eager_median.pkl"
            )
        )
        self.linear_context_90th = lgb.Booster(
            model_file=str(
                model_dir / "gemm_context_global_eager_90th.pkl"
            )
        )
        required = {
            "Sequence_Length",
            "Hidden_Size",
            "Intermediate_Size",
            "Block_Depth",
            "Layer_Position",
            "Projection_Index",
            "Previous_Linear_Weight_Bytes",
            "Previous_Op_Gap_Bytes",
            "Is_Transformer_Context",
        }
        missing = required.difference(self.linear_context_median.feature_name())
        if missing:
            raise RuntimeError(
                "Transformer Linear model lacks context features: "
                + ", ".join(sorted(missing))
            )
        print(
            "v6 P90 is diagnostic only: held-out BERT/RoBERTa "
            "coverage is not yet calibrated to 90%."
        )

    def _predict_node(
        self,
        node,
        batch_size,
        prediction_mode,
        force_memory_zero=False,
    ):
        if not (
            node.get("type") == "Linear"
            and node.get("Use_Transformer_Linear_Context", 0) == 1
        ):
            return super()._predict_node(
                node,
                batch_size,
                prediction_mode,
                force_memory_zero,
            )

        m, n, k = int(node["M"]), int(node["N"]), int(node["K"])
        total_flops = 2 * m * n * k
        bias_bytes = n * self.precision_bytes if node["Has_Bias"] else 0
        ideal_bytes = (
            0
            if force_memory_zero
            else (m * k + k * n + m * n) * self.precision_bytes + bias_bytes
        )
        theoretical_ms = max(
            total_flops / self.peak_flops * 1000,
            ideal_bytes / self.peak_bandwidth * 1000,
            1e-6,
        )
        row = {
            "Batch_Size": batch_size,
            "Dim_M": m,
            "Dim_N": n,
            "Dim_K": k,
            "Total_FLOPs": total_flops,
            "Ideal_Memory_Bytes": ideal_bytes,
            "Arithmetic_Intensity": total_flops / max(ideal_bytes, 1),
            "Estimated_L2_Fit": int(ideal_bytes <= self.l2_cache_bytes),
            "Orig_Kernel_Size": node.get("kernel_size", 1),
            "Precision_Bytes": self.precision_bytes,
            "Is_Conv1D": 0,
            "Is_Conv2D": 0,
            "Is_BMM": 0,
            "Is_Fused": node.get("Is_Fused", 0),
            "Estimated_Wave_Count": (
                math.ceil(m / 128.0)
                * math.ceil(n / 128.0)
                / self.sm_count
            ),
            "Weight_Bytes": node["Weight_Bytes"],
            "RHS_Is_Transposed": node["RHS_Is_Transposed"],
            "Estimated_Weight_L2_Resident": node[
                "Estimated_Weight_L2_Resident"
            ],
            "Reuse_Distance_Bytes": node["Reuse_Distance_Bytes"],
            "Context_Working_Set_Bytes": node["Context_Working_Set_Bytes"],
            "Sequence_Length": node["Sequence_Length"],
            "Hidden_Size": node["Hidden_Size"],
            "Intermediate_Size": node["Intermediate_Size"],
            "Block_Depth": node["Block_Depth"],
            "Layer_Position": node["Layer_Position"],
            "Projection_Index": node["Projection_Index"],
            "Is_QKV": node["Is_QKV"],
            "Is_AttentionOut": node["Is_AttentionOut"],
            "Is_FFNUp": node["Is_FFNUp"],
            "Is_FFNDown": node["Is_FFNDown"],
            "Has_Bias": node["Has_Bias"],
            "Previous_Linear_Weight_Bytes": node[
                "Previous_Linear_Weight_Bytes"
            ],
            "Previous_Op_Gap_Bytes": node["Previous_Op_Gap_Bytes"],
            "Is_Transformer_Context": 1,
        }
        booster = (
            self.linear_context_median
            if prediction_mode == "median"
            else self.linear_context_90th
        )
        features = pd.DataFrame([row]).reindex(columns=booster.feature_name())
        if features.isna().any().any():
            missing = features.columns[features.isna().any()].tolist()
            raise RuntimeError("Missing Transformer Linear features: " + ", ".join(missing))
        ratio_log = booster.predict(features)[0]
        predicted_ms = theoretical_ms * np.expm1(ratio_log)
        return predicted_ms, theoretical_ms


def main():
    base.FXGraphParser = LinearContextFXGraphParser
    base.SimulatorRouter = LinearContextSimulatorRouter
    base.prediction_family = memory_v4.prediction_family
    base.run_all_simulations(target_mode="eager")


if __name__ == "__main__":
    main()
