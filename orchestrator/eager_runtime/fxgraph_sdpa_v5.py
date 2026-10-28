import math
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import fxgraph_memory_v4 as memory_v4
import fxgraph_reduction_v2 as base


SM_COUNT = 58


def argument(node, name, index, default=None):
    if name in node.kwargs:
        return node.kwargs[name]
    if len(node.args) > index:
        return node.args[index]
    return default


def tensor_meta(value):
    if isinstance(value, torch.fx.Node):
        meta = value.meta.get("tensor_meta")
        if hasattr(meta, "shape"):
            return meta
    return None


def literal_bool(value, default=False):
    if isinstance(value, (bool, int)):
        return bool(value)
    return default


def mask_type(value):
    if value is None:
        return 0
    meta = tensor_meta(value)
    dtype = getattr(meta, "dtype", getattr(value, "dtype", None))
    return 2 if dtype == torch.bool else 1


def sdpa_graph_nodes(gm):
    return [
        node
        for node in gm.graph.nodes
        if node.op == "call_function"
        and "scaled_dot_product_attention" in str(node.target)
    ]


def parsed_sdpa_nodes(parsed):
    result = []
    for group in parsed:
        operation = group.get("base_op", group)
        if operation.get("type") == "BMM_SDPA":
            result.append(operation)
    return result


def annotate_sdpa(operation, graph_node):
    query_meta = tensor_meta(argument(graph_node, "query", 0))
    key_meta = tensor_meta(argument(graph_node, "key", 1))
    if query_meta is None or key_meta is None:
        return

    query_shape = tuple(int(value) for value in query_meta.shape)
    key_shape = tuple(int(value) for value in key_meta.shape)
    if len(query_shape) < 4 or len(key_shape) < 4:
        return

    batch_size = query_shape[-4]
    query_heads = query_shape[-3]
    query_length = query_shape[-2]
    head_dim = query_shape[-1]
    kv_heads = key_shape[-3]
    key_length = key_shape[-2]
    current_mask_type = mask_type(argument(graph_node, "attn_mask", 3))

    operation.update(
        {
            "Use_SDPA_Specialist": 1,
            "SDPA_Batch_Size": batch_size,
            "Num_Query_Heads": query_heads,
            "Num_KV_Heads": kv_heads,
            "Query_Length": query_length,
            "Key_Length": key_length,
            "Head_Dim": head_dim,
            "Mask_Type": current_mask_type,
            "Has_Mask": int(current_mask_type != 0),
            "Is_Causal": int(
                literal_bool(argument(graph_node, "is_causal", 5), False)
            ),
            "Is_GQA": int(
                query_heads != kv_heads
                or literal_bool(argument(graph_node, "enable_gqa", 7), False)
            ),
            "Precision_Bytes": memory_v4.dtype_bytes(query_meta.dtype),
        }
    )


class SDPAAwareFXGraphParser(memory_v4.MemoryAwareFXGraphParser):
    def _parse_nodes(self, gm, engine_mode):
        parsed = super()._parse_nodes(gm, engine_mode)
        graph_nodes = sdpa_graph_nodes(gm)
        operations = parsed_sdpa_nodes(parsed)
        for operation, graph_node in zip(operations, graph_nodes):
            annotate_sdpa(operation, graph_node)
        if len(graph_nodes) != len(operations):
            print(
                "SDPA routing warning: "
                f"FX nodes={len(graph_nodes)}, parsed nodes={len(operations)}"
            )
        return parsed


class SDPAAwareSimulatorRouter(memory_v4.MemoryAwareSimulatorRouter):
    def __init__(self, engine_mode="eager"):
        super().__init__(engine_mode=engine_mode)
        model_dir = Path(os.environ.get("SDPA_MODEL_DIR", "sdpa_model"))
        self.sdpa_median = lgb.Booster(
            model_file=str(model_dir / "sdpa_eager_median.pkl")
        )
        self.sdpa_90th = lgb.Booster(
            model_file=str(model_dir / "sdpa_eager_90th.pkl")
        )

    def _predict_node(
        self,
        node,
        batch_size,
        prediction_mode,
        force_memory_zero=False,
    ):
        if node.get("type") != "BMM_SDPA" or not node.get(
            "Use_SDPA_Specialist", 0
        ):
            return super()._predict_node(
                node,
                batch_size,
                prediction_mode,
                force_memory_zero,
            )

        batch = node["SDPA_Batch_Size"]
        query_heads = node["Num_Query_Heads"]
        kv_heads = node["Num_KV_Heads"]
        query_length = node["Query_Length"]
        key_length = node["Key_Length"]
        head_dim = node["Head_Dim"]
        precision = node["Precision_Bytes"]
        score_elements = batch * query_heads * query_length * key_length
        total_flops = 4 * score_elements * head_dim + 5 * score_elements

        query_elements = batch * query_heads * query_length * head_dim
        kv_elements = 2 * batch * kv_heads * key_length * head_dim
        output_elements = query_elements
        mask_elements = (
            batch * query_length * key_length if node["Has_Mask"] else 0
        )
        ideal_bytes = (
            query_elements + kv_elements + output_elements + mask_elements
        ) * precision
        theoretical_ms = max(
            total_flops / self.peak_flops * 1000,
            ideal_bytes / self.peak_bandwidth * 1000,
            1e-6,
        )
        wave_count = (
            batch
            * query_heads
            * math.ceil(query_length / 128.0)
            * math.ceil(key_length / 128.0)
            / SM_COUNT
        )
        row = {
            "Batch_Size": batch,
            "Num_Query_Heads": query_heads,
            "Num_KV_Heads": kv_heads,
            "Query_Length": query_length,
            "Key_Length": key_length,
            "Head_Dim": head_dim,
            "Mask_Type": node["Mask_Type"],
            "Has_Mask": node["Has_Mask"],
            "Is_Causal": node["Is_Causal"],
            "Is_GQA": node["Is_GQA"],
            "Precision_Bytes": precision,
            "Total_FLOPs": total_flops,
            "Ideal_Memory_Bytes": ideal_bytes,
            "Arithmetic_Intensity": total_flops / ideal_bytes,
            "Estimated_L2_Fit": int(ideal_bytes <= self.l2_cache_bytes),
            "Estimated_Wave_Count": wave_count,
        }
        booster = (
            self.sdpa_median
            if prediction_mode == "median"
            else self.sdpa_90th
        )
        features = pd.DataFrame([row]).reindex(columns=booster.feature_name())
        ratio = max(float(np.expm1(booster.predict(features)[0])), 0.0)
        return theoretical_ms * ratio, theoretical_ms

    def _prepare_node(self, node, batch_size, force_memory_zero=False):
        """Batch-API twin of _predict_node (same features, no predict)."""
        if node.get("type") != "BMM_SDPA" or not node.get(
            "Use_SDPA_Specialist", 0
        ):
            return super()._prepare_node(node, batch_size, force_memory_zero)

        batch = node["SDPA_Batch_Size"]
        query_heads = node["Num_Query_Heads"]
        kv_heads = node["Num_KV_Heads"]
        query_length = node["Query_Length"]
        key_length = node["Key_Length"]
        head_dim = node["Head_Dim"]
        precision = node["Precision_Bytes"]
        score_elements = batch * query_heads * query_length * key_length
        total_flops = 4 * score_elements * head_dim + 5 * score_elements

        query_elements = batch * query_heads * query_length * head_dim
        kv_elements = 2 * batch * kv_heads * key_length * head_dim
        output_elements = query_elements
        mask_elements = (
            batch * query_length * key_length if node["Has_Mask"] else 0
        )
        ideal_bytes = (
            query_elements + kv_elements + output_elements + mask_elements
        ) * precision
        theoretical_ms = max(
            total_flops / self.peak_flops * 1000,
            ideal_bytes / self.peak_bandwidth * 1000,
            1e-6,
        )
        wave_count = (
            batch
            * query_heads
            * math.ceil(query_length / 128.0)
            * math.ceil(key_length / 128.0)
            / SM_COUNT
        )
        row = {
            "Batch_Size": batch,
            "Num_Query_Heads": query_heads,
            "Num_KV_Heads": kv_heads,
            "Query_Length": query_length,
            "Key_Length": key_length,
            "Head_Dim": head_dim,
            "Mask_Type": node["Mask_Type"],
            "Has_Mask": node["Has_Mask"],
            "Is_Causal": node["Is_Causal"],
            "Is_GQA": node["Is_GQA"],
            "Precision_Bytes": precision,
            "Total_FLOPs": total_flops,
            "Ideal_Memory_Bytes": ideal_bytes,
            "Arithmetic_Intensity": total_flops / ideal_bytes,
            "Estimated_L2_Fit": int(ideal_bytes <= self.l2_cache_bytes),
            "Estimated_Wave_Count": wave_count,
        }
        return base.BatchMLRequest(
            "v5_sdpa", {"median": self.sdpa_median, "90th": self.sdpa_90th}, row, True, None,
            theoretical_ms, lambda y, t=theoretical_ms: t * max(float(np.expm1(y)), 0.0),
        )


def main():
    base.FXGraphParser = SDPAAwareFXGraphParser
    base.SimulatorRouter = SDPAAwareSimulatorRouter
    base.prediction_family = memory_v4.prediction_family
    base.run_all_simulations(target_mode="eager")


if __name__ == "__main__":
    main()
