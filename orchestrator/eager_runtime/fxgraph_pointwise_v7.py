#!/usr/bin/env python3
"""Eager simulator with cache-context-aware Pointwise routing."""

from __future__ import annotations

import math
import operator
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import fxgraph_linear_v6 as linear_v6
import fxgraph_memory_v4 as memory_v4
import fxgraph_reduction_v2 as base


L2_CACHE_BYTES = 48 * 1024 * 1024
MAX_REUSE_DISTANCE_BYTES = 120 * 1024 * 1024

POINTWISE_OPERATIONS = (
    "Add",
    "Mul",
    "Div",
    "Pow2",
    "Sqrt",
    "Rsqrt",
    "GELU",
    "ReLU",
    "ReLU6",
    "SiLU",
    "Tanh",
    "Sigmoid",
    "ClampMin",
    "MaskedFill",
    "Where",
    "Minimum",
    "Compare",
)

SPECIALIST_OPERATIONS = set(POINTWISE_OPERATIONS)
METADATA_METHODS = {"view", "reshape", "transpose", "permute", "unsqueeze", "squeeze", "expand"}

LEGACY_FUNCTIONS = {
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.sqrt,
    torch.rsqrt,
    torch.pow,
    torch.exp,
    F.gelu,
    F.relu,
    F.relu6,
}

LEGACY_METHODS = {
    "pow",
    "rsqrt",
    "mul",
    "add",
    "div",
    "sqrt",
    "unsqueeze",
    "squeeze",
    "expand",
}

LEGACY_MODULES = (
    nn.ReLU,
    nn.ReLU6,
    nn.GELU,
    nn.Dropout,
)


def dtype_bytes(dtype):
    return {
        torch.bool: 1,
        torch.int8: 1,
        torch.uint8: 1,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int16: 2,
        torch.float32: 4,
        torch.int32: 4,
        torch.float64: 8,
        torch.int64: 8,
    }.get(dtype, 4)


def node_tensor_meta(node):
    if not isinstance(node, torch.fx.Node):
        return None
    meta = node.meta.get("tensor_meta")
    return meta if hasattr(meta, "shape") else None


def collect_tensor_sources(value):
    if isinstance(value, torch.fx.Node):
        return [value] if node_tensor_meta(value) is not None else []
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(collect_tensor_sources(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(collect_tensor_sources(item))
        return result
    return []


def meta_elements(meta):
    return math.prod(int(value) for value in meta.shape)


def target_name(node, modules):
    if node.op == "call_module":
        return modules[str(node.target)].__class__.__name__.lower()
    target = node.target
    if hasattr(target, "__name__"):
        return target.__name__.lower()
    return str(target).lower()


def canonical_operation(node, modules):
    name = target_name(node, modules)

    if node.op == "call_module":
        module = modules[str(node.target)]
        if isinstance(module, nn.Dropout):
            return "Dropout"
        if isinstance(module, nn.GELU):
            return "GELU"
        if isinstance(module, nn.ReLU6):
            return "ReLU6"
        if isinstance(module, nn.ReLU):
            return "ReLU"
        if isinstance(module, nn.SiLU):
            return "SiLU"
        if isinstance(module, nn.Tanh):
            return "Tanh"
        if isinstance(module, nn.Sigmoid):
            return "Sigmoid"
        return None

    if node.op == "call_method" and str(node.target) in METADATA_METHODS:
        return "Metadata"

    if "masked_fill" in name:
        return "MaskedFill"
    if "clamp_min" in name or name == "relu":
        return "ReLU" if name == "relu" else "ClampMin"
    if "relu6" in name or "hardtanh" in name:
        return "ReLU6"
    if "gelu" in name:
        return "GELU"
    if "silu" in name:
        return "SiLU"
    if "sigmoid" in name:
        return "Sigmoid"
    if "tanh" in name:
        return "Tanh"
    if "rsqrt" in name:
        return "Rsqrt"
    if name == "sqrt" or name.endswith(".sqrt"):
        return "Sqrt"
    if name == "pow" or name.endswith(".pow"):
        return "Pow2"
    if "minimum" in name:
        return "Minimum"
    if name == "where" or name.endswith(".where"):
        return "Where"
    if name in {"eq", "ne", "lt", "le", "gt", "ge"}:
        return "Compare"
    if name == "add" or name.endswith(".add"):
        return "Add"
    if name in {"sub", "rsub"} or name.endswith(".sub"):
        return "Add"
    if name == "mul" or name.endswith(".mul"):
        return "Mul"
    if name in {"div", "truediv"} or name.endswith(".div"):
        return "Div"
    return None


def is_legacy_pointwise_fx(node, modules):
    if node.op == "call_module":
        return isinstance(modules.get(str(node.target)), LEGACY_MODULES)
    if node.op == "call_function":
        return node.target in LEGACY_FUNCTIONS
    if node.op == "call_method":
        return str(node.target) in LEGACY_METHODS
    return False


def is_legacy_pointwise_group(group):
    if group.get("type") != "FusionGroup":
        return False
    op_type = str(group["base_op"].get("type", ""))
    return op_type in {"ElementWise", "GELU", "ReLU", "ReLU6", "Dropout"}


def graph_memory_state(gm, modules):
    nodes = list(gm.graph.nodes)
    positions = {node: index for index, node in enumerate(nodes)}
    output_bytes = []
    for node in nodes:
        meta = node_tensor_meta(node)
        materializes = meta is not None
        if node.op in {"placeholder", "get_attr", "output"}:
            materializes = False
        elif node.op == "call_method" and str(node.target) in METADATA_METHODS:
            materializes = False
        elif node.op == "call_module" and isinstance(
            modules.get(str(node.target)), (nn.Dropout, nn.Identity)
        ):
            materializes = False
        output_bytes.append(
            meta_elements(meta) * dtype_bytes(meta.dtype) if materializes else 0
        )
    prefix = [0]
    for value in output_bytes:
        prefix.append(prefix[-1] + value)
    return nodes, positions, prefix


def source_reuse_distance(source, consumer_index, positions, prefix):
    if source.op == "get_attr":
        return 0
    producer_index = positions.get(source)
    if producer_index is None or producer_index >= consumer_index:
        return MAX_REUSE_DISTANCE_BYTES
    distance = prefix[consumer_index] - prefix[producer_index + 1]
    return min(max(int(distance), 0), MAX_REUSE_DISTANCE_BYTES)


def pointwise_node_from_fx(fx_node, operation, positions, prefix):
    output_meta = node_tensor_meta(fx_node)
    if output_meta is None:
        return None
    output_shape = tuple(int(value) for value in output_meta.shape)
    output_elements = meta_elements(output_meta)
    if output_elements <= 0:
        return None

    sources = collect_tensor_sources((fx_node.args, fx_node.kwargs))
    unique_sources = []
    seen = set()
    for source in sources:
        if source not in seen:
            unique_sources.append(source)
            seen.add(source)

    source_data = []
    consumer_index = positions[fx_node]
    for source in unique_sources:
        meta = node_tensor_meta(source)
        elements = meta_elements(meta)
        source_data.append(
            {
                "source": source,
                "elements": elements,
                "bytes": elements * dtype_bytes(meta.dtype),
                "distance": source_reuse_distance(
                    source, consumer_index, positions, prefix
                ),
                "dtype": meta.dtype,
            }
        )

    if operation in {"MaskedFill", "Where"}:
        floats = [item for item in source_data if item["dtype"] != torch.bool]
        bools = [item for item in source_data if item["dtype"] == torch.bool]
        ordered = sorted(floats, key=lambda item: item["elements"], reverse=True)
        if bools:
            ordered.append(max(bools, key=lambda item: item["elements"]))
    else:
        ordered = sorted(source_data, key=lambda item: item["elements"], reverse=True)

    input0 = ordered[0] if ordered else None
    input1 = ordered[1] if len(ordered) > 1 else None
    input0_elements = input0["elements"] if input0 else output_elements
    input1_elements = input1["elements"] if input1 else 0
    input0_bytes = input0["bytes"] if input0 else output_elements * dtype_bytes(output_meta.dtype)
    input1_bytes = input1["bytes"] if input1 else 0
    distance0 = input0["distance"] if input0 else MAX_REUSE_DISTANCE_BYTES
    distance1 = input1["distance"] if input1 else 0
    total_read_bytes = sum(item["bytes"] for item in source_data)
    if not total_read_bytes:
        total_read_bytes = input0_bytes
    output_bytes = output_elements * dtype_bytes(output_meta.dtype)

    rank = len(output_shape)
    dims = list(output_shape[:4]) + [1] * max(0, 4 - rank)
    binary = int(operation in {"Add", "Mul", "Div", "MaskedFill", "Where", "Minimum", "Compare"})

    result = {
        "type": f"Pointwise_{operation}",
        "pointwise_operation": operation,
        "Use_Pointwise_Context": 1,
        "Batch_Size_From_Shape": output_shape[0] if output_shape else 1,
        "Rank": rank,
        "Dim_0": dims[0],
        "Dim_1": dims[1],
        "Dim_2": dims[2],
        "Dim_3": dims[3],
        "Input0_Elements": input0_elements,
        "Input1_Elements": input1_elements,
        "Output_Elements": output_elements,
        "Read_Bytes": total_read_bytes,
        "Write_Bytes": output_bytes,
        "Ideal_Memory_Bytes": total_read_bytes + output_bytes,
        "Input0_Reuse_Distance_Bytes": distance0,
        "Input1_Reuse_Distance_Bytes": distance1,
        "Input0_Bytes": input0_bytes,
        "Input1_Bytes": input1_bytes,
        "Broadcast_Ratio": output_elements / max(input1_elements, 1) if binary else 1.0,
        "Is_Binary": binary,
        "Is_Broadcast": int(binary and input1_elements not in {0, output_elements}),
        "Is_Scalar_RHS": int(binary and input1_elements == 0),
        "Is_Comparison": int(operation == "Compare"),
        "Is_Mask": int(operation in {"MaskedFill", "Where"}),
        "Is_Activation": int(operation in {"GELU", "ReLU", "ReLU6", "SiLU", "Tanh", "Sigmoid", "ClampMin"}),
        "Is_Transcendental": int(operation in {"GELU", "SiLU", "Tanh", "Sigmoid", "Sqrt", "Rsqrt", "Pow2"}),
        "Precision_Bytes": 2,
        "FX_Node": fx_node.name,
        "Is_Fused": 0,
    }
    return result


def fusion_group(node):
    return {
        "type": "FusionGroup",
        "base_op": node,
        "absorbed_base_ops": [],
        "tail_ops": [],
    }


class PointwiseContextFXGraphParser(linear_v6.LinearContextFXGraphParser):
    def extract_nodes(self, gm, dummy_input, engine_mode="eager"):
        if engine_mode != "eager":
            raise ValueError("fxgraph_pointwise_v7.py supports eager mode only")

        fused_graph = super().extract_nodes(gm, dummy_input, engine_mode)
        modules = dict(gm.named_modules())
        graph_nodes, positions, prefix = graph_memory_state(gm, modules)
        legacy_fx = [
            node for node in graph_nodes if is_legacy_pointwise_fx(node, modules)
        ]
        legacy_groups = [
            group for group in fused_graph if is_legacy_pointwise_group(group)
        ]

        if len(legacy_fx) != len(legacy_groups):
            print(
                "  Pointwise routing skipped: FX/simulator count mismatch "
                f"({len(legacy_fx)} vs {len(legacy_groups)})"
            )
            return fused_graph

        replacements = {}
        routed = 0
        skipped_dropout = 0
        skipped_metadata = 0
        skipped_non_tensor = 0

        for fx_node, group in zip(legacy_fx, legacy_groups):
            operation = canonical_operation(fx_node, modules)
            if operation == "Dropout":
                replacements[id(group)] = None
                skipped_dropout += 1
                continue
            if operation == "Metadata":
                replacements[id(group)] = None
                skipped_metadata += 1
                continue
            if operation not in SPECIALIST_OPERATIONS:
                continue
            pointwise = pointwise_node_from_fx(
                fx_node, operation, positions, prefix
            )
            if pointwise is None:
                replacements[id(group)] = None
                skipped_non_tensor += 1
                continue
            replacements[id(group)] = fusion_group(pointwise)
            routed += 1

        rebuilt = []
        for group in fused_graph:
            key = id(group)
            if key not in replacements:
                rebuilt.append(group)
            elif replacements[key] is not None:
                rebuilt.append(replacements[key])

        legacy_set = set(legacy_fx)
        added = 0
        for fx_node in graph_nodes:
            if fx_node in legacy_set:
                continue
            operation = canonical_operation(fx_node, modules)
            if operation not in SPECIALIST_OPERATIONS:
                continue
            pointwise = pointwise_node_from_fx(
                fx_node, operation, positions, prefix
            )
            if pointwise is None:
                continue
            rebuilt.append(fusion_group(pointwise))
            added += 1

        print(
            f"  Pointwise routing: {routed + added} nodes "
            f"(replaced={routed}, added={added}, dropout_zero={skipped_dropout}, "
            f"metadata_zero={skipped_metadata}, non_tensor={skipped_non_tensor})"
        )
        return rebuilt


class PointwiseContextSimulatorRouter(linear_v6.LinearContextSimulatorRouter):
    def __init__(self, engine_mode="eager"):
        super().__init__(engine_mode=engine_mode)
        model_dir = Path(
            os.environ.get("POINTWISE_MODEL_DIR", "pointwise_model")
        )
        self.pointwise_median = lgb.Booster(
            model_file=str(model_dir / "pointwise_eager_median.pkl")
        )
        self.pointwise_90th = lgb.Booster(
            model_file=str(model_dir / "pointwise_eager_90th.pkl")
        )

    def _predict_node(
        self,
        node,
        batch_size,
        prediction_mode,
        force_memory_zero=False,
    ):
        if not node.get("Use_Pointwise_Context", 0):
            return super()._predict_node(
                node,
                batch_size,
                prediction_mode,
                force_memory_zero,
            )

        operation = node["pointwise_operation"]
        ideal_bytes = 0 if force_memory_zero else node["Ideal_Memory_Bytes"]
        read_bytes = 0 if force_memory_zero else node["Read_Bytes"]
        write_bytes = 0 if force_memory_zero else node["Write_Bytes"]
        theoretical_ms = max(
            ideal_bytes / self.peak_bandwidth * 1000,
            1e-6,
        )
        distance0 = node["Input0_Reuse_Distance_Bytes"]
        distance1 = node["Input1_Reuse_Distance_Bytes"]
        input1_elements = node["Input1_Elements"]

        row = {
            "Batch_Size": batch_size,
            "Output_Elements": node["Output_Elements"],
            "Read_Bytes": read_bytes,
            "Write_Bytes": write_bytes,
            "Ideal_Memory_Bytes": ideal_bytes,
            "Estimated_L2_Fit": int(ideal_bytes <= self.l2_cache_bytes),
            "Precision_Bytes": node["Precision_Bytes"],
            "Estimated_Wave_Count": math.ceil(node["Output_Elements"] / 1024) / self.sm_count,
            "Is_Binary": node["Is_Binary"],
            "Is_Broadcast": node["Is_Broadcast"],
            "Rank": node["Rank"],
            "Dim_0": node["Dim_0"],
            "Dim_1": node["Dim_1"],
            "Dim_2": node["Dim_2"],
            "Dim_3": node["Dim_3"],
            "Input0_Elements": node["Input0_Elements"],
            "Input1_Elements": input1_elements,
            "Input0_Reuse_Distance_Bytes": distance0,
            "Input1_Reuse_Distance_Bytes": distance1,
            "Input0_L2_Resident": int(distance0 + node["Input0_Bytes"] <= self.l2_cache_bytes),
            "Input1_L2_Resident": int(input1_elements > 0 and distance1 + node["Input1_Bytes"] <= self.l2_cache_bytes),
            "Max_Reuse_Distance_Bytes": max(distance0, distance1),
            "Broadcast_Ratio": node["Broadcast_Ratio"],
            "Is_Scalar_RHS": node["Is_Scalar_RHS"],
            "Is_Comparison": node["Is_Comparison"],
            "Is_Mask": node["Is_Mask"],
            "Is_Activation": node["Is_Activation"],
            "Is_Transcendental": node["Is_Transcendental"],
        }
        for name in POINTWISE_OPERATIONS:
            row[f"Op_{name}"] = int(operation == name)

        booster = (
            self.pointwise_median
            if prediction_mode == "median"
            else self.pointwise_90th
        )
        features = pd.DataFrame([row]).reindex(columns=booster.feature_name())
        if features.isna().any().any():
            missing = features.columns[features.isna().any()].tolist()
            raise RuntimeError("Missing Pointwise features: " + ", ".join(missing))
        ratio_log = booster.predict(features)[0]
        predicted_ms = theoretical_ms * max(np.expm1(ratio_log), 0)
        return predicted_ms, theoretical_ms

    def _prepare_node(self, node, batch_size, force_memory_zero=False):
        """Batch-API twin of _predict_node (same features, no predict)."""
        if not node.get("Use_Pointwise_Context", 0):
            return super()._prepare_node(node, batch_size, force_memory_zero)

        operation = node["pointwise_operation"]
        ideal_bytes = 0 if force_memory_zero else node["Ideal_Memory_Bytes"]
        read_bytes = 0 if force_memory_zero else node["Read_Bytes"]
        write_bytes = 0 if force_memory_zero else node["Write_Bytes"]
        theoretical_ms = max(
            ideal_bytes / self.peak_bandwidth * 1000,
            1e-6,
        )
        distance0 = node["Input0_Reuse_Distance_Bytes"]
        distance1 = node["Input1_Reuse_Distance_Bytes"]
        input1_elements = node["Input1_Elements"]

        row = {
            "Batch_Size": batch_size,
            "Output_Elements": node["Output_Elements"],
            "Read_Bytes": read_bytes,
            "Write_Bytes": write_bytes,
            "Ideal_Memory_Bytes": ideal_bytes,
            "Estimated_L2_Fit": int(ideal_bytes <= self.l2_cache_bytes),
            "Precision_Bytes": node["Precision_Bytes"],
            "Estimated_Wave_Count": math.ceil(node["Output_Elements"] / 1024) / self.sm_count,
            "Is_Binary": node["Is_Binary"],
            "Is_Broadcast": node["Is_Broadcast"],
            "Rank": node["Rank"],
            "Dim_0": node["Dim_0"],
            "Dim_1": node["Dim_1"],
            "Dim_2": node["Dim_2"],
            "Dim_3": node["Dim_3"],
            "Input0_Elements": node["Input0_Elements"],
            "Input1_Elements": input1_elements,
            "Input0_Reuse_Distance_Bytes": distance0,
            "Input1_Reuse_Distance_Bytes": distance1,
            "Input0_L2_Resident": int(distance0 + node["Input0_Bytes"] <= self.l2_cache_bytes),
            "Input1_L2_Resident": int(input1_elements > 0 and distance1 + node["Input1_Bytes"] <= self.l2_cache_bytes),
            "Max_Reuse_Distance_Bytes": max(distance0, distance1),
            "Broadcast_Ratio": node["Broadcast_Ratio"],
            "Is_Scalar_RHS": node["Is_Scalar_RHS"],
            "Is_Comparison": node["Is_Comparison"],
            "Is_Mask": node["Is_Mask"],
            "Is_Activation": node["Is_Activation"],
            "Is_Transcendental": node["Is_Transcendental"],
        }
        for name in POINTWISE_OPERATIONS:
            row[f"Op_{name}"] = int(operation == name)
        return base.BatchMLRequest(
            "v7_pointwise", {"median": self.pointwise_median, "90th": self.pointwise_90th},
            row, True, "Pointwise",
            theoretical_ms, lambda y, t=theoretical_ms: t * max(np.expm1(y), 0),
        )


def prediction_family(layer_type):
    if str(layer_type).startswith("Pointwise_"):
        return "pointwise"
    return memory_v4.prediction_family(layer_type)


def main():
    base.FXGraphParser = PointwiseContextFXGraphParser
    base.SimulatorRouter = PointwiseContextSimulatorRouter
    base.prediction_family = prediction_family
    base.run_all_simulations(target_mode="eager")


if __name__ == "__main__":
    main()
