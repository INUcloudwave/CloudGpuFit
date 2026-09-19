#!/usr/bin/env python3

from __future__ import annotations

import math
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import fxgraph_fidelity_v8 as v8
import fxgraph_host_v9 as v9
import fxgraph_pointwise_v7 as v7
import fxgraph_reduction_v2 as base


PRECISION_BYTES = 2


def iter_base_nodes(graph):
    for item in graph:
        if item.get("type") == "FusionGroup":
            yield item["base_op"]


def node_shape(node):
    meta = getattr(node, "meta", {}).get("tensor_meta")
    if meta is not None and hasattr(meta, "shape"):
        return tuple(int(value) for value in meta.shape)
    return None


def first_input_shape(node):
    for value in node.args:
        if isinstance(value, torch.fx.Node):
            shape = node_shape(value)
            if shape is not None:
                return shape
    return None


class Conv1dContextFXGraphParser(v8.FidelityFXGraphParser):
    def extract_nodes(self, gm, dummy_input, engine_mode="eager"):
        graph = super().extract_nodes(gm, dummy_input, engine_mode)
        modules = dict(gm.named_modules())
        fx_convs = []
        for fx_node in gm.graph.nodes:
            module = modules.get(str(fx_node.target))
            if fx_node.op == "call_module" and isinstance(module, nn.Conv1d):
                fx_convs.append((fx_node, module))

        parsed_convs = [
            node for node in iter_base_nodes(graph)
            if node.get("type") == "Conv1d"
        ]
        if not parsed_convs:
            return graph
        if len(parsed_convs) != len(fx_convs):
            print(
                "⚠️ Conv1d context routing skipped: count mismatch "
                f"({len(parsed_convs)} parsed, {len(fx_convs)} FX)"
            )
            return graph

        depth = len(parsed_convs)
        previous_output_bytes = 0
        for index, (parsed, (fx_node, module)) in enumerate(zip(parsed_convs, fx_convs)):
            input_shape = first_input_shape(fx_node)
            output_shape = node_shape(fx_node)
            if input_shape is None or output_shape is None or len(output_shape) != 3:
                continue
            batch, out_channels, output_length = output_shape
            input_length = int(input_shape[-1])
            kernel = int(module.kernel_size[0])
            stride = int(module.stride[0])
            groups = int(module.groups)
            weight_bytes = module.weight.numel() * module.weight.element_size()
            parsed.update(
                {
                    "Use_Conv1d_Context": 1,
                    "Input_Length": input_length,
                    "Output_Length": output_length,
                    "In_Channels": int(module.in_channels),
                    "Out_Channels": int(module.out_channels),
                    "Kernel_Size": kernel,
                    "Stride": stride,
                    "Groups": groups,
                    "Padding": int(module.padding[0]),
                    "Layer_Index": index,
                    "Layer_Position": index / max(depth - 1, 1),
                    "Conv_Depth": depth,
                    "Previous_Output_Bytes": previous_output_bytes,
                    "Weight_Bytes": weight_bytes,
                    "Is_Grouped": int(groups > 1),
                    "Is_Sequential_Context": int(depth > 1),
                }
            )
            previous_output_bytes = (
                batch * out_channels * output_length * PRECISION_BYTES
            )

        routed = sum(node.get("Use_Conv1d_Context", 0) for node in parsed_convs)
        print(f"  Conv1d context routing: {routed}/{depth} nodes")
        return graph


class Conv1dContextSimulatorRouter(v9.HostScheduleSimulatorRouter):
    def __init__(self, engine_mode="eager"):
        super().__init__(engine_mode=engine_mode)
        model_dir = Path(os.environ.get("CONV1D_MODEL_DIR", "conv1d_model"))
        self.conv1d_median = lgb.Booster(
            model_file=str(model_dir / "conv1d_eager_median.pkl")
        )
        self.conv1d_90th = lgb.Booster(
            model_file=str(model_dir / "conv1d_eager_90th.pkl")
        )

    def _predict_node(
        self,
        node,
        batch_size,
        prediction_mode,
        force_memory_zero=False,
    ):
        if not (
            node.get("type") == "Conv1d"
            and node.get("Use_Conv1d_Context", 0) == 1
        ):
            return super()._predict_node(
                node, batch_size, prediction_mode, force_memory_zero
            )

        batch = int(batch_size)
        input_length = int(node["Input_Length"])
        output_length = int(node["Output_Length"])
        in_channels = int(node["In_Channels"])
        out_channels = int(node["Out_Channels"])
        kernel = int(node["Kernel_Size"])
        groups = int(node["Groups"])
        reduced_channels = in_channels // groups
        flops = (
            2
            * batch
            * output_length
            * out_channels
            * reduced_channels
            * kernel
        )
        input_elements = batch * in_channels * input_length
        output_elements = batch * out_channels * output_length
        ideal_bytes = (
            0
            if force_memory_zero
            else (
                input_elements
                + output_elements
                + out_channels * reduced_channels * kernel
                + out_channels
            )
            * self.precision_bytes
        )
        theoretical_ms = max(
            flops / self.peak_flops * 1000,
            ideal_bytes / self.peak_bandwidth * 1000,
            1e-6,
        )
        row = {
            "Batch_Size": batch,
            "Input_Length": input_length,
            "Output_Length": output_length,
            "In_Channels": in_channels,
            "Out_Channels": out_channels,
            "Kernel_Size": kernel,
            "Stride": node["Stride"],
            "Groups": groups,
            "Total_FLOPs": flops,
            "Ideal_Memory_Bytes": ideal_bytes,
            "Arithmetic_Intensity": flops / max(ideal_bytes, 1),
            "Estimated_L2_Fit": int(ideal_bytes <= self.l2_cache_bytes),
            "Estimated_Wave_Count": (
                math.ceil((batch * output_length) / 128)
                * math.ceil(out_channels / 128)
                / self.sm_count
            ),
            "Precision_Bytes": self.precision_bytes,
            "Is_Grouped": node["Is_Grouped"],
            "Layer_Index": node["Layer_Index"],
            "Layer_Position": node["Layer_Position"],
            "Conv_Depth": node["Conv_Depth"],
            "Previous_Output_Bytes": node["Previous_Output_Bytes"],
            "Weight_Bytes": node["Weight_Bytes"],
            "Is_Sequential_Context": node["Is_Sequential_Context"],
        }
        booster = self.conv1d_median if prediction_mode == "median" else self.conv1d_90th
        features = pd.DataFrame([row]).reindex(columns=booster.feature_name())
        if features.isna().any().any():
            names = features.columns[features.isna().any()].tolist()
            raise RuntimeError("Missing Conv1d features: " + ", ".join(names))
        ratio = max(float(np.expm1(booster.predict(features)[0])), 0.0)
        return theoretical_ms * ratio, theoretical_ms


def main():
    base.FXGraphParser = Conv1dContextFXGraphParser
    base.SimulatorRouter = Conv1dContextSimulatorRouter
    base.prediction_family = v7.prediction_family
    base.build_prediction_breakdown = v9.build_prediction_breakdown
    print("Eager Conv1d context specialist active")
    base.run_all_simulations(target_mode="eager")


if __name__ == "__main__":
    main()
