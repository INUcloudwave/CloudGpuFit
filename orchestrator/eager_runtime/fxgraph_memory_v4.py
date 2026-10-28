import math
import operator
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

import fxgraph_cache_v3 as cache_v3
import fxgraph_reduction_v2 as base


MEMORY_CONVERSION_METHODS = {"to", "type_as", "half", "float"}
METADATA_METHODS = {"view", "transpose", "permute"}


def first_tensor_meta(value):
    if isinstance(value, torch.fx.Node):
        meta = value.meta.get("tensor_meta")
        if hasattr(meta, "shape"):
            return meta
    if isinstance(value, (tuple, list)):
        for item in value:
            meta = first_tensor_meta(item)
            if meta is not None:
                return meta
    if isinstance(value, dict):
        for item in value.values():
            meta = first_tensor_meta(item)
            if meta is not None:
                return meta
    return None


def dtype_bytes(dtype):
    sizes = {
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
    }
    return sizes.get(dtype, 4)


def meta_is_contiguous(meta):
    expected = 1
    for size, stride in zip(reversed(meta.shape), reversed(meta.stride)):
        if size == 1:
            continue
        if stride != expected:
            return False
        expected *= size
    return True


def inner_contiguous_elements(meta):
    expected = 1
    elements = 1
    for size, stride in zip(reversed(meta.shape), reversed(meta.stride)):
        if size == 1:
            elements *= size
            continue
        if stride != expected:
            break
        elements *= size
        expected *= size
    return elements


def contiguous_output_meta(input_meta):
    shape = tuple(input_meta.shape)
    stride = []
    current = 1
    for size in reversed(shape):
        stride.append(current)
        current *= size
    return type(
        "OutputMeta",
        (),
        {
            "shape": shape,
            "stride": tuple(reversed(stride)),
            "dtype": input_meta.dtype,
        },
    )()


def make_memory_node(input_meta, output_meta, method, implicit_bmm=False):
    input_shape = tuple(int(value) for value in input_meta.shape)
    output_shape = tuple(int(value) for value in output_meta.shape)
    input_elements = math.prod(input_shape)
    output_elements = math.prod(output_shape)
    input_precision = dtype_bytes(input_meta.dtype)
    output_precision = dtype_bytes(output_meta.dtype)
    input_bytes = input_elements * input_precision
    output_bytes = output_elements * output_precision
    ideal_bytes = input_bytes + output_bytes
    contiguous = meta_is_contiguous(input_meta)
    rank = len(output_shape)
    padded = list(output_shape[:4]) + [1] * max(0, 4 - rank)
    transpose_last2 = int(
        rank >= 2
        and not contiguous
        and input_meta.stride[-2] == 1
    )
    permute_axis12 = int(not contiguous and not transpose_last2)

    return {
        "type": "MemoryCopy",
        "method": method,
        "elements": output_elements,
        "input_elements": input_elements,
        "output_elements": output_elements,
        "Batch_Size_From_Shape": output_shape[0] if output_shape else 1,
        "Rank": rank,
        "Dim_0": padded[0],
        "Dim_1": padded[1],
        "Dim_2": padded[2],
        "Dim_3": padded[3],
        "Input_Bytes": input_bytes,
        "Output_Bytes": output_bytes,
        "Ideal_Memory_Bytes": ideal_bytes,
        "Input_Precision_Bytes": input_precision,
        "Output_Precision_Bytes": output_precision,
        "Source_Contiguous": int(contiguous),
        "Inner_Contiguous_Elements": inner_contiguous_elements(input_meta),
        "Is_Transpose_Last2": transpose_last2,
        "Is_Permute_Axis12": permute_axis12,
        "Is_Dtype_Conversion": int(input_meta.dtype != output_meta.dtype),
        "Is_Implicit_BMM_Copy": int(implicit_bmm),
        "Is_Fused": 0,
    }


def fusion_group(node):
    return {
        "type": "FusionGroup",
        "base_op": node,
        "absorbed_base_ops": [],
        "tail_ops": [],
    }


class MemoryAwareFXGraphParser(cache_v3.CacheAwareFXGraphParser):
    def _parse_nodes(self, gm, engine_mode):
        if engine_mode != "eager":
            raise ValueError("fxgraph_memory_v4.py supports eager mode only")

        original_make_light_node = base.make_light_node

        def routed_light_node(op_type, node, output_elements, out_shape, module=None):
            if op_type != "MemoryFormat":
                return original_make_light_node(
                    op_type,
                    node,
                    output_elements,
                    out_shape,
                    module,
                )

            method = str(node.target)
            if method in METADATA_METHODS:
                return None

            input_meta = first_tensor_meta(node.args)
            output_meta = node.meta.get("tensor_meta")
            if input_meta is None or not hasattr(output_meta, "shape"):
                return None

            if method == "reshape":
                if meta_is_contiguous(input_meta):
                    return None
                if not meta_is_contiguous(output_meta):
                    return None
            if method == "contiguous" and meta_is_contiguous(input_meta):
                return None
            return make_memory_node(input_meta, output_meta, method)

        base.make_light_node = routed_light_node
        try:
            parsed = super()._parse_nodes(gm, engine_mode)
        finally:
            base.make_light_node = original_make_light_node

        extra_groups = []
        for node in gm.graph.nodes:
            method = str(node.target)
            if node.op == "call_method" and method in MEMORY_CONVERSION_METHODS:
                input_meta = first_tensor_meta(node.args)
                output_meta = node.meta.get("tensor_meta")
                if input_meta is None or not hasattr(output_meta, "shape"):
                    continue
                if input_meta.dtype == output_meta.dtype:
                    continue
                extra_groups.append(
                    fusion_group(
                        make_memory_node(input_meta, output_meta, method)
                    )
                )

            if node.op == "call_function" and node.target in {
                torch.bmm,
                torch.matmul,
                operator.matmul,
            }:
                for argument in node.args[:2]:
                    input_meta = first_tensor_meta(argument)
                    if input_meta is None or meta_is_contiguous(input_meta):
                        continue
                    extra_groups.append(
                        fusion_group(
                            make_memory_node(
                                input_meta,
                                contiguous_output_meta(input_meta),
                                "implicit_bmm_copy",
                                implicit_bmm=True,
                            )
                        )
                    )

        return parsed + extra_groups


class MemoryAwareSimulatorRouter(cache_v3.CacheAwareSimulatorRouter):
    def __init__(self, engine_mode="eager"):
        super().__init__(engine_mode=engine_mode)
        model_dir = Path(os.environ.get("MEMORY_MODEL_DIR", "memory_model"))
        self.memory_median = lgb.Booster(
            model_file=str(model_dir / "memory_eager_median.pkl")
        )
        self.memory_90th = lgb.Booster(
            model_file=str(model_dir / "memory_eager_90th.pkl")
        )

    def _predict_node(
        self,
        node,
        batch_size,
        prediction_mode,
        force_memory_zero=False,
    ):
        if node.get("type") != "MemoryCopy":
            return super()._predict_node(
                node,
                batch_size,
                prediction_mode,
                force_memory_zero,
            )

        ideal_bytes = 0 if force_memory_zero else node["Ideal_Memory_Bytes"]
        theoretical_ms = max(
            ideal_bytes / self.peak_bandwidth * 1000,
            1e-6,
        )
        row = {
            "Batch_Size": batch_size,
            "Rank": node["Rank"],
            "Dim_0": node["Dim_0"],
            "Dim_1": node["Dim_1"],
            "Dim_2": node["Dim_2"],
            "Dim_3": node["Dim_3"],
            "Total_Elements": node["output_elements"],
            "Input_Bytes": 0 if force_memory_zero else node["Input_Bytes"],
            "Output_Bytes": 0 if force_memory_zero else node["Output_Bytes"],
            "Ideal_Memory_Bytes": ideal_bytes,
            "Estimated_L2_Fit": int(ideal_bytes <= self.l2_cache_bytes),
            "Input_Precision_Bytes": node["Input_Precision_Bytes"],
            "Output_Precision_Bytes": node["Output_Precision_Bytes"],
            "Source_Contiguous": node["Source_Contiguous"],
            "Inner_Contiguous_Elements": node["Inner_Contiguous_Elements"],
            "Is_Transpose_Last2": node["Is_Transpose_Last2"],
            "Is_Permute_Axis12": node["Is_Permute_Axis12"],
            "Is_Dtype_Conversion": node["Is_Dtype_Conversion"],
        }
        booster = (
            self.memory_median
            if prediction_mode == "median"
            else self.memory_90th
        )
        features = pd.DataFrame([row]).reindex(columns=booster.feature_name())
        ratio_log = booster.predict(features)[0]
        predicted_ms = theoretical_ms * np.expm1(ratio_log)
        return predicted_ms, theoretical_ms

    def _prepare_node(self, node, batch_size, force_memory_zero=False):
        """Batch-API twin of _predict_node (same features, no predict)."""
        if node.get("type") != "MemoryCopy":
            return super()._prepare_node(node, batch_size, force_memory_zero)

        ideal_bytes = 0 if force_memory_zero else node["Ideal_Memory_Bytes"]
        theoretical_ms = max(
            ideal_bytes / self.peak_bandwidth * 1000,
            1e-6,
        )
        row = {
            "Batch_Size": batch_size,
            "Rank": node["Rank"],
            "Dim_0": node["Dim_0"],
            "Dim_1": node["Dim_1"],
            "Dim_2": node["Dim_2"],
            "Dim_3": node["Dim_3"],
            "Total_Elements": node["output_elements"],
            "Input_Bytes": 0 if force_memory_zero else node["Input_Bytes"],
            "Output_Bytes": 0 if force_memory_zero else node["Output_Bytes"],
            "Ideal_Memory_Bytes": ideal_bytes,
            "Estimated_L2_Fit": int(ideal_bytes <= self.l2_cache_bytes),
            "Input_Precision_Bytes": node["Input_Precision_Bytes"],
            "Output_Precision_Bytes": node["Output_Precision_Bytes"],
            "Source_Contiguous": node["Source_Contiguous"],
            "Inner_Contiguous_Elements": node["Inner_Contiguous_Elements"],
            "Is_Transpose_Last2": node["Is_Transpose_Last2"],
            "Is_Permute_Axis12": node["Is_Permute_Axis12"],
            "Is_Dtype_Conversion": node["Is_Dtype_Conversion"],
        }
        return base.BatchMLRequest(
            "v4_memory", {"median": self.memory_median, "90th": self.memory_90th}, row, True, None,
            theoretical_ms, lambda y, t=theoretical_ms: t * np.expm1(y),
        )


original_prediction_family = base.prediction_family


def prediction_family(layer_type):
    if layer_type == "MemoryCopy":
        return "memory"
    return original_prediction_family(layer_type)


def main():
    base.FXGraphParser = MemoryAwareFXGraphParser
    base.SimulatorRouter = MemoryAwareSimulatorRouter
    base.prediction_family = prediction_family
    base.run_all_simulations(target_mode="eager")


if __name__ == "__main__":
    main()
