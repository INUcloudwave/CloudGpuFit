#!/usr/bin/env python3
"""Compiled external-kernel prediction directly from an FX graph.

This router predicts only kernels that Inductor dispatches to external CUDA
libraries: GEMM, cuDNN convolution, and fused SDPA.  Triton-generated kernels
remain the responsibility of ``CompiledFusionKernelRouter``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp

from compiled_dispatch import is_degenerate_inductor_matmul


SCHEMA_VERSION = 2
PREDICTOR_VERSION = "compiled_external_v2"

GEMM_FEATURES = (
    "Batch_Size",
    "Dim_M",
    "Dim_N",
    "Dim_K",
    "Total_FLOPs",
    "Ideal_Memory_Bytes",
    "Arithmetic_Intensity",
    "Estimated_L2_Fit",
    "Estimated_Wave_Count",
    "Precision_Bytes",
    "Is_AddMM",
    "Is_MM",
    "Is_BMM",
    "A_Last_Stride",
    "B_Last_Stride",
    "A_Contiguous",
    "B_Contiguous",
    "B_Transposed",
    "Output_Elements",
)

CONV_FEATURES = (
    "Batch_Size",
    "In_Channels",
    "Out_Channels",
    "Input_Height",
    "Input_Width",
    "Kernel_Height",
    "Kernel_Width",
    "Output_Height",
    "Output_Width",
    "Groups",
    "Stride_Height",
    "Stride_Width",
    "Padding_Height",
    "Padding_Width",
    "Dilation_Height",
    "Dilation_Width",
    "Total_FLOPs",
    "Ideal_Memory_Bytes",
    "Arithmetic_Intensity",
    "Estimated_L2_Fit",
    "Estimated_Wave_Count",
    "Precision_Bytes",
    "Is_Depthwise",
    "Is_Pointwise",
    "Is_Conv1D",
    "Output_Elements",
)

SDPA_FEATURES = (
    "Batch_Size",
    "Num_Query_Heads",
    "Num_KV_Heads",
    "Query_Length",
    "Key_Length",
    "Head_Dim",
    "GQA_Ratio",
    "Mask_Elements",
    "Has_Mask",
    "Is_Flash",
    "Total_FLOPs",
    "Ideal_Memory_Bytes",
    "Arithmetic_Intensity",
    "Estimated_L2_Fit",
    "Estimated_Wave_Count",
    "Precision_Bytes",
    "Output_Elements",
)

FEATURES = {
    "gemm": GEMM_FEATURES,
    "conv1d": CONV_FEATURES,
    "conv2d": CONV_FEATURES,
    "sdpa": SDPA_FEATURES,
}


class CompiledFXTracer:
    """Trace models without constructing any Eager latency-model objects."""

    def trace_model(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor | dict[str, torch.Tensor],
        is_hf: bool = False,
    ) -> torch.fx.GraphModule:
        model.eval()
        if is_hf:
            from transformers.utils.fx import symbolic_trace as hf_symbolic_trace

            input_names = (
                list(dummy_input.keys())
                if isinstance(dummy_input, dict)
                else ["input_ids"]
            )
            return hf_symbolic_trace(model, input_names=input_names)
        return torch.fx.symbolic_trace(model)


def propagate_graph_shapes(
    gm: torch.fx.GraphModule,
    dummy_input: torch.Tensor | dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        if isinstance(dummy_input, dict):
            ShapeProp(gm).propagate(*dummy_input.values())
        else:
            ShapeProp(gm).propagate(dummy_input)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def external_model_root(runtime_root: Path, target: str) -> Path:
    return runtime_root / "targets" / target / "compiled_external_model"


def load_external_manifest(runtime_root: Path, target: str) -> dict[str, Any]:
    root = external_model_root(runtime_root, target)
    path = root / "compiled_external_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Compiled external manifest is missing: {path}. "
            "Collect external calls and train the Compiled external models first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported Compiled external schema: {path}")
    if payload.get("predictor_version") != PREDICTOR_VERSION:
        raise ValueError(f"Unsupported Compiled external predictor: {path}")
    if payload.get("target_instance") != target:
        raise ValueError(
            f"Compiled external target mismatch: requested={target}, "
            f"artifact={payload.get('target_instance')}"
        )
    if payload.get("model_name_is_feature") is not False:
        raise ValueError("Compiled external model must not use model name as a feature")
    if payload.get("global_context_scale") is not None:
        raise ValueError("Compiled external model unexpectedly contains a global scale")
    for family, feature_names in FEATURES.items():
        actual = payload.get("families", {}).get(family, {}).get("feature_schema")
        if actual != list(feature_names):
            raise ValueError(
                f"Compiled external {family} feature schema mismatch"
            )
    for artifact in payload.get("artifacts", {}).values():
        artifact_path = root / artifact["path"]
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if artifact_path.stat().st_size != int(artifact["size_bytes"]):
            raise RuntimeError(
                f"Compiled external artifact size mismatch: {artifact_path}"
            )
        if sha256(artifact_path) != artifact["sha256"]:
            raise RuntimeError(
                f"Compiled external artifact hash mismatch: {artifact_path}"
            )
    return payload


def _load_booster(path: Path) -> lgb.Booster:
    return lgb.Booster(model_str=path.read_text(encoding="utf-8"))


def _argument(node: torch.fx.Node, name: str, index: int, default: Any = None) -> Any:
    if name in node.kwargs:
        return node.kwargs[name]
    if len(node.args) > index:
        return node.args[index]
    return default


def _tensor_meta(value: Any) -> Any | None:
    if not isinstance(value, torch.fx.Node):
        return None
    meta = value.meta.get("tensor_meta")
    return meta if hasattr(meta, "shape") else None


def _shape(value: Any) -> tuple[int, ...] | None:
    meta = _tensor_meta(value)
    if meta is None:
        return None
    return tuple(int(item) for item in meta.shape)


def _stride(value: Any) -> tuple[int, ...] | None:
    meta = _tensor_meta(value)
    stride = getattr(meta, "stride", None) if meta is not None else None
    if stride is None:
        return None
    return tuple(int(item) for item in stride)


def _product(shape: tuple[int, ...] | list[int]) -> int:
    return int(math.prod(int(item) for item in shape))


def _pair(value: Any, default: int = 1) -> tuple[int, int]:
    if value is None:
        return default, default
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _contiguous(shape: tuple[int, ...], stride: tuple[int, ...] | None) -> bool:
    if stride is None or len(shape) != len(stride):
        return False
    expected = 1
    for size, current in zip(reversed(shape), reversed(stride)):
        if size > 1 and current != expected:
            return False
        expected *= max(size, 1)
    return True


def _first_tensor_argument(node: torch.fx.Node) -> torch.fx.Node | None:
    for value in node.args:
        if isinstance(value, torch.fx.Node) and _shape(value) is not None:
            return value
    return None


def _target_name(node: torch.fx.Node) -> str:
    target = node.target
    name = getattr(target, "__name__", None)
    text = str(name if name else target).lower()
    for suffix in (".default", ".tensor", ".scalar"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.rsplit(".", 1)[-1]


def _base_features(
    *,
    flops: float,
    ideal_bytes: float,
    output_elements: int,
    hardware: dict[str, Any],
) -> dict[str, float]:
    return {
        "Total_FLOPs": float(flops),
        "Ideal_Memory_Bytes": float(ideal_bytes),
        "Arithmetic_Intensity": float(flops / max(ideal_bytes, 1.0)),
        "Estimated_L2_Fit": float(
            ideal_bytes <= float(hardware["l2_cache_bytes"])
        ),
        "Precision_Bytes": float(hardware["precision_bytes"]),
        "Output_Elements": float(output_elements),
    }


def _gemm_features(
    *,
    op: str,
    batch: int,
    m: int,
    n: int,
    k: int,
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    a_stride: tuple[int, ...] | None,
    b_stride: tuple[int, ...] | None,
    hardware: dict[str, Any],
) -> dict[str, float]:
    output_elements = batch * m * n
    flops = 2 * batch * m * n * k
    elements = batch * m * k + batch * k * n + output_elements
    ideal_bytes = elements * int(hardware["precision_bytes"])
    features = _base_features(
        flops=flops,
        ideal_bytes=ideal_bytes,
        output_elements=output_elements,
        hardware=hardware,
    )
    features.update(
        {
            "Batch_Size": float(batch),
            "Dim_M": float(m),
            "Dim_N": float(n),
            "Dim_K": float(k),
            "Estimated_Wave_Count": float(
                batch
                * math.ceil(m / 128.0)
                * math.ceil(n / 128.0)
                / int(hardware["sm_count"])
            ),
            "Is_AddMM": float(op == "addmm"),
            "Is_MM": float(op == "mm"),
            "Is_BMM": float(op == "bmm"),
            "A_Last_Stride": float(a_stride[-1] if a_stride else 0),
            "B_Last_Stride": float(b_stride[-1] if b_stride else 0),
            "A_Contiguous": float(_contiguous(a_shape, a_stride)),
            "B_Contiguous": float(_contiguous(b_shape, b_stride)),
            "B_Transposed": float(
                bool(b_stride) and len(b_stride) >= 2 and b_stride[-2] == 1
            ),
        }
    )
    return features


def _module_linear_record(
    node: torch.fx.Node,
    module: nn.Linear,
    hardware: dict[str, Any],
) -> dict[str, Any] | None:
    source = _first_tensor_argument(node)
    input_shape = _shape(source)
    if input_shape is None or not input_shape:
        return None
    k = int(module.in_features)
    n = int(module.out_features)
    if input_shape[-1] != k:
        return None
    m = _product(input_shape[:-1]) if len(input_shape) > 1 else 1
    op = "addmm" if module.bias is not None else "mm"
    a_shape = (m, k)
    a_stride = (k, 1)
    b_shape = (k, n)
    b_stride = (1, k)
    return {
        "family": "gemm",
        "operation": f"aten::{op}",
        "node_name": node.name,
        "features": _gemm_features(
            op=op,
            batch=1,
            m=m,
            n=n,
            k=k,
            a_shape=a_shape,
            b_shape=b_shape,
            a_stride=a_stride,
            b_stride=b_stride,
            hardware=hardware,
        ),
    }


def _function_gemm_record(
    node: torch.fx.Node,
    name: str,
    hardware: dict[str, Any],
) -> dict[str, Any] | None:
    if name == "linear":
        left = _argument(node, "input", 0)
        weight = _argument(node, "weight", 1)
        bias = _argument(node, "bias", 2)
        left_shape = _shape(left)
        weight_shape = _shape(weight)
        if left_shape is None or weight_shape is None or len(weight_shape) != 2:
            return None
        m = _product(left_shape[:-1]) if len(left_shape) > 1 else 1
        k = int(weight_shape[1])
        n = int(weight_shape[0])
        op = "addmm" if bias is not None else "mm"
        a_shape, a_stride = (m, k), (k, 1)
        b_shape, b_stride = (k, n), (1, k)
    else:
        offset = 1 if name == "addmm" else 0
        left = _argument(node, "mat1", offset)
        right = _argument(node, "mat2", offset + 1)
        left_shape = _shape(left)
        right_shape = _shape(right)
        if (
            left_shape is None
            or right_shape is None
            or len(left_shape) < 2
            or len(right_shape) < 2
        ):
            return None
        m, k = int(left_shape[-2]), int(left_shape[-1])
        n = int(right_shape[-1])
        if int(right_shape[-2]) != k:
            return None
        op = name
        if name == "matmul":
            op = "bmm" if max(len(left_shape), len(right_shape)) >= 3 else "mm"
        if is_degenerate_inductor_matmul(name, left_shape, right_shape):
            return None
        batch = (
            max(
                _product(left_shape[:-2]) if len(left_shape) > 2 else 1,
                _product(right_shape[:-2]) if len(right_shape) > 2 else 1,
            )
            if op == "bmm"
            else 1
        )
        a_shape, b_shape = left_shape, right_shape
        a_stride, b_stride = _stride(left), _stride(right)
        return {
            "family": "gemm",
            "operation": f"aten::{op}",
            "node_name": node.name,
            "features": _gemm_features(
                op=op,
                batch=batch,
                m=m,
                n=n,
                k=k,
                a_shape=a_shape,
                b_shape=b_shape,
                a_stride=a_stride,
                b_stride=b_stride,
                hardware=hardware,
            ),
        }
    return {
        "family": "gemm",
        "operation": f"aten::{op}",
        "node_name": node.name,
        "features": _gemm_features(
            op=op,
            batch=1,
            m=m,
            n=n,
            k=k,
            a_shape=a_shape,
            b_shape=b_shape,
            a_stride=a_stride,
            b_stride=b_stride,
            hardware=hardware,
        ),
    }


def _conv_features(
    *,
    input_shape: tuple[int, int, int, int],
    weight_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    is_conv1d: bool,
    hardware: dict[str, Any],
) -> dict[str, float]:
    batch, in_channels, input_height, input_width = input_shape
    out_channels, channels_per_group, kernel_height, kernel_width = weight_shape
    _, _, output_height, output_width = output_shape
    output_elements = batch * out_channels * output_height * output_width
    flops = (
        2
        * output_elements
        * channels_per_group
        * kernel_height
        * kernel_width
    )
    ideal_bytes = (
        _product(input_shape) + _product(weight_shape) + output_elements
    ) * int(hardware["precision_bytes"])
    features = _base_features(
        flops=flops,
        ideal_bytes=ideal_bytes,
        output_elements=output_elements,
        hardware=hardware,
    )
    features.update(
        {
            "Batch_Size": float(batch),
            "In_Channels": float(in_channels),
            "Out_Channels": float(out_channels),
            "Input_Height": float(input_height),
            "Input_Width": float(input_width),
            "Kernel_Height": float(kernel_height),
            "Kernel_Width": float(kernel_width),
            "Output_Height": float(output_height),
            "Output_Width": float(output_width),
            "Groups": float(groups),
            "Stride_Height": float(stride[0]),
            "Stride_Width": float(stride[1]),
            "Padding_Height": float(padding[0]),
            "Padding_Width": float(padding[1]),
            "Dilation_Height": float(dilation[0]),
            "Dilation_Width": float(dilation[1]),
            "Estimated_Wave_Count": float(
                math.ceil(output_elements / 256.0) / int(hardware["sm_count"])
            ),
            "Is_Depthwise": float(groups == in_channels and groups > 1),
            "Is_Pointwise": float(kernel_height == 1 and kernel_width == 1),
            "Is_Conv1D": float(is_conv1d),
        }
    )
    return features


def _module_conv_record(
    node: torch.fx.Node,
    module: nn.Conv1d | nn.Conv2d,
    hardware: dict[str, Any],
) -> dict[str, Any] | None:
    source = _first_tensor_argument(node)
    source_shape = _shape(source)
    result_shape = _shape(node)
    is_conv1d = isinstance(module, nn.Conv1d)
    if source_shape is None or result_shape is None:
        return None
    if is_conv1d:
        if len(source_shape) != 3 or len(result_shape) != 3:
            return None
        input_shape = (source_shape[0], source_shape[1], 1, source_shape[2])
        output_shape = (result_shape[0], result_shape[1], 1, result_shape[2])
        weight_shape = (
            int(module.out_channels),
            int(module.in_channels // module.groups),
            1,
            int(module.kernel_size[0]),
        )
        stride = (1, int(module.stride[0]))
        padding = (0, int(module.padding[0]))
        dilation = (1, int(module.dilation[0]))
    else:
        if len(source_shape) != 4 or len(result_shape) != 4:
            return None
        input_shape = tuple(source_shape)
        output_shape = tuple(result_shape)
        weight_shape = tuple(int(item) for item in module.weight.shape)
        stride = _pair(module.stride)
        padding = _pair(module.padding, default=0)
        dilation = _pair(module.dilation)
    return {
        "family": "conv",
        "operation": "aten::cudnn_convolution",
        "node_name": node.name,
        "features": _conv_features(
            input_shape=input_shape,
            weight_shape=weight_shape,
            output_shape=output_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=int(module.groups),
            is_conv1d=is_conv1d,
            hardware=hardware,
        ),
    }


def _function_conv_record(
    node: torch.fx.Node,
    name: str,
    hardware: dict[str, Any],
) -> dict[str, Any] | None:
    source = _argument(node, "input", 0)
    weight = _argument(node, "weight", 1)
    source_shape = _shape(source)
    weight_shape_raw = _shape(weight)
    result_shape = _shape(node)
    if source_shape is None or weight_shape_raw is None or result_shape is None:
        return None
    is_conv1d = len(source_shape) == 3 or name == "conv1d"
    if is_conv1d:
        if len(source_shape) != 3 or len(weight_shape_raw) != 3 or len(result_shape) != 3:
            return None
        input_shape = (source_shape[0], source_shape[1], 1, source_shape[2])
        weight_shape = (
            weight_shape_raw[0], weight_shape_raw[1], 1, weight_shape_raw[2]
        )
        output_shape = (result_shape[0], result_shape[1], 1, result_shape[2])
        stride_1d = _argument(node, "stride", 3, 1)
        padding_1d = _argument(node, "padding", 4, 0)
        dilation_1d = _argument(node, "dilation", 5, 1)
        stride = (1, _pair(stride_1d)[0])
        padding = (0, _pair(padding_1d, default=0)[0])
        dilation = (1, _pair(dilation_1d)[0])
        groups_index = 8 if name in {"convolution", "_convolution"} else 6
        groups = int(_argument(node, "groups", groups_index, 1))
    else:
        if len(source_shape) != 4 or len(weight_shape_raw) != 4 or len(result_shape) != 4:
            return None
        input_shape = tuple(source_shape)
        weight_shape = tuple(weight_shape_raw)
        output_shape = tuple(result_shape)
        stride = _pair(_argument(node, "stride", 3, 1))
        padding = _pair(_argument(node, "padding", 4, 0), default=0)
        dilation = _pair(_argument(node, "dilation", 5, 1))
        groups_index = 8 if name in {"convolution", "_convolution"} else 6
        groups = int(_argument(node, "groups", groups_index, 1))
    return {
        "family": "conv",
        "operation": "aten::cudnn_convolution",
        "node_name": node.name,
        "features": _conv_features(
            input_shape=input_shape,
            weight_shape=weight_shape,
            output_shape=output_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            is_conv1d=is_conv1d,
            hardware=hardware,
        ),
    }


def _sdpa_record(
    node: torch.fx.Node,
    hardware: dict[str, Any],
) -> dict[str, Any] | None:
    query = _argument(node, "query", 0)
    key = _argument(node, "key", 1)
    value = _argument(node, "value", 2)
    mask = _argument(node, "attn_mask", 3)
    query_shape = _shape(query)
    key_shape = _shape(key)
    value_shape = _shape(value)
    if (
        query_shape is None
        or key_shape is None
        or value_shape is None
        or len(query_shape) < 4
        or len(key_shape) < 4
    ):
        return None
    batch = int(query_shape[-4])
    query_heads = int(query_shape[-3])
    query_length = int(query_shape[-2])
    head_dim = int(query_shape[-1])
    kv_heads = int(key_shape[-3])
    key_length = int(key_shape[-2])
    mask_shape = _shape(mask)
    mask_elements = _product(mask_shape) if mask_shape else 0
    output_elements = batch * query_heads * query_length * head_dim
    score_elements = batch * query_heads * query_length * key_length
    flops = 4 * score_elements * head_dim
    ideal_elements = (
        _product(query_shape)
        + _product(key_shape)
        + _product(value_shape)
        + output_elements
        + mask_elements
    )
    ideal_bytes = ideal_elements * int(hardware["precision_bytes"])
    features = _base_features(
        flops=flops,
        ideal_bytes=ideal_bytes,
        output_elements=output_elements,
        hardware=hardware,
    )
    is_flash = mask_elements == 0
    features.update(
        {
            "Batch_Size": float(batch),
            "Num_Query_Heads": float(query_heads),
            "Num_KV_Heads": float(kv_heads),
            "Query_Length": float(query_length),
            "Key_Length": float(key_length),
            "Head_Dim": float(head_dim),
            "GQA_Ratio": float(query_heads / max(kv_heads, 1)),
            "Mask_Elements": float(mask_elements),
            "Has_Mask": float(mask_elements > 0),
            "Is_Flash": float(is_flash),
            "Estimated_Wave_Count": float(
                batch
                * query_heads
                * math.ceil(query_length / 128.0)
                * math.ceil(key_length / 128.0)
                / int(hardware["sm_count"])
            ),
        }
    )
    operation = (
        "aten::_flash_attention_forward"
        if is_flash
        else "aten::_efficient_attention_forward"
    )
    return {
        "family": "sdpa",
        "operation": operation,
        "node_name": node.name,
        "features": features,
    }


def _record_for_node(
    node: torch.fx.Node,
    modules: dict[str, nn.Module],
    hardware: dict[str, Any],
) -> dict[str, Any] | None:
    if node.op == "call_module":
        module = modules[str(node.target)]
        if isinstance(module, nn.Linear):
            return _module_linear_record(node, module, hardware)
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            return _module_conv_record(node, module, hardware)
        return None
    if node.op not in {"call_function", "call_method"}:
        return None
    name = str(node.target).lower() if node.op == "call_method" else _target_name(node)
    if "scaled_dot_product_attention" in name:
        return _sdpa_record(node, hardware)
    if name in {"addmm", "mm", "bmm", "matmul", "linear"}:
        return _function_gemm_record(node, name, hardware)
    if name in {"conv1d", "conv2d", "convolution", "_convolution"}:
        return _function_conv_record(node, name, hardware)
    return None


class CompiledExternalKernelRouter:
    def __init__(
        self,
        *,
        runtime_root: Path,
        target: str,
        hardware_profile: dict[str, Any],
    ):
        self.root = external_model_root(runtime_root, target)
        self.manifest = load_external_manifest(runtime_root, target)
        self.hardware = hardware_profile
        self.models: dict[tuple[str, str], lgb.Booster] = {}
        for family, feature_names in FEATURES.items():
            for mode in ("median", "p90"):
                artifact = self.manifest["artifacts"][f"{family}_{mode}"]
                model = _load_booster(self.root / artifact["path"])
                if model.feature_name() != list(feature_names):
                    raise ValueError(
                        f"Compiled external {family}/{mode} LightGBM feature "
                        "order mismatch"
                    )
                self.models[(family, mode)] = model

    def extract_graph(self, gm: torch.fx.GraphModule) -> list[dict[str, Any]]:
        modules = dict(gm.named_modules())
        records = []
        for node in gm.graph.nodes:
            record = _record_for_node(node, modules, self.hardware)
            if record is not None:
                records.append(record)
        return records

    def predict_graph(
        self,
        gm: torch.fx.GraphModule,
        prediction_mode: str,
    ) -> list[dict[str, Any]]:
        if prediction_mode not in {"median", "p90"}:
            raise ValueError(f"Unknown prediction mode: {prediction_mode}")
        predictions = []
        for record in self.extract_graph(gm):
            family = record["family"]
            features = record["features"]
            model_family = family
            if family == "conv":
                model_family = (
                    "conv1d" if float(features["Is_Conv1D"]) > 0.5 else "conv2d"
                )
            feature_names = FEATURES[model_family]
            frame = pd.DataFrame([features], columns=feature_names)
            if frame.isna().any().any():
                missing = frame.columns[frame.isna().any()].tolist()
                raise ValueError(
                    f"Missing Compiled external {model_family} features: "
                    + ", ".join(missing)
                )
            roofline_ms = max(
                float(features["Total_FLOPs"])
                / float(self.hardware["peak_flops"])
                * 1000.0,
                float(features["Ideal_Memory_Bytes"])
                / float(self.hardware["peak_bandwidth"])
                * 1000.0,
                1e-9,
            )
            floor_ms = float(
                self.manifest["families"][model_family]["kernel_floor_ms"]
            )
            theoretical_ms = max(roofline_ms, floor_ms)
            model = self.models[(model_family, prediction_mode)]
            ratio = max(float(np.expm1(model.predict(frame)[0])), 0.0)
            predicted_ms = theoretical_ms * ratio
            if not math.isfinite(predicted_ms) or predicted_ms <= 0.0:
                raise RuntimeError(
                    f"Invalid Compiled external prediction: {predicted_ms}"
                )
            predictions.append(
                {
                    **record,
                    "model_family": model_family,
                    "prediction_mode": prediction_mode,
                    "roofline_ms": roofline_ms,
                    "kernel_floor_ms": floor_ms,
                    "theoretical_ms": theoretical_ms,
                    "ratio": ratio,
                    "predicted_ms": predicted_ms,
                }
            )
        return predictions
