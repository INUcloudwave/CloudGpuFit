"""Generate phase routing: export Graph -> existing Eager / Compiled predictors.

Derived from bundle generate_export_adapter_v22.py (CLI parts, the analytical
CompiledGenerateAnalyticalRouter and the fixed model configs removed).
Changes vs the bundle:
  * model_dimensions() reads the request config's own fields instead of
    branching on a model name (values identical for OPT/Qwen2/Llama/T5/BART).
  * Compiled uses the Generate-only planner (generate/compiled_fusion_ir.py)
    with the target's existing external / fusion / boundary models.
  * prediction_mode is a parameter so the diagnostic P90 can reuse the
    existing P90 operation models ("90th" Eager, "p90" Compiled).
No new constants, scales or ML models.
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import operator
import os
import re
from pathlib import Path
from typing import Any

import torch

import fxgraph_host_v9 as host_v9
import fxgraph_linear_v6 as linear_v6
from compiled_external_model import CompiledExternalKernelRouter
from compiled_fusion_boundary_model import load_boundary_predictor
from compiled_fusion_model import CompiledFusionKernelRouter

from .compiled_fusion_ir import CompiledFusionPlanner, canonical_name  # Generate-only planner
from .generate_eager_host_model_v33 import (
    feature_row as eager_host_feature_row,
    optional_host_model,
    schedule_with_overlap,
)

PRECISION_BYTES = 2
PROJECTION_INDEX = {
    "Q": 0,
    "K": 1,
    "V": 2,
    "AttentionOut": 3,
    "FFNUp": 4,
    "FFNDown": 5,
}



_EAGER_MODEL_ENV = {
    # env name -> default relative path when the runtime manifest has none
    "EAGER_BASE_MODEL_DIR": ".",
    "CACHE_MODEL_DIR": "cache_model",
    "MEMORY_MODEL_DIR": "memory_model",
    "SDPA_MODEL_DIR": "sdpa_model",
    "LINEAR_CONTEXT_MODEL_DIR": "linear_context_model_deep_v2",
    "POINTWISE_MODEL_DIR": "pointwise_model_v4",
    "EAGER_HARDWARE_PROFILE": "hardware_profile.json",
}


def configure_eager_target(root: Path, target: str) -> None:
    """Operation-model locations come from the target's runtime_manifest.json
    `environment` (relative to the target folder), exactly as the existing
    Prefill path resolves them (runtime_manifest.configure_environment). A
    thin size target such as gr6.4xlarge points its L4 operation models at
    ../g6.xlarge there; full targets list their own folders. The Generate
    files (specialist profile, host model) always come from the target's own
    folder."""
    target_root = root / "targets" / target
    manifest_env = {}
    manifest_path = target_root / "runtime_manifest.json"
    if manifest_path.is_file():
        manifest_env = json.loads(manifest_path.read_text(encoding="utf-8")).get("environment", {}) or {}
    mapping = {"EAGER_TARGET_ROOT": target_root}
    for name, default in _EAGER_MODEL_ENV.items():
        mapping[name] = target_root / str(manifest_env.get(name, default))
    mapping["GENERATE_SPECIALIST_PROFILE"] = target_root / "generate_specialist_profile.json"
    for name, value in mapping.items():
        os.environ[name] = str(value.resolve())


class ExportGenerateEagerRouter(host_v9.HostScheduleSimulatorRouter):
    def __init__(self):
        super().__init__(engine_mode="eager")
        path = Path(os.environ["GENERATE_SPECIALIST_PROFILE"])
        profile = json.loads(path.read_text(encoding="utf-8"))
        schedule = profile.get("generate_host_schedule", {})
        if schedule:
            self.t_dispatch_light = float(schedule["t_dispatch_light_ms"])
            self.t_dispatch_heavy = float(schedule["t_dispatch_heavy_ms"])
            self.t_sync = float(schedule.get("t_sync_ms", self.t_sync))
        target_root = Path(os.environ["EAGER_TARGET_ROOT"])
        self.generate_host_model = (
            None
            if os.environ.get("DISABLE_GENERATE_EAGER_HOST_MODEL") == "1"
            else optional_host_model(target_root)
        )


def node_meta(value: Any) -> Any:
    if not isinstance(value, torch.fx.Node):
        return None
    meta = value.meta.get("tensor_meta")
    return meta if hasattr(meta, "shape") else None


def shape(value: Any) -> tuple[int, ...] | None:
    meta = node_meta(value)
    return tuple(int(item) for item in meta.shape) if meta is not None else None


def elements(value: Any) -> int:
    current = shape(value)
    return int(math.prod(current)) if current is not None else 0


def dtype_bytes(dtype: torch.dtype | None) -> int:
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


def target_name(node: torch.fx.Node) -> str:
    target = node.target
    name = str(getattr(target, "__name__", None) or target).lower()
    normalized = name.replace("::", ".")
    if normalized.startswith("aten."):
        parts = normalized.split(".")
        if len(parts) >= 2:
            return parts[1]
    return name.split(".", 1)[0]


def module_path(node: torch.fx.Node) -> str:
    stack = node.meta.get("nn_module_stack")
    if not isinstance(stack, dict):
        return ""
    paths = []
    for key, value in stack.items():
        if isinstance(value, tuple) and value:
            paths.append(str(value[0]))
        else:
            paths.append(str(key))
    return paths[-1] if paths else ""


def layer_index(path: str) -> int | None:
    match = re.search(r"(?:layers|layer|block)\.(\d+)(?:\.|$)", path.lower())
    return int(match.group(1)) if match else None


def projection(path: str) -> str | None:
    tail = path.lower().rsplit(".", 1)[-1]
    if tail in {"q", "query", "q_proj"}:
        return "Q"
    if tail in {"k", "key", "k_proj"}:
        return "K"
    if tail in {"v", "value", "v_proj"}:
        return "V"
    if tail in {"o", "out_proj", "o_proj"}:
        return "AttentionOut"
    if tail in {"fc1", "wi", "wi_0", "wi_1", "gate_proj", "up_proj"}:
        return "FFNUp"
    if tail in {"fc2", "wo", "down_proj"}:
        return "FFNDown"
    return None



def model_dimensions(config: Any) -> tuple[int, int, int]:
    """(hidden, intermediate, depth) from the config's own fields.

    Same values as bundle v22 for its models: T5 -> (d_model, d_ff,
    num_decoder_layers), BART -> (d_model, decoder_ffn_dim, decoder_layers),
    decoder-only -> (hidden_size, intermediate_size or ffn_dim,
    num_hidden_layers). A missing field raises (routing failure), it is never
    approximated.
    """
    def first(*names):
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        raise KeyError(f"config has none of {names}")

    if getattr(config, "is_encoder_decoder", False):
        return (
            first("d_model", "hidden_size"),
            first("d_ff", "decoder_ffn_dim", "ffn_dim", "intermediate_size"),
            first("num_decoder_layers", "decoder_layers", "num_hidden_layers"),
        )
    return (
        first("hidden_size"),
        first("intermediate_size", "ffn_dim"),
        first("num_hidden_layers"),
    )


def linear_record(
    node: torch.fx.Node,
    config: Any,
    batch: int,
) -> dict[str, Any] | None:
    if len(node.args) < 2:
        return None
    source_shape = shape(node.args[0])
    weight_shape = shape(node.args[1])
    if source_shape is None or weight_shape is None or len(weight_shape) != 2:
        return None
    m = math.prod(source_shape[:-1]) if len(source_shape) > 1 else 1
    k = int(weight_shape[1])
    n = int(weight_shape[0])
    path = module_path(node)
    current_projection = projection(path)
    layer = layer_index(path)
    record = {
        "type": "Linear",
        "M": int(m),
        "N": n,
        "K": k,
        "kernel_size": 1,
        "Is_Conv1D": 0,
        "Is_Conv2D": 0,
        "Is_BMM": 0,
        "Is_Fused": 0,
        "Is_LM_Head": int("lm_head" in path.lower()),
    }
    if current_projection is None or layer is None:
        return record
    hidden, intermediate, depth = model_dimensions(config)
    sequence = max(int(m) // max(batch, 1), 1)
    has_bias = int(len(node.args) > 2 and node.args[2] is not None)
    context_bytes = linear_v6._context_working_set(
        batch, sequence, hidden, intermediate, depth, has_bias
    )
    weight_bytes = k * n * PRECISION_BYTES
    reuse_distance = max(context_bytes - weight_bytes, 0)
    record.update(
        {
            "Use_Transformer_Linear_Context": 1,
            "Is_Transformer_Context": 1,
            "Sequence_Length": sequence,
            "Hidden_Size": hidden,
            "Intermediate_Size": intermediate,
            "Block_Depth": depth,
            "Layer_Position": layer / max(depth - 1, 1),
            "Projection_Index": PROJECTION_INDEX[current_projection],
            "Is_QKV": int(current_projection in {"Q", "K", "V"}),
            "Is_AttentionOut": int(current_projection == "AttentionOut"),
            "Is_FFNUp": int(current_projection == "FFNUp"),
            "Is_FFNDown": int(current_projection == "FFNDown"),
            "Has_Bias": has_bias,
            "Previous_Linear_Weight_Bytes": linear_v6._previous_weight_bytes(
                current_projection, hidden, intermediate
            ),
            "Previous_Op_Gap_Bytes": linear_v6._previous_gap_bytes(
                current_projection, int(m), hidden, intermediate
            ),
            "Weight_Bytes": weight_bytes,
            "RHS_Is_Transposed": 1,
            "Context_Working_Set_Bytes": context_bytes,
            "Reuse_Distance_Bytes": reuse_distance,
            "Estimated_Weight_L2_Resident": int(
                weight_bytes + reuse_distance <= 48 * 1024 * 1024
            ),
        }
    )
    return record


def bmm_record(node: torch.fx.Node) -> dict[str, Any] | None:
    if len(node.args) < 2:
        return None
    left = shape(node.args[0])
    right = shape(node.args[1])
    output = shape(node)
    if left is None or right is None or output is None or len(left) < 2:
        return None
    batch_dim = max(
        math.prod(left[:-2]) if len(left) > 2 else 1,
        math.prod(right[:-2]) if len(right) > 2 else 1,
    )
    m_true, k, n = int(left[-2]), int(left[-1]), int(output[-1])
    return {
        "type": "BMM",
        "M": int(batch_dim * m_true),
        "N": n,
        "K": k,
        "M_true": m_true,
        "B_dim": int(batch_dim),
        "kernel_size": 1,
        "Is_Conv1D": 0,
        "Is_Conv2D": 0,
        "Is_BMM": 1,
        "Is_Fused": 0,
    }


def baddbmm_record(node: torch.fx.Node) -> dict[str, Any] | None:
    """baddbmm(input, batch1, batch2): one BMM over batch1 @ batch2 (the
    fused add is part of the same kernel, not counted again)."""
    if len(node.args) < 3:
        return None
    left = shape(node.args[1])
    right = shape(node.args[2])
    output = shape(node)
    if left is None or right is None or output is None or len(left) != 3 or len(right) != 3:
        return None
    batch_dim, m_true, k = (int(v) for v in left)
    n = int(right[-1])
    return {
        "type": "BMM",
        "M": int(batch_dim * m_true),
        "N": n,
        "K": k,
        "M_true": m_true,
        "B_dim": int(batch_dim),
        "kernel_size": 1,
        "Is_Conv1D": 0,
        "Is_Conv2D": 0,
        "Is_BMM": 1,
        "Is_Fused": 0,
    }


def light_record(node: torch.fx.Node, operation: str) -> dict[str, Any]:
    output_shape = shape(node) or (1,)
    output_elements = int(math.prod(output_shape))
    inputs = [value for value in node.args if isinstance(value, torch.fx.Node)]
    input_elements = [elements(value) for value in inputs]
    input_total = sum(input_elements)
    output_meta = node_meta(node)
    precision = dtype_bytes(output_meta.dtype if output_meta else None)
    if operation == "MemoryCopy":
        source_meta = node_meta(inputs[0]) if inputs else None
        input_precision = dtype_bytes(source_meta.dtype if source_meta else None)
        dims = list(output_shape[:4]) + [1] * max(0, 4 - len(output_shape))
        return {
            "type": operation,
            "Rank": len(output_shape),
            "Dim_0": dims[0],
            "Dim_1": dims[1],
            "Dim_2": dims[2],
            "Dim_3": dims[3],
            "output_elements": output_elements,
            "Input_Bytes": (input_elements[0] if input_elements else output_elements) * input_precision,
            "Output_Bytes": output_elements * precision,
            "Ideal_Memory_Bytes": ((input_elements[0] if input_elements else output_elements) * input_precision) + output_elements * precision,
            "Input_Precision_Bytes": input_precision,
            "Output_Precision_Bytes": precision,
            "Source_Contiguous": 1,
            "Inner_Contiguous_Elements": int(output_shape[-1]),
            "Is_Transpose_Last2": 0,
            "Is_Permute_Axis12": 0,
            "Is_Dtype_Conversion": int(input_precision != precision),
            "Is_Fused": 0,
        }
    if operation.startswith("Pointwise_"):
        op = operation.removeprefix("Pointwise_")
        binary = int(len(inputs) >= 2)
        input1 = input_elements[1] if len(input_elements) >= 2 else 0
        dims = list(output_shape[:4]) + [1] * max(0, 4 - len(output_shape))
        input0 = input_elements[0] if input_elements else output_elements
        read_bytes = (input0 + input1) * precision
        write_bytes = output_elements * precision
        transcendental = {"GELU", "SiLU", "Tanh", "Sigmoid", "Sqrt", "Rsqrt", "Pow2", "Log"}
        activation = {"GELU", "ReLU", "ReLU6", "SiLU", "Tanh", "Sigmoid", "ClampMin"}
        return {
            "type": operation,
            "pointwise_operation": op,
            "Use_Pointwise_Context": 1,
            "Batch_Size_From_Shape": output_shape[0],
            "Rank": len(output_shape),
            "Dim_0": dims[0],
            "Dim_1": dims[1],
            "Dim_2": dims[2],
            "Dim_3": dims[3],
            "Input0_Elements": input0,
            "Input1_Elements": input1,
            "Output_Elements": output_elements,
            "Read_Bytes": read_bytes,
            "Write_Bytes": write_bytes,
            "Ideal_Memory_Bytes": read_bytes + write_bytes,
            "Input0_Reuse_Distance_Bytes": 0,
            "Input1_Reuse_Distance_Bytes": 0,
            "Input0_Bytes": input0 * precision,
            "Input1_Bytes": input1 * precision,
            "Broadcast_Ratio": output_elements / max(input1, 1) if binary else 1.0,
            "Is_Binary": binary,
            "Is_Broadcast": int(binary and input1 not in {0, output_elements}),
            "Is_Scalar_RHS": int(binary and input1 == 0),
            "Is_Comparison": int(op == "Compare"),
            "Is_Mask": int(op in {"MaskedFill", "Where"}),
            "Is_Activation": int(op in activation),
            "Is_Transcendental": int(op in transcendental),
            "Precision_Bytes": precision,
            "Is_Fused": 0,
        }
    reduction_size = max(input_total // max(output_elements, 1), 1)
    return {
        "type": operation,
        "elements": output_elements,
        "input_elements": input_total or output_elements,
        "output_elements": output_elements,
        "parameter_elements": 0,
        "reduction_size": reduction_size,
        "is_binary": int(len(inputs) >= 2),
        "is_adaptive_pool": 0,
        "index_input": int(operation == "Embedding"),
        "Is_Fused": 0,
    }


METADATA = {
    "_assert_tensor_metadata",
    "alias",
    "detach",
    "detach_",
    "dim",
    "expand",
    "getitem",
    "permute",
    "reshape",
    "select",
    "slice",
    "sym_size",
    "transpose",
    "unsqueeze",
    "view",
    "wrap_with_set_grad_enabled",
}
MEMORY = {"cat", "clone", "contiguous", "copy", "copy_", "repeat", "to", "type_as"}
REDUCTIONS = {
    "argmax": "Reduction",
    "any": "Reduction",
    "cumsum": "Reduction",
    "layer_norm": "LayerNorm",
    "max": "Reduction",
    "mean": "Reduction",
    "min": "Reduction",
    "softmax": "Softmax",
    "sum": "Reduction",
}
POINTWISE = {
    "abs": "Abs",
    "add": "Add",
    "add_": "Add",
    "arange": "Fill",
    "clamp": "ClampMin",
    "div": "Div",
    "eq": "Compare",
    "full": "Fill",
    "full_like": "Fill",
    "gelu": "GELU",
    "gt": "Compare",
    "isinf": "Compare",
    "le": "Compare",
    "log": "Log",
    "lt": "Compare",
    "masked_fill": "MaskedFill",
    "masked_fill_": "MaskedFill",
    "mul": "Mul",
    "mul_": "Mul",
    "neg": "Mul",
    "ones": "Fill",
    "pow": "Pow2",
    "relu": "ReLU",
    "rsqrt": "Rsqrt",
    "rsub": "Add",
    "silu": "SiLU",
    "tanh": "Tanh",  # existing Pointwise Tanh category (GELU tanh approximation etc.)
    "sub": "Add",
    "triu": "MaskedFill",
    "where": "Where",
    "zeros": "Fill",
    "zeros_like": "Fill",
}


def eager_groups(
    gm: torch.fx.GraphModule,
    config: Any,
    batch: int,
    lifted_constants: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], list[str]]:
    groups = []
    unsupported = []
    for node in gm.graph.nodes:
        if node.op in {"placeholder", "get_attr", "output"}:
            continue
        name = target_name(node)
        record = None
        if name == "dropout":
            continue
        if name == "lift_fresh_copy":
            source = node.args[0] if node.args else None
            if isinstance(source, torch.fx.Node) and source.op == "placeholder" and source.name in lifted_constants:
                continue  # copy of a static lifted constant -> metadata
            unsupported.append(name)  # dynamic tensor copy: not routed
            continue
        if name == "baddbmm":
            record = baddbmm_record(node)
            if record is None:
                unsupported.append(name)
                continue
        if name in METADATA or node.target is operator.getitem:
            continue
        if record is not None:
            pass
        elif name == "linear":
            record = linear_record(node, config, batch)
        elif name in {"bmm", "matmul", "mm", "addmm"}:
            record = bmm_record(node)
        elif name == "embedding":
            record = light_record(node, "Embedding")
            if len(node.args) >= 1:
                record["parameter_elements"] = elements(node.args[0])
            if len(node.args) >= 2:
                record["input_elements"] = elements(node.args[1])
        elif name in MEMORY:
            record = light_record(node, "MemoryCopy")
        elif name in REDUCTIONS:
            record = light_record(node, REDUCTIONS[name])
        elif name in POINTWISE:
            record = light_record(node, f"Pointwise_{POINTWISE[name]}")
        elif node_meta(node) is not None:
            unsupported.append(name)
        if record is not None:
            groups.append(
                {
                    "type": "FusionGroup",
                    "base_op": record,
                    "absorbed_base_ops": [],
                    "tail_ops": [],
                }
            )
    return groups, sorted(set(unsupported))


def schedule_eager(router: ExportGenerateEagerRouter, log_df: Any) -> float:
    cpu_time = float(router.t_pytorch_overhead)
    gpu_time = 0.0
    for _, row in log_df.iterrows():
        heavy = row["Layer_Type"] in {"Linear", "BMM", "BMM_SDPA"}
        dispatch = router.t_dispatch_heavy if heavy else router.t_dispatch_light
        cpu_time += router.t_python_loop + dispatch
        gpu_time = max(gpu_time, cpu_time) + float(row["Predicted_Actual_ms"])
    return gpu_time


def _eager_result(router, gm, groups, unsupported, log_df, config, batch, *, phase, seq_len, cache_len):
    baseline_ms = float(schedule_eager(router, log_df))
    framework_extra_ms = 0.0
    scheduled_ms = baseline_ms
    host_model_applied = 0
    if router.generate_host_model is not None:
        features = eager_host_feature_row(
            gm,
            log_df,
            config,
            router,
            phase=phase,
            batch_size=batch,
            seq_len=seq_len,
            cache_len=cache_len,
        )
        framework_extra_ms = router.generate_host_model.predict(features)
        scheduled_ms = schedule_with_overlap(log_df, router, framework_extra_ms)
        host_model_applied = 1
    return {
        "routed_nodes": len(groups),
        "unsupported_operations": unsupported,
        "kernel_sum_ms": float(log_df["Predicted_Actual_ms"].sum()),
        "baseline_scheduled_ms": baseline_ms,
        "predicted_framework_host_ms": float(framework_extra_ms),
        "host_model_applied": host_model_applied,
        "host_schedule_method": "per_node_cpu_gpu_overlap",
        "scheduled_ms": float(scheduled_ms),
        "by_type": {
            str(key): int(value)
            for key, value in log_df.groupby("Layer_Type").size().items()
        },
    }


def predict_eager(
    router: ExportGenerateEagerRouter,
    gm: torch.fx.GraphModule,
    config: Any,
    batch: int,
    *,
    phase: str,
    seq_len: int,
    cache_len: int,
    prediction_mode: str = "median",
    lifted_constants: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Scalar path (one prediction mode, existing simulate_fx_graph)."""
    groups, unsupported = eager_groups(gm, config, batch, lifted_constants)
    _, log_df = router.simulate_fx_graph(groups, batch_size=batch, prediction_mode=prediction_mode)
    return _eager_result(router, gm, groups, unsupported, log_df, config, batch,
                         phase=phase, seq_len=seq_len, cache_len=cache_len)


def predict_eager_modes(
    router: ExportGenerateEagerRouter,
    gm: torch.fx.GraphModule,
    config: Any,
    batch: int,
    *,
    phase: str,
    seq_len: int,
    cache_len: int,
    prediction_modes: tuple[str, ...] = ("median", "90th"),
    lifted_constants: frozenset[str] = frozenset(),
    use_batch: bool = True,
) -> dict[str, dict[str, Any]]:
    """Median and P90 from one routed graph. Batch path: the common routers'
    opt-in simulate_fx_graph_batch (feature rows shared, one predict() per
    specialist group and mode). Scheduling and host model unchanged."""
    groups, unsupported = eager_groups(gm, config, batch, lifted_constants)
    if use_batch:
        simulated = router.simulate_fx_graph_batch(groups, batch, prediction_modes)
    else:
        simulated = {
            mode: router.simulate_fx_graph(groups, batch_size=batch, prediction_mode=mode)
            for mode in prediction_modes
        }
    return {
        mode: _eager_result(router, gm, groups, unsupported, log_df, config, batch,
                            phase=phase, seq_len=seq_len, cache_len=cache_len)
        for mode, (_total, log_df) in simulated.items()
    }


def compiled_semantic_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    result = copy.deepcopy(gm)
    for node in result.graph.nodes:
        if target_name(node) != "embedding" or len(node.args) < 2:
            continue
        node.target = torch.clone
        node.args = (node.args[1],)
        node.kwargs = {}
    result.graph.lint()
    return result


def predict_compiled(
    router: "CompiledGenerateRouter",
    gm: torch.fx.GraphModule,
    prediction_mode: str = "median",
    lifted_constants: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    graph = compiled_semantic_graph(gm)
    external = router.external.predict_graph(graph, prediction_mode)
    plan = router.planner.plan(graph, lifted_constants)
    # existing single-group API, called once per group (same values as the
    # bundle's batched call)
    fusion = [router.fusion.predict_group(group, prediction_mode) for group in plan.groups]
    external_ms = sum(float(item["predicted_ms"]) for item in external)
    fusion_ms = sum(float(item["predicted_ms"]) for item in fusion)
    host_ms = (
        len(external) * float(router.hardware["t_dispatch_heavy"])
        + len(fusion) * float(router.hardware["t_dispatch_light"])
    )
    return {
        "external_kernel_count": len(external),
        "fusion_group_count": len(fusion),
        "unsupported_operations": sorted(
            {item["target"] for item in plan.unsupported}
        ),
        "kernel_sum_ms": external_ms + fusion_ms,
        "scheduled_ms": max(external_ms + fusion_ms, host_ms),
        "heavy_node_count": plan.heavy_node_count,
    }



class CompiledGenerateRouter:
    """The four Compiled components of bundle v17's router, without its
    analytical graph templates."""

    def __init__(self, *, runtime_root: Path, target: str):
        self.runtime_root = Path(runtime_root).resolve()
        self.target = str(target)
        self.hardware = json.loads(
            (self.runtime_root / "targets" / self.target / "hardware_profile.json").read_text(encoding="utf-8")
        )
        self.external = CompiledExternalKernelRouter(
            runtime_root=self.runtime_root, target=self.target, hardware_profile=self.hardware
        )
        self.fusion = CompiledFusionKernelRouter(
            runtime_root=self.runtime_root, target=self.target, hardware_profile=self.hardware
        )
        self.boundary = load_boundary_predictor(self.runtime_root, self.target)
        self.planner = CompiledFusionPlanner(
            l2_cache_bytes=int(self.hardware["l2_cache_bytes"]),
            sm_count=int(self.hardware["sm_count"]),
            precision_bytes=int(self.hardware["precision_bytes"]),
            boundary_predictor=self.boundary,
        )


@contextlib.contextmanager
def eager_target_environment(root: Path, target: str):
    """configure_eager_target() for the duration of a call; previous values restored."""
    names = (
        "EAGER_TARGET_ROOT", "EAGER_BASE_MODEL_DIR", "CACHE_MODEL_DIR", "MEMORY_MODEL_DIR", "SDPA_MODEL_DIR",
        "LINEAR_CONTEXT_MODEL_DIR", "POINTWISE_MODEL_DIR", "EAGER_HARDWARE_PROFILE", "GENERATE_SPECIALIST_PROFILE",
    )
    saved = {name: os.environ.get(name) for name in names}
    configure_eager_target(root, target)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
