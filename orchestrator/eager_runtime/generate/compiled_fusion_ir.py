#!/usr/bin/env python3
"""DAG-aware analytical fusion groups for torch.compile prediction."""

from __future__ import annotations

import math
import operator
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from compiled_dispatch import is_degenerate_inductor_matmul


METADATA_METHODS = {
    "_assert_tensor_metadata",
    "alias",
    "detach",
    "detach_",
    "view",
    "reshape",
    "transpose",
    "permute",
    "unsqueeze",
    "squeeze",
    "expand",
    "flatten",
    "size",
    "slice",
    "select",
    "sym_size",
    "dim",
    "contiguous_format",
}

POINTWISE_NAMES = {
    "abs",
    "add",
    "clamp",
    "clamp_min",
    "clamp_max",
    "div",
    "dropout",
    "eq",
    "erf",
    "exp",
    "gelu",
    "ge",
    "gt",
    "hardtanh",
    "isinf",
    "isnan",
    "le",
    "lt",
    "masked_fill",
    "minimum",
    "maximum",
    "mul",
    "ne",
    "neg",
    "pow",
    "relu",
    "relu6",
    "rsqrt",
    "sigmoid",
    "silu",
    "sqrt",
    "sub",
    "tanh",
    "truediv",
    "where",
}

REDUCTION_NAMES = {
    "all",
    "amax",
    "amin",
    "any",
    "argmax",
    "layer_norm",
    "log_softmax",
    "mean",
    "native_layer_norm",
    "rms_norm",
    "softmax",
    "sum",
    "prod",
    "var",
}

SCAN_NAMES = {"cumprod", "cumsum", "logcumsumexp"}

MEMORY_NAMES = {"clone", "contiguous", "copy", "copy_", "to", "type_as"}

HEAVY_NAMES = {
    "addmm",
    "baddbmm",
    "bmm",
    "conv1d",
    "conv2d",
    "convolution",
    "linear",
    "matmul",
    "mm",
    "scaled_dot_product_attention",
}

TRANSCENDENTAL_NAMES = {
    "cos",
    "erf",
    "exp",
    "gelu",
    "log",
    "log_softmax",
    "rsqrt",
    "sigmoid",
    "sin",
    "silu",
    "softmax",
    "sqrt",
    "tanh",
}

INDUCTOR_POINTWISE_EXACT_NAMES = {
    "arange",
    "bool",
    "cat",
    "float",
    "full",
    "full_like",
    "int",
    "long",
    "log",
    "cos",
    "ones",
    "ones_like",
    "rand",
    "rand_like",
    "randn",
    "randn_like",
    "repeat",
    "sin",
    "tensor",
    "triu",
    "zeros",
    "zeros_like",
}

INDUCTOR_REDUCTION_EXACT_NAMES = {"max", "min"}

BINARY_NAMES = {
    "add",
    "div",
    "eq",
    "ge",
    "gt",
    "le",
    "lt",
    "maximum",
    "minimum",
    "mul",
    "ne",
    "sub",
    "truediv",
    "where",
}

OP_COST = defaultdict(
    lambda: 1.0,
    {
        "erf": 8.0,
        "exp": 6.0,
        "gelu": 10.0,
        "log": 6.0,
        "cos": 6.0,
        "layer_norm": 8.0,
        "log_softmax": 7.0,
        "native_layer_norm": 8.0,
        "rms_norm": 6.0,
        "rsqrt": 5.0,
        "sigmoid": 6.0,
        "sin": 6.0,
        "silu": 7.0,
        "softmax": 7.0,
        "sqrt": 4.0,
        "tanh": 8.0,
    },
)

FUSION_FEATURE_NAMES = (
    "Input_Bytes",
    "Output_Bytes",
    "Ideal_Memory_Bytes",
    "Estimated_FLOPs",
    "Arithmetic_Intensity",
    "Output_Elements",
    "Max_Node_Elements",
    "Reduction_Size",
    "Op_Count",
    "Pointwise_Count",
    "Reduction_Count",
    "Memory_Count",
    "Binary_Count",
    "Broadcast_Count",
    "Transcendental_Count",
    "Input_Tensor_Count",
    "Output_Tensor_Count",
    "Rank",
    "Last_Dim",
    "Estimated_L2_Fit",
    "Estimated_Wave_Count",
    "Has_Heavy_Predecessor",
    "Has_Heavy_Successor",
    "Is_Pointwise_Group",
    "Is_Reduction_Group",
    "Is_Memory_Group",
    "Precision_Bytes",
)

FUSION_EDGE_FEATURE_NAMES = (
    "Producer_Elements",
    "Consumer_Elements",
    "Element_Ratio",
    "Producer_Bytes",
    "Consumer_Bytes",
    "Same_Shape",
    "Same_Numel",
    "Rank_Delta",
    "Producer_User_Count",
    "Consumer_Input_Count",
    "Graph_Distance",
    "Producer_Is_Pointwise",
    "Producer_Is_Reduction",
    "Producer_Is_Memory",
    "Consumer_Is_Pointwise",
    "Consumer_Is_Reduction",
    "Consumer_Is_Memory",
    "Both_Reduction",
    "Has_Broadcast",
    "Producer_Is_Binary",
    "Consumer_Is_Binary",
    "Producer_Is_Transcendental",
    "Consumer_Is_Transcendental",
    "Producer_Has_Heavy_Input",
    "Consumer_Has_Heavy_Output",
    "Fits_L2",
    "Precision_Bytes",
)


class FusionBoundaryPredictor(Protocol):
    def should_fuse(self, features: dict[str, float]) -> bool: ...


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


def tensor_meta(node: torch.fx.Node):
    for key in ("tensor_meta", "example_value", "val"):
        meta = node.meta.get(key)
        if hasattr(meta, "shape") and hasattr(meta, "dtype"):
            return meta
    return None


def tensor_shape(node: torch.fx.Node) -> tuple[int, ...] | None:
    meta = tensor_meta(node)
    if meta is None:
        return None
    return tuple(int(value) for value in meta.shape)


def tensor_elements(node: torch.fx.Node) -> int:
    shape = tensor_shape(node)
    return math.prod(shape) if shape is not None else 0


def tensor_bytes(node: torch.fx.Node) -> int:
    meta = tensor_meta(node)
    return tensor_elements(node) * dtype_bytes(getattr(meta, "dtype", None))


def collect_nodes(value: Any) -> list[torch.fx.Node]:
    if isinstance(value, torch.fx.Node):
        return [value]
    if isinstance(value, (tuple, list)):
        result: list[torch.fx.Node] = []
        for item in value:
            result.extend(collect_nodes(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(collect_nodes(item))
        return result
    return []


def target_name(node: torch.fx.Node, modules: dict[str, nn.Module]) -> str:
    if node.op == "call_module":
        return modules[str(node.target)].__class__.__name__.lower()
    target = node.target
    if hasattr(target, "__name__"):
        return str(target.__name__).lower().split(".", 1)[0]
    return str(target).lower().rsplit(".", 1)[-1]


def canonical_name(node: torch.fx.Node, modules: dict[str, nn.Module]) -> str:
    name = target_name(node, modules)
    aliases = {
        "add_": "add",
        "iadd": "add",
        "mul_": "mul",
        "imul": "mul",
        "isub": "sub",
        "itruediv": "div",
        "masked_fill_": "masked_fill",
        "truediv": "div",
        "rsub": "sub",
        "layernorm": "layer_norm",
        "rmsnorm": "rms_norm",
        "qwen2rmsnorm": "rms_norm",
        "t5layernorm": "rms_norm",
    }
    for key, value in aliases.items():
        if key in name:
            return value
    qualified_name = name.replace("::", ".")
    for candidate in sorted(
        METADATA_METHODS
        | POINTWISE_NAMES
        | REDUCTION_NAMES
        | SCAN_NAMES
        | MEMORY_NAMES
        | HEAVY_NAMES,
        key=len,
        reverse=True,
    ):
        if name == candidate or re.search(
            rf"(?:^|\.){re.escape(candidate)}(?:$|\.)",
            qualified_name,
        ):
            return candidate
    return name


def is_fusion_barrier(node: torch.fx.Node, modules: dict[str, nn.Module]) -> bool:
    return canonical_name(node, modules) in SCAN_NAMES


def classify_node(node: torch.fx.Node, modules: dict[str, nn.Module]) -> str:
    if node.op in {"placeholder", "get_attr", "output"}:
        return node.op
    if tensor_meta(node) is None:
        return "metadata"
    if node.op == "call_method" and str(node.target) in METADATA_METHODS:
        return "metadata"
    if node.op == "call_function" and canonical_name(node, modules) in METADATA_METHODS:
        return "metadata"
    if node.op == "call_function" and node.target is operator.getitem:
        return "metadata"
    if node.op == "call_module":
        module = modules[str(node.target)]
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            return "heavy"
        if isinstance(module, nn.Embedding):
            return "embedding"
        if isinstance(module, (nn.Dropout, nn.Identity)):
            return "metadata"
        class_name = module.__class__.__name__.lower()
        if "norm" in class_name or isinstance(
            module, (nn.Softmax, nn.LogSoftmax, nn.AdaptiveAvgPool2d, nn.MaxPool2d)
        ):
            return "reduction"
        if isinstance(
            module,
            (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh),
        ):
            return "pointwise"
    name = canonical_name(node, modules)
    if name == "dropout":
        training = node.kwargs.get(
            "train",
            node.kwargs.get(
                "training",
                node.args[2] if len(node.args) > 2 else True,
            ),
        )
        if training is False:
            return "metadata"
    if name in {"mm", "bmm", "matmul"}:
        arguments = collect_nodes((node.args, node.kwargs))
        left_shape = tensor_shape(arguments[0]) if len(arguments) >= 1 else None
        right_shape = tensor_shape(arguments[1]) if len(arguments) >= 2 else None
        if is_degenerate_inductor_matmul(name, left_shape, right_shape):
            return "pointwise"
    if name in HEAVY_NAMES:
        return "heavy"
    if "embedding" in name:
        return "embedding"
    if name in REDUCTION_NAMES or "norm" in name or "pool" in name:
        return "reduction"
    if name in SCAN_NAMES:
        return "reduction"
    if name in INDUCTOR_REDUCTION_EXACT_NAMES:
        return "reduction"
    if name in MEMORY_NAMES:
        return "memory"
    if name in POINTWISE_NAMES:
        return "pointwise"
    if name in INDUCTOR_POINTWISE_EXACT_NAMES:
        return "pointwise"
    return "unsupported"


@dataclass(frozen=True)
class FusionPlan:
    groups: tuple[dict[str, Any], ...]
    unsupported: tuple[dict[str, str], ...]
    heavy_node_count: int
    embedding_node_count: int


class _DisjointSet:
    def __init__(self, nodes: list[torch.fx.Node]):
        self.parent = {node: node for node in nodes}

    def find(self, node: torch.fx.Node) -> torch.fx.Node:
        parent = self.parent[node]
        if parent is not node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: torch.fx.Node, right: torch.fx.Node) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root is not right_root:
            self.parent[right_root] = left_root


class CompiledFusionPlanner:
    def __init__(
        self,
        *,
        l2_cache_bytes: int,
        sm_count: int,
        precision_bytes: int = 2,
        boundary_predictor: FusionBoundaryPredictor | None = None,
    ):
        self.l2_cache_bytes = int(l2_cache_bytes)
        self.sm_count = int(sm_count)
        self.precision_bytes = int(precision_bytes)
        self.boundary_predictor = boundary_predictor

    def plan(
        self,
        gm: torch.fx.GraphModule,
        lifted_constants: frozenset[str] = frozenset(),
    ) -> FusionPlan:
        modules = dict(gm.named_modules())
        nodes = list(gm.graph.nodes)
        positions = {node: index for index, node in enumerate(nodes)}
        classes = {node: classify_node(node, modules) for node in nodes}
        # [Generate-only rule] lift_fresh_copy of a constant that the
        # ExportedProgram signature lists as a lifted tensor constant is a
        # static constant -> metadata. Anything else stays unsupported.
        for node in nodes:
            if node.op == "call_function" and canonical_name(node, modules) == "lift_fresh_copy":
                source = node.args[0] if node.args else None
                if (
                    isinstance(source, torch.fx.Node)
                    and source.op == "placeholder"
                    and source.name in lifted_constants
                ):
                    classes[node] = "metadata"
        light = [
            node
            for node in nodes
            if classes[node] in {"pointwise", "reduction", "memory"}
        ]
        effective_inputs = {
            node: self._effective_inputs(node, classes) for node in nodes
        }
        effective_users: dict[torch.fx.Node, set[torch.fx.Node]] = defaultdict(set)
        for consumer, producers in effective_inputs.items():
            for producer in producers:
                effective_users[producer].add(consumer)

        dsu = _DisjointSet(light)
        reductions = {node: int(classes[node] == "reduction") for node in light}
        for consumer in light:
            for producer in effective_inputs[consumer]:
                if producer not in dsu.parent:
                    continue
                if not self._edge_is_fusible(
                    producer,
                    consumer,
                    classes,
                    effective_users,
                    effective_inputs,
                    positions,
                    modules,
                ):
                    continue
                left = dsu.find(producer)
                right = dsu.find(consumer)
                if left is right:
                    continue
                reduction_count = reductions[left] + reductions[right]
                if reduction_count > 1:
                    continue
                dsu.union(left, right)
                root = dsu.find(left)
                reductions[root] = reduction_count

        members: dict[torch.fx.Node, list[torch.fx.Node]] = defaultdict(list)
        for node in light:
            members[dsu.find(node)].append(node)

        groups = []
        for group_nodes in sorted(
            members.values(), key=lambda values: min(positions[node] for node in values)
        ):
            groups.append(
                self._build_group(
                    group_nodes,
                    classes,
                    effective_inputs,
                    effective_users,
                    modules,
                    positions,
                )
            )

        unsupported = tuple(
            {
                "name": node.name,
                "op": node.op,
                "target": target_name(node, modules),
            }
            for node in nodes
            if classes[node] == "unsupported"
        )
        return FusionPlan(
            groups=tuple(groups),
            unsupported=unsupported,
            heavy_node_count=sum(value == "heavy" for value in classes.values()),
            embedding_node_count=sum(
                value == "embedding" for value in classes.values()
            ),
        )

    def _effective_inputs(
        self,
        node: torch.fx.Node,
        classes: dict[torch.fx.Node, str],
    ) -> tuple[torch.fx.Node, ...]:
        result: list[torch.fx.Node] = []
        seen: set[torch.fx.Node] = set()

        def visit(candidate: torch.fx.Node) -> None:
            if candidate in seen:
                return
            seen.add(candidate)
            if classes.get(candidate) == "metadata":
                for parent in collect_nodes((candidate.args, candidate.kwargs)):
                    visit(parent)
            else:
                result.append(candidate)

        for source in collect_nodes((node.args, node.kwargs)):
            visit(source)
        return tuple(dict.fromkeys(result))

    def _edge_is_fusible(
        self,
        producer: torch.fx.Node,
        consumer: torch.fx.Node,
        classes: dict[torch.fx.Node, str],
        effective_users: dict[torch.fx.Node, set[torch.fx.Node]],
        effective_inputs: dict[torch.fx.Node, tuple[torch.fx.Node, ...]],
        positions: dict[torch.fx.Node, int],
        modules: dict[str, nn.Module],
    ) -> bool:
        if is_fusion_barrier(producer, modules) or is_fusion_barrier(
            consumer, modules
        ):
            return False
        materialized_users = {
            user
            for user in effective_users.get(producer, set())
            if classes.get(user) not in {"metadata", "output"}
        }
        if materialized_users != {consumer}:
            return False
        producer_shape = tensor_shape(producer)
        consumer_shape = tensor_shape(consumer)
        if producer_shape is None or consumer_shape is None:
            return False
        structurally_fusible = False
        if "memory" in {classes[producer], classes[consumer]}:
            if tensor_elements(producer) == tensor_elements(consumer):
                structurally_fusible = True
        if classes[consumer] in {"pointwise", "memory"}:
            structurally_fusible = structurally_fusible or self._broadcast_compatible(
                producer_shape, consumer_shape
            )
        if classes[consumer] == "reduction":
            structurally_fusible = True
        if not structurally_fusible:
            return False
        if self.boundary_predictor is None:
            return True
        features = self._edge_features(
            producer,
            consumer,
            classes,
            effective_inputs,
            effective_users,
            positions,
            modules,
        )
        return self.boundary_predictor.should_fuse(features)

    def candidate_edges(self, gm: torch.fx.GraphModule) -> tuple[dict[str, Any], ...]:
        modules = dict(gm.named_modules())
        nodes = list(gm.graph.nodes)
        positions = {node: index for index, node in enumerate(nodes)}
        classes = {node: classify_node(node, modules) for node in nodes}
        effective_inputs = {
            node: self._effective_inputs(node, classes) for node in nodes
        }
        effective_users: dict[torch.fx.Node, set[torch.fx.Node]] = defaultdict(set)
        for consumer, producers in effective_inputs.items():
            for producer in producers:
                effective_users[producer].add(consumer)
        rows = []
        for consumer in nodes:
            if classes[consumer] not in {"pointwise", "reduction", "memory"}:
                continue
            for producer in effective_inputs[consumer]:
                if classes.get(producer) not in {"pointwise", "reduction", "memory"}:
                    continue
                if is_fusion_barrier(producer, modules) or is_fusion_barrier(
                    consumer, modules
                ):
                    continue
                materialized_users = {
                    user
                    for user in effective_users.get(producer, set())
                    if classes.get(user) not in {"metadata", "output"}
                }
                if materialized_users != {consumer}:
                    continue
                producer_shape = tensor_shape(producer)
                consumer_shape = tensor_shape(consumer)
                if producer_shape is None or consumer_shape is None:
                    continue
                features = self._edge_features(
                    producer,
                    consumer,
                    classes,
                    effective_inputs,
                    effective_users,
                    positions,
                    modules,
                )
                rows.append(
                    {
                        "producer": producer.name,
                        "consumer": consumer.name,
                        "producer_operation": canonical_name(producer, modules),
                        "consumer_operation": canonical_name(consumer, modules),
                        "features": features,
                    }
                )
        return tuple(rows)

    def _edge_features(
        self,
        producer: torch.fx.Node,
        consumer: torch.fx.Node,
        classes: dict[torch.fx.Node, str],
        effective_inputs: dict[torch.fx.Node, tuple[torch.fx.Node, ...]],
        effective_users: dict[torch.fx.Node, set[torch.fx.Node]],
        positions: dict[torch.fx.Node, int],
        modules: dict[str, nn.Module],
    ) -> dict[str, float]:
        producer_shape = tensor_shape(producer) or ()
        consumer_shape = tensor_shape(consumer) or ()
        producer_elements = tensor_elements(producer)
        consumer_elements = tensor_elements(consumer)
        producer_name = canonical_name(producer, modules)
        consumer_name = canonical_name(consumer, modules)
        input_classes = {
            classes.get(node)
            for node in effective_inputs.get(producer, ())
        }
        output_classes = {
            classes.get(node)
            for node in effective_users.get(consumer, set())
        }
        values = {
            "Producer_Elements": float(producer_elements),
            "Consumer_Elements": float(consumer_elements),
            "Element_Ratio": float(
                max(producer_elements, consumer_elements)
                / max(min(producer_elements, consumer_elements), 1)
            ),
            "Producer_Bytes": float(tensor_bytes(producer)),
            "Consumer_Bytes": float(tensor_bytes(consumer)),
            "Same_Shape": float(producer_shape == consumer_shape),
            "Same_Numel": float(producer_elements == consumer_elements),
            "Rank_Delta": float(abs(len(producer_shape) - len(consumer_shape))),
            "Producer_User_Count": float(len(effective_users.get(producer, set()))),
            "Consumer_Input_Count": float(len(effective_inputs.get(consumer, ()))),
            "Graph_Distance": float(max(positions[consumer] - positions[producer], 0)),
            "Producer_Is_Pointwise": float(classes[producer] == "pointwise"),
            "Producer_Is_Reduction": float(classes[producer] == "reduction"),
            "Producer_Is_Memory": float(classes[producer] == "memory"),
            "Consumer_Is_Pointwise": float(classes[consumer] == "pointwise"),
            "Consumer_Is_Reduction": float(classes[consumer] == "reduction"),
            "Consumer_Is_Memory": float(classes[consumer] == "memory"),
            "Both_Reduction": float(
                classes[producer] == classes[consumer] == "reduction"
            ),
            "Has_Broadcast": float(producer_shape != consumer_shape),
            "Producer_Is_Binary": float(producer_name in BINARY_NAMES),
            "Consumer_Is_Binary": float(consumer_name in BINARY_NAMES),
            "Producer_Is_Transcendental": float(
                producer_name in TRANSCENDENTAL_NAMES
            ),
            "Consumer_Is_Transcendental": float(
                consumer_name in TRANSCENDENTAL_NAMES
            ),
            "Producer_Has_Heavy_Input": float("heavy" in input_classes),
            "Consumer_Has_Heavy_Output": float("heavy" in output_classes),
            "Fits_L2": float(
                tensor_bytes(producer) + tensor_bytes(consumer)
                <= self.l2_cache_bytes
            ),
            "Precision_Bytes": float(self.precision_bytes),
        }
        return {name: values[name] for name in FUSION_EDGE_FEATURE_NAMES}

    @staticmethod
    def _broadcast_compatible(
        producer_shape: tuple[int, ...],
        consumer_shape: tuple[int, ...],
    ) -> bool:
        for left, right in zip(reversed(producer_shape), reversed(consumer_shape)):
            if left not in {1, right}:
                return False
        return len(producer_shape) <= len(consumer_shape) or producer_shape == consumer_shape

    def _build_group(
        self,
        group_nodes: list[torch.fx.Node],
        classes: dict[torch.fx.Node, str],
        effective_inputs: dict[torch.fx.Node, tuple[torch.fx.Node, ...]],
        effective_users: dict[torch.fx.Node, set[torch.fx.Node]],
        modules: dict[str, nn.Module],
        positions: dict[torch.fx.Node, int],
    ) -> dict[str, Any]:
        group_nodes = sorted(group_nodes, key=positions.__getitem__)
        group_set = set(group_nodes)
        external_inputs = {
            source
            for node in group_nodes
            for source in effective_inputs[node]
            if source not in group_set and tensor_meta(source) is not None
        }
        boundary_outputs = {
            node
            for node in group_nodes
            if not effective_users.get(node)
            or any(user not in group_set for user in effective_users[node])
        }
        if not boundary_outputs:
            boundary_outputs = {group_nodes[-1]}

        names = [canonical_name(node, modules) for node in group_nodes]
        pointwise_count = sum(classes[node] == "pointwise" for node in group_nodes)
        reduction_count = sum(classes[node] == "reduction" for node in group_nodes)
        memory_count = sum(classes[node] == "memory" for node in group_nodes)
        input_bytes = sum(tensor_bytes(node) for node in external_inputs)
        output_bytes = sum(tensor_bytes(node) for node in boundary_outputs)
        output_elements = sum(tensor_elements(node) for node in boundary_outputs)
        max_node_elements = max(tensor_elements(node) for node in group_nodes)
        estimated_flops = sum(
            tensor_elements(node) * OP_COST[canonical_name(node, modules)]
            for node in group_nodes
        )
        reduction_size = 1
        for node in group_nodes:
            if classes[node] != "reduction":
                continue
            input_elements = max(
                (tensor_elements(source) for source in effective_inputs[node]),
                default=tensor_elements(node),
            )
            reduction_size = max(
                reduction_size,
                input_elements // max(tensor_elements(node), 1),
            )
        broadcast_count = 0
        for node in group_nodes:
            output = tensor_elements(node)
            sources = [
                tensor_elements(source)
                for source in effective_inputs[node]
                if tensor_elements(source) > 0
            ]
            if any(value not in {1, output} for value in sources):
                broadcast_count += 1
            elif len(set(sources)) > 1:
                broadcast_count += 1

        output_shape = tensor_shape(max(boundary_outputs, key=positions.__getitem__)) or ()
        predecessor_classes = {
            classes[source]
            for node in group_nodes
            for source in effective_inputs[node]
            if source not in group_set
        }
        successor_classes = {
            classes[user]
            for node in group_nodes
            for user in effective_users.get(node, set())
            if user not in group_set
        }
        if reduction_count:
            group_kind = "reduction"
        elif pointwise_count:
            group_kind = "pointwise"
        else:
            group_kind = "memory"

        ideal_bytes = input_bytes + output_bytes
        features = {
            "Input_Bytes": float(input_bytes),
            "Output_Bytes": float(output_bytes),
            "Ideal_Memory_Bytes": float(ideal_bytes),
            "Estimated_FLOPs": float(estimated_flops),
            "Arithmetic_Intensity": float(estimated_flops / max(ideal_bytes, 1)),
            "Output_Elements": float(output_elements),
            "Max_Node_Elements": float(max_node_elements),
            "Reduction_Size": float(reduction_size),
            "Op_Count": float(len(group_nodes)),
            "Pointwise_Count": float(pointwise_count),
            "Reduction_Count": float(reduction_count),
            "Memory_Count": float(memory_count),
            "Binary_Count": float(sum(name in BINARY_NAMES for name in names)),
            "Broadcast_Count": float(broadcast_count),
            "Transcendental_Count": float(
                sum(name in TRANSCENDENTAL_NAMES for name in names)
            ),
            "Input_Tensor_Count": float(len(external_inputs)),
            "Output_Tensor_Count": float(len(boundary_outputs)),
            "Rank": float(len(output_shape)),
            "Last_Dim": float(output_shape[-1] if output_shape else 1),
            "Estimated_L2_Fit": float(ideal_bytes <= self.l2_cache_bytes),
            "Estimated_Wave_Count": float(
                math.ceil(max(output_elements, 1) / 1024.0) / self.sm_count
            ),
            "Has_Heavy_Predecessor": float("heavy" in predecessor_classes),
            "Has_Heavy_Successor": float("heavy" in successor_classes),
            "Is_Pointwise_Group": float(group_kind == "pointwise"),
            "Is_Reduction_Group": float(group_kind == "reduction"),
            "Is_Memory_Group": float(group_kind == "memory"),
            "Precision_Bytes": float(self.precision_bytes),
        }
        return {
            "type": "CompiledFusionGroup",
            "kind": group_kind,
            "node_names": [node.name for node in group_nodes],
            "operations": names,
            "features": {name: features[name] for name in FUSION_FEATURE_NAMES},
        }


def theoretical_time_ms(
    features: dict[str, float],
    *,
    peak_flops: float,
    peak_bandwidth: float,
    kernel_floor_ms: float = 1e-6,
) -> float:
    compute_ms = features["Estimated_FLOPs"] / peak_flops * 1000.0
    memory_ms = features["Ideal_Memory_Bytes"] / peak_bandwidth * 1000.0
    return max(compute_ms, memory_ms, kernel_floor_ms)
