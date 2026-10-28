#!/usr/bin/env python3
"""Eager v8 simulator with a structural T5 FX-fidelity correction."""

from __future__ import annotations

import math
import os
import re

import torch

import fxgraph_pointwise_v7 as v7
import fxgraph_reduction_v2 as base


_BASE_BUILD_BREAKDOWN = base.build_prediction_breakdown


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _tensor_sequence_length(tensor):
    if tensor is None or tensor.ndim < 2:
        return 1
    return int(tensor.shape[1])


def _input_metadata(dummy_input):
    first = _first_tensor(dummy_input)
    batch_size = int(first.shape[0]) if first is not None and first.ndim else 1
    if not isinstance(dummy_input, dict):
        return {
            "batch_size": batch_size,
            "encoder_sequence_length": _tensor_sequence_length(first),
            "decoder_sequence_length": 0,
        }
    return {
        "batch_size": batch_size,
        "encoder_sequence_length": _tensor_sequence_length(
            dummy_input.get("input_ids")
        ),
        "decoder_sequence_length": _tensor_sequence_length(
            dummy_input.get("decoder_input_ids")
        ) if dummy_input.get("decoder_input_ids") is not None else 0,
    }


def _block_count(module_names, stack):
    pattern = re.compile(rf"(?:^|\.){stack}\.block\.(\d+)(?:\.|$)")
    indices = {
        int(match.group(1))
        for name in module_names
        for match in [pattern.search(name)]
        if match is not None
    }
    return len(indices)


def _t5_metadata(gm, dummy_input):
    modules = dict(gm.named_modules())
    names = tuple(modules)
    encoder_layers = _block_count(names, "encoder")
    decoder_layers = _block_count(names, "decoder")
    is_t5_graph = any(
        "densereludense" in name.lower()
        or "relative_attention_bias" in name.lower()
        for name in names
    )
    if encoder_layers == 0 or decoder_layers == 0 or not is_t5_graph:
        return None

    hidden_size = 0
    for name, module in modules.items():
        is_token_embedding = name.endswith("shared") or name.endswith("embed_tokens")
        if is_token_embedding and isinstance(module, torch.nn.Embedding):
            hidden_size = int(module.embedding_dim)
            break
    if hidden_size == 0:
        hidden_size = 512

    precision_bytes = 2
    parameter = next(gm.parameters(), None)
    if parameter is not None:
        precision_bytes = int(parameter.element_size())

    input_metadata = _input_metadata(dummy_input)
    return {
        "type": "FidelityMetadata",
        "family": "t5",
        "sequence_length": input_metadata["encoder_sequence_length"],
        "encoder_sequence_length": input_metadata["encoder_sequence_length"],
        "decoder_sequence_length": input_metadata["decoder_sequence_length"],
        "hidden_size": hidden_size,
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "guard_count": 2 * encoder_layers + 3 * decoder_layers,
        "precision_bytes": precision_bytes,
    }


class FidelityFXGraphParser(v7.PointwiseContextFXGraphParser):
    def extract_nodes(self, gm, dummy_input, engine_mode="eager"):
        fused_graph = super().extract_nodes(gm, dummy_input, engine_mode)
        metadata = _t5_metadata(gm, dummy_input)
        base_nodes = [
            item["base_op"]
            for item in fused_graph
            if item.get("type") == "FusionGroup"
        ]
        layer_types = {item.get("type") for item in base_nodes}
        if metadata is not None:
            workload_family = "text_encdec"
        elif "Embedding" in layer_types and not layer_types.intersection(
            {"Conv1d", "Conv2d"}
        ):
            workload_family = "text"
        elif "Conv2d" in layer_types:
            workload_family = "vision"
        elif "Conv1d" in layer_types:
            workload_family = "audio"
        else:
            workload_family = "unknown"
        workload_metadata = {
            "type": "WorkloadMetadata",
            "family": workload_family,
            **_input_metadata(dummy_input),
        }
        if metadata is not None:
            print(
                "  T5 fidelity metadata: "
                f"guards={metadata['guard_count']}, "
                f"encoder_sequence={metadata['encoder_sequence_length']}, "
                f"decoder_sequence={metadata['decoder_sequence_length']}, "
                f"hidden={metadata['hidden_size']}"
            )
            fused_graph.append(metadata)
        fused_graph.append(workload_metadata)
        return fused_graph


class FidelitySimulatorRouter(v7.PointwiseContextSimulatorRouter):
    def simulate_fx_graph(
        self,
        fused_graph,
        batch_size,
        prediction_mode="median",
    ):
        metadata = next(
            (
                item
                for item in fused_graph
                if item.get("type") == "FidelityMetadata"
            ),
            None,
        )
        graph = [
            item
            for item in fused_graph
            if item.get("type") not in {"FidelityMetadata", "WorkloadMetadata"}
        ]
        total_ms, log_df = super().simulate_fx_graph(
            graph,
            batch_size,
            prediction_mode,
        )

        correction = {
            "fidelity_correction_ms": 0.0,
            "fidelity_guard_gpu_ms": 0.0,
            "fidelity_guard_cpu_ms": 0.0,
            "fidelity_guard_count": 0,
            "fidelity_sequence_length": 0,
            "fidelity_encoder_sequence_length": 0,
            "fidelity_decoder_sequence_length": 0,
            "fidelity_mode": "none",
        }
        if metadata is not None and metadata["family"] == "t5":
            correction = self._t5_fidelity_correction(
                metadata,
                batch_size,
                total_ms,
            )
        log_df.attrs.update(correction)
        return total_ms, log_df

    def simulate_fx_graph_batch(
        self,
        fused_graph,
        batch_size,
        prediction_modes=("median", "90th"),
    ):
        """Batch-API twin of simulate_fx_graph (same metadata handling)."""
        metadata = next(
            (
                item
                for item in fused_graph
                if item.get("type") == "FidelityMetadata"
            ),
            None,
        )
        graph = [
            item
            for item in fused_graph
            if item.get("type") not in {"FidelityMetadata", "WorkloadMetadata"}
        ]
        results = super().simulate_fx_graph_batch(graph, batch_size, prediction_modes)
        for mode, (total_ms, log_df) in results.items():
            correction = {
                "fidelity_correction_ms": 0.0,
                "fidelity_guard_gpu_ms": 0.0,
                "fidelity_guard_cpu_ms": 0.0,
                "fidelity_guard_count": 0,
                "fidelity_sequence_length": 0,
                "fidelity_encoder_sequence_length": 0,
                "fidelity_decoder_sequence_length": 0,
                "fidelity_mode": "none",
            }
            if metadata is not None and metadata["family"] == "t5":
                correction = self._t5_fidelity_correction(
                    metadata,
                    batch_size,
                    total_ms,
                )
            log_df.attrs.update(correction)
        return results

    def _t5_fidelity_correction(self, metadata, batch_size, fx_kernel_ms):
        guard_count = metadata["guard_count"]
        reference_guards = 30.0
        guard_scale = guard_count / reference_guards
        sequence_length = metadata["sequence_length"]
        encoder_sequence_length = metadata["encoder_sequence_length"]
        decoder_sequence_length = metadata["decoder_sequence_length"]
        hidden_size = metadata["hidden_size"]
        precision_bytes = metadata["precision_bytes"]

        memory_passes = float(os.environ.get("T5_GUARD_MEMORY_PASSES", "5.0"))
        cpu_floor_ms = float(os.environ.get("T5_GUARD_CPU_FLOOR_MS", "3.0"))
        batched_cpu_ms = float(os.environ.get("T5_GUARD_BATCHED_CPU_MS", "1.1"))
        overlap_threshold_ms = float(
            os.environ.get("T5_GUARD_OVERLAP_THRESHOLD_MS", "12.0")
        )

        guard_bytes = (
            guard_count
            * batch_size
            * sequence_length
            * hidden_size
            * precision_bytes
            * memory_passes
        )
        launch_floor_ms = 0.575 * guard_scale
        guard_gpu_ms = max(
            launch_floor_ms,
            guard_bytes / self.peak_bandwidth * 1000.0,
        )
        first_token_fidelity_ms = float(
            getattr(self, "t5_first_token_fidelity_ms", 0.0)
        )
        is_first_token = decoder_sequence_length == 1
        if is_first_token and first_token_fidelity_ms > 0.0:
            guard_cpu_ms = first_token_fidelity_ms * guard_scale
            fidelity_mode = "t5_first_token"
        else:
            guard_cpu_ms = cpu_floor_ms * guard_scale
            fidelity_mode = "t5_legacy"
        if (
            not is_first_token
            and
            batch_size > 1
            and fx_kernel_ms < overlap_threshold_ms * guard_scale
        ):
            guard_cpu_ms += batched_cpu_ms * guard_scale

        return {
            "fidelity_correction_ms": max(guard_gpu_ms, guard_cpu_ms),
            "fidelity_guard_gpu_ms": guard_gpu_ms,
            "fidelity_guard_cpu_ms": guard_cpu_ms,
            "fidelity_guard_count": guard_count,
            "fidelity_sequence_length": sequence_length,
            "fidelity_encoder_sequence_length": encoder_sequence_length,
            "fidelity_decoder_sequence_length": decoder_sequence_length,
            "fidelity_mode": fidelity_mode,
        }

    def simulate_e2e_pipeline(self, log_df, input_bytes, output_bytes):
        total_ms, timeline = super().simulate_e2e_pipeline(
            log_df,
            input_bytes,
            output_bytes,
        )
        return total_ms + log_df.attrs.get("fidelity_correction_ms", 0.0), timeline


def build_prediction_breakdown(*args, **kwargs):
    log_df = kwargs.get("log_df")
    if log_df is None and len(args) >= 5:
        log_df = args[4]
    record = _BASE_BUILD_BREAKDOWN(*args, **kwargs)
    for name in (
        "fidelity_correction_ms",
        "fidelity_guard_gpu_ms",
        "fidelity_guard_cpu_ms",
        "fidelity_guard_count",
        "fidelity_sequence_length",
        "fidelity_encoder_sequence_length",
        "fidelity_decoder_sequence_length",
        "fidelity_mode",
    ):
        record[name] = log_df.attrs.get(name, 0.0)
    return record


def main():
    base.FXGraphParser = FidelityFXGraphParser
    base.SimulatorRouter = FidelitySimulatorRouter
    base.prediction_family = v7.prediction_family
    base.build_prediction_breakdown = build_prediction_breakdown
    print("T5 structural FX-fidelity correction active")
    base.run_all_simulations(target_mode="eager")


if __name__ == "__main__":
    main()
