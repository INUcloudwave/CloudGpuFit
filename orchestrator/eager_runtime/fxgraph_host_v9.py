#!/usr/bin/env python3
"""Eager v9 simulator with graph-conditional exposed host scheduling."""

from __future__ import annotations

import os

import fxgraph_fidelity_v8 as v8
import fxgraph_pointwise_v7 as v7
import fxgraph_reduction_v2 as base


class HostScheduleSimulatorRouter(v8.FidelitySimulatorRouter):
    def __init__(self, engine_mode="eager"):
        super().__init__(engine_mode=engine_mode)
        self.host_schedule_scale = float(
            os.environ.get(
                "EAGER_HOST_SCHEDULE_SCALE",
                str(self.host_schedule_scale),
            )
        )
        self.host_schedule_policy = os.environ.get(
            "EAGER_HOST_SCHEDULE_POLICY",
            self.host_schedule_policy,
        )

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
                if item.get("type") == "WorkloadMetadata"
            ),
            {},
        )
        total_ms, log_df = super().simulate_fx_graph(
            fused_graph,
            batch_size,
            prediction_mode,
        )
        log_df.attrs["workload_family"] = metadata.get("family", "unknown")
        log_df.attrs["input_batch_size"] = int(
            metadata.get("batch_size", batch_size)
        )
        return total_ms, log_df

    def simulate_fx_graph_batch(
        self,
        fused_graph,
        batch_size,
        prediction_modes=("median", "90th"),
    ):
        """Batch-API twin of simulate_fx_graph (same attrs)."""
        metadata = next(
            (
                item
                for item in fused_graph
                if item.get("type") == "WorkloadMetadata"
            ),
            {},
        )
        results = super().simulate_fx_graph_batch(
            fused_graph,
            batch_size,
            prediction_modes,
        )
        for _total_ms, log_df in results.values():
            log_df.attrs["workload_family"] = metadata.get("family", "unknown")
            log_df.attrs["input_batch_size"] = int(
                metadata.get("batch_size", batch_size)
            )
        return results

    def _schedule_scale(self, log_df):
        if self.engine_mode != "eager":
            return 1.0
        if self.host_schedule_policy == "none":
            return 1.0
        if self.host_schedule_policy == "global":
            return self.host_schedule_scale
        if "Layer_Type" not in log_df.columns:
            return 1.0
        has_conv2d = log_df["Layer_Type"].eq("Conv2d").any()
        return self.host_schedule_scale if has_conv2d else 1.0

    def simulate_e2e_pipeline(self, log_df, input_bytes, output_bytes):
        scale = self._schedule_scale(log_df)
        log_df.attrs["host_schedule_scale"] = scale
        log_df.attrs["host_schedule_policy"] = self.host_schedule_policy
        log_df.attrs["host_schedule_applied"] = int(scale != 1.0)

        if scale == 1.0:
            total_ms, timeline = super().simulate_e2e_pipeline(
                log_df, input_bytes, output_bytes
            )
        else:
            original = (
                self.t_dispatch_light,
                self.t_dispatch_heavy,
                self.t_python_loop,
            )
            try:
                self.t_dispatch_light *= scale
                self.t_dispatch_heavy *= scale
                self.t_python_loop *= scale
                total_ms, timeline = super().simulate_e2e_pipeline(
                    log_df, input_bytes, output_bytes
                )
            finally:
                (
                    self.t_dispatch_light,
                    self.t_dispatch_heavy,
                    self.t_python_loop,
                ) = original

        kernel_sum_ms = float(log_df["Predicted_Actual_ms"].sum())
        workload_family = log_df.attrs.get("workload_family", "unknown")
        input_batch_size = int(log_df.attrs.get("input_batch_size", 1))
        additive_ms = float(
            getattr(self, "text_small_workload_additive_ms", 0.0)
        )
        threshold_ms = float(
            getattr(self, "text_small_workload_kernel_threshold_ms", 0.0)
        )
        max_batch = int(getattr(self, "text_small_workload_max_batch", 1))
        apply_text_correction = (
            workload_family == "text"
            and input_batch_size <= max_batch
            and threshold_ms > 0.0
            and kernel_sum_ms < threshold_ms
            and additive_ms > 0.0
        )
        correction_ms = additive_ms if apply_text_correction else 0.0
        log_df.attrs["text_small_workload_correction_ms"] = correction_ms
        log_df.attrs["workload_family"] = workload_family
        return total_ms + correction_ms, timeline


def build_prediction_breakdown(
    router,
    model_name,
    batch_size,
    mode,
    log_df,
    input_bytes,
    output_bytes,
    predicted_e2e_ms,
):
    record = v8.build_prediction_breakdown(
        router=router,
        model_name=model_name,
        batch_size=batch_size,
        mode=mode,
        log_df=log_df,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        predicted_e2e_ms=predicted_e2e_ms,
    )
    scale = float(log_df.attrs.get("host_schedule_scale", 1.0))
    raw_unscaled = record["host_enqueue_raw_ms"]
    raw_scaled = router.t_pytorch_overhead + (
        raw_unscaled - router.t_pytorch_overhead
    ) * scale
    record["host_enqueue_raw_unscaled_ms"] = raw_unscaled
    record["host_enqueue_raw_ms"] = raw_scaled
    record["host_schedule_scale"] = scale
    record["host_schedule_policy"] = log_df.attrs.get(
        "host_schedule_policy", "none"
    )
    record["host_schedule_applied"] = int(
        log_df.attrs.get("host_schedule_applied", 0)
    )
    record["workload_family"] = log_df.attrs.get("workload_family", "unknown")
    record["text_small_workload_correction_ms"] = float(
        log_df.attrs.get("text_small_workload_correction_ms", 0.0)
    )
    return record


def main():
    base.FXGraphParser = v8.FidelityFXGraphParser
    base.SimulatorRouter = HostScheduleSimulatorRouter
    base.prediction_family = v7.prediction_family
    base.build_prediction_breakdown = build_prediction_breakdown
    print("Eager Conv2d host-schedule correction active")
    base.run_all_simulations(target_mode="eager")


if __name__ == "__main__":
    main()
