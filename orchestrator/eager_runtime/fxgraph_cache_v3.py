"""Cache-aware eager FX latency simulator.

This file extends ``fxgraph_reduction_v2.py`` without modifying it.  Keep both
files in the same directory and put the cache-aware GEMM models under
``cache_model/``.
"""

import os
from pathlib import Path

import lightgbm as lgb
import math
import numpy as np
import pandas as pd

import fxgraph_reduction_v2 as base


PRECISION_BYTES = 2
L2_CACHE_BYTES = 48 * 1024 * 1024


def _gemm_ideal_bytes(node):
    """Return the same ideal-byte definition used by GEMM data collection."""
    m, n, k = node["M"], node["N"], node["K"]
    if node.get("Is_BMM", 0):
        batch_dim = node.get("B_dim", 1)
        true_m = node.get("M_true", m)
        return batch_dim * (
            true_m * k + k * n + true_m * n
        ) * PRECISION_BYTES
    return (m * k + k * n + m * n) * PRECISION_BYTES


def _iter_base_nodes(fused_graph):
    for group in fused_graph:
        if group.get("type") == "FusionGroup":
            yield group["base_op"]


class CacheAwareFXGraphParser(base.FXGraphParser):
    """Add model-execution cache context to parsed Linear nodes."""

    def extract_nodes(self, gm, dummy_input, engine_mode="eager"):
        fused_graph = super().extract_nodes(gm, dummy_input, engine_mode)
        base_nodes = list(_iter_base_nodes(fused_graph))

        # One weight is normally revisited only on the next inference.  The
        # GEMM traffic of one whole graph execution is therefore a practical
        # reuse-distance proxy.  LightGBM uses it mainly to distinguish a
        # resident/hot weight from a model-context/cold weight.
        graph_working_set = sum(
            _gemm_ideal_bytes(node)
            for node in base_nodes
            if node.get("type") in base.HEAVY_TYPES
        )

        for node in base_nodes:
            if node.get("type") not in base.HEAVY_TYPES:
                continue

            if node.get("type") != "Linear":
                # Cache supplements were collected for Linear only.  These
                # defaults preserve the original Conv/BMM interpretation.
                node["Weight_Bytes"] = (
                    node.get("K", 0)
                    * node.get("N", 0)
                    * PRECISION_BYTES
                    if not node.get("Is_BMM", 0)
                    else 0
                )
                node["RHS_Is_Transposed"] = 0
                node["Estimated_Weight_L2_Resident"] = 1
                node["Reuse_Distance_Bytes"] = 0
                node["Context_Working_Set_Bytes"] = _gemm_ideal_bytes(node)
                continue

            weight_bytes = node["K"] * node["N"] * PRECISION_BYTES
            reuse_distance = max(0, graph_working_set - weight_bytes)

            node["Weight_Bytes"] = weight_bytes
            node["RHS_Is_Transposed"] = 1
            node["Reuse_Distance_Bytes"] = reuse_distance
            node["Context_Working_Set_Bytes"] = max(
                graph_working_set,
                _gemm_ideal_bytes(node),
            )
            node["Estimated_Weight_L2_Resident"] = int(
                weight_bytes + reuse_distance <= L2_CACHE_BYTES
            )

        return fused_graph


class CacheAwareSimulatorRouter(base.SimulatorRouter):
    """Use cache-aware GEMM PKLs while retaining Reduction V2 behavior."""

    def __init__(self, engine_mode="eager"):
        if engine_mode != "eager":
            raise ValueError(
                "fxgraph_cache_v3.py is calibrated for eager mode only. "
                "Treat compiled prediction as a separate model."
            )

        print(
            f"🚀 CloudSim Router 초기화 중 "
            f"(모드: {engine_mode.upper()}, CACHE-AWARE)..."
        )
        self.engine_mode = engine_mode
        model_dir = Path(os.environ.get("CACHE_MODEL_DIR", "cache_model"))
        base_model_dir = Path(os.environ.get("EAGER_BASE_MODEL_DIR", "."))
        self.gemm_median = lgb.Booster(
            model_file=str(model_dir / "gemm_cache_eager_median.pkl")
        )
        self.gemm_90th = lgb.Booster(
            model_file=str(model_dir / "gemm_cache_eager_90th.pkl")
        )
        self.red_median = lgb.Booster(
            model_file=str(base_model_dir / "reduction_eager_median.pkl")
        )
        self.red_90th = lgb.Booster(
            model_file=str(base_model_dir / "reduction_eager_90th.pkl")
        )

        base.apply_hardware_profile(self)

        required = {
            "Weight_Bytes",
            "RHS_Is_Transposed",
            "Estimated_Weight_L2_Resident",
            "Reuse_Distance_Bytes",
            "Context_Working_Set_Bytes",
        }
        missing = required.difference(self.gemm_median.feature_name())
        if missing:
            raise RuntimeError(
                "The loaded GEMM median model is not cache-aware; missing "
                + ", ".join(sorted(missing))
            )

    def _predict_node(
        self,
        node,
        batch_size,
        prediction_mode,
        force_memory_zero=False,
    ):
        op_type = node.get("type")
        if op_type not in base.HEAVY_TYPES:
            return super()._predict_node(
                node,
                batch_size,
                prediction_mode,
                force_memory_zero,
            )

        m, n, k = node["M"], node["N"], node["K"]
        is_bmm = node.get("Is_BMM", 0)
        is_fused = node.get("Is_Fused", 0)
        total_flops = 2 * m * n * k
        if is_bmm and is_fused:
            total_flops *= 2

        ideal_memory_bytes = 0 if force_memory_zero else _gemm_ideal_bytes(node)
        arithmetic_intensity = (
            total_flops / ideal_memory_bytes if ideal_memory_bytes > 0 else 0
        )
        theoretical_time_ms = max(
            total_flops / self.peak_flops * 1000,
            ideal_memory_bytes / self.peak_bandwidth * 1000,
            1e-6,
        )

        row = {
            "Batch_Size": batch_size,
            "Dim_M": m,
            "Dim_N": n,
            "Dim_K": k,
            "Total_FLOPs": total_flops,
            "Ideal_Memory_Bytes": ideal_memory_bytes,
            "Arithmetic_Intensity": arithmetic_intensity,
            "Estimated_L2_Fit": int(ideal_memory_bytes <= self.l2_cache_bytes),
            "Orig_Kernel_Size": node.get("kernel_size", 1),
            "Precision_Bytes": self.precision_bytes,
            "Is_Conv1D": node.get("Is_Conv1D", 0),
            "Is_Conv2D": node.get("Is_Conv2D", 0),
            "Is_BMM": is_bmm,
            "Is_Fused": is_fused,
            "Estimated_Wave_Count": (
                math.ceil(m / 128.0)
                * math.ceil(n / 128.0)
                / self.sm_count
            ),
            "Weight_Bytes": node.get("Weight_Bytes", 0),
            "RHS_Is_Transposed": node.get("RHS_Is_Transposed", 0),
            "Estimated_Weight_L2_Resident": node.get(
                "Estimated_Weight_L2_Resident", 1
            ),
            "Reuse_Distance_Bytes": node.get("Reuse_Distance_Bytes", 0),
            "Context_Working_Set_Bytes": node.get(
                "Context_Working_Set_Bytes", ideal_memory_bytes
            ),
        }

        booster = (
            self.gemm_median
            if prediction_mode == "median"
            else self.gemm_90th
        )
        features = pd.DataFrame([row]).reindex(columns=booster.feature_name())
        ratio_log = booster.predict(features)[0]
        predicted_ms = theoretical_time_ms * np.expm1(ratio_log)
        return predicted_ms, theoretical_time_ms


def main():
    # run_all_simulations resolves these names from the imported module.
    base.FXGraphParser = CacheAwareFXGraphParser
    base.SimulatorRouter = CacheAwareSimulatorRouter
    base.run_all_simulations(target_mode="eager")


if __name__ == "__main__":
    main()
