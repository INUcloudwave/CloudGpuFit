"""Offline regression checks; no Docker, torch, model download or GPU required."""
import argparse
import ast
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import warnings

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = module_at("setup_manager", "orchestrator/setup_manager.py")
cache = module_at("cache_key", "orchestrator/cache_key.py")


class PredictionRegressionTests(unittest.TestCase):
    def test_reserved_is_used_before_outside_not_allocated(self):
        from vram_policy import memory_result
        value = memory_result(1000, 4096, "fp16", "NVIDIA L4")
        target = value["target_gpu_estimate"]
        self.assertEqual(value["required_vram_bytes"], 4096 + target["outside_observed_upper_bytes"])
        self.assertEqual(target["estimated_nvml_bytes"], 4096 + target["outside_estimate_bytes"])
        self.assertEqual(value["breakdown_bytes"]["fixed_margin"], 0)
        self.assertEqual(value["breakdown_bytes"]["empirical_tensor_correction"], 0)

    def test_dtype_does_not_rescale_native_trace(self):
        from vram_policy import memory_result
        for dtype in ["fp16", "fp32", "bf16"]:
            value = memory_result(12345, 20000, dtype)
            self.assertEqual(value["predicted_allocated_bytes"], 12345)
            self.assertEqual(value["predicted_reserved_bytes"], 20000)

    def test_unknown_gpu_is_not_assigned_zero_overhead(self):
        from vram_policy import memory_result
        with self.assertRaisesRegex(ValueError, "calibration"):
            memory_result(1000, 4096, "fp16", "NVIDIA L40S")

    def test_t4_bf16_is_not_a_supported_target(self):
        from vram_policy import memory_result
        value = memory_result(1000, 4096, "bf16")
        self.assertEqual(value["gpu_requirements"]["NVIDIA T4"]["status"], "unsupported_dtype")

    def test_invalid_trace_rejected(self):
        from vram_policy import memory_result
        for a,r in [(200,100),(-1,100),(float("nan"),100)]:
            with self.assertRaises(ValueError):memory_result(a,r,"fp32")


class DeploymentRegressionTests(unittest.TestCase):
    def test_ready_container_refreshes_stale_predictor_without_reinstall(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            repo, work = base / "repo", base / "work"
            repo.mkdir()
            (work / "xMem").mkdir(parents=True)
            (repo / "predict_vram.py").write_bytes(b"new-predictor")
            for name in ["cpu_tensor_trace.py", "vram_policy.py", "gpu_overheads.json"]:
                (repo / name).write_bytes(b"new-helper")
            target = work / "xMem" / "predict_vram.py"
            target.write_bytes(b"old-predictor")
            preserved = work / "xMem" / "keep-result.json"
            preserved.write_bytes(b"existing-result")
            with patch.object(setup, "CLOUDGPUFIT_DIR", repo), \
                 patch.object(setup, "WORK_DIR", work), \
                 patch.object(setup, "container_state", return_value="running"), \
                 patch.object(setup, "_pip_ready_in_container", return_value=True), \
                 patch.object(setup, "_finish_xmem_setup") as reinstall:
                self.assertEqual(setup.ensure_ready(), {"status": "already_running"})
                self.assertEqual(target.read_bytes(), b"new-predictor")
                self.assertEqual(preserved.read_bytes(), b"existing-result")
                reinstall.assert_not_called()
                self.assertEqual(list(target.parent.glob(".predict-vram-*")), [])

    def test_identical_predictor_is_left_in_place(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            repo, work = base / "repo", base / "work"
            repo.mkdir()
            (work / "xMem").mkdir(parents=True)
            (repo / "predict_vram.py").write_bytes(b"same-predictor")
            for name in ["cpu_tensor_trace.py", "vram_policy.py", "gpu_overheads.json"]:
                (repo / name).write_bytes(b"same-helper")
                (work / "xMem" / name).write_bytes(b"same-helper")
            target = work / "xMem" / "predict_vram.py"
            target.write_bytes(b"same-predictor")
            with patch.object(setup, "CLOUDGPUFIT_DIR", repo), \
                 patch.object(setup, "WORK_DIR", work), \
                 patch.object(setup.os, "replace") as replace:
                setup._sync_predictor()
                replace.assert_not_called()

    def test_old_margin_results_use_a_different_pipeline_cache_key(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            args = {"model": "openai-community/gpt2", "dtype": "fp16", "batch_size": 1}
            kwargs = {"eager_runtime_root": Path(directory), "catalog_version": "same"}
            current = cache.pipeline_cache_key_for(args, {}, **kwargs)
            with patch.object(cache, "VRAM_PREDICTOR_VERSION", "1"):
                old = cache.pipeline_cache_key_for(args, {}, **kwargs)
            self.assertNotEqual(current, old)


if __name__ == "__main__":
    unittest.main()
