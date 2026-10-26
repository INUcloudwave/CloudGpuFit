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


def predictor():
    # Execute the real calculation path with the CPU profiler's observed output.
    tree = ast.parse((ROOT / "predict_vram.py").read_text(encoding="utf-8"))
    constants = {"DTYPE_SCALE", "DTYPE_BYTES", "UNDERESTIMATE_CONSTANT",
                 "WORKLOAD_MODEL_CLASS"}
    functions = {"run", "collect_warnings", "kv_cache_bytes"}
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in functions
             or isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)]
    cfg = types.SimpleNamespace(num_hidden_layers=12, num_attention_heads=12,
                                hidden_size=768, vocab_size=50257)
    env = {"time": time, "warnings": warnings, "os": os,
           "torch": types.SimpleNamespace(manual_seed=lambda seed: None),
           "load_model": lambda *args: (object(), "llm", cfg),
           "param_bytes": lambda model: 510342192,
           "build_loader": lambda *args, **kwargs: object(),
           "profile_on_cpu": lambda *args, **kwargs: "observed-profile",
           "estimate": lambda *args: {"memory": {"tensor": 564365312}}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "predict_vram.py", "exec"), env)
    return env["run"]


def request(dtype="fp16", mode="generate"):
    return argparse.Namespace(model="openai-community/gpt2", workload_type="decoder_llm",
                              model_config=None, batch_size=1, seq_len=256,
                              encoder_seq_len=256, decoder_seq_len=1, resolution=None,
                              dtype=dtype, mode=mode, max_new_tokens=64, gpu_memory_gb=24)


class PredictionRegressionTests(unittest.TestCase):
    def evaluate(self, **kwargs):
        with patch.dict(os.environ, {}, clear=False), warnings.catch_warnings():
            return predictor()(request(**kwargs))

    def test_observed_gpt2_profile_has_no_hidden_two_gib_addition(self):
        value = self.evaluate()
        self.assertEqual(value["status"], "ok")
        self.assertEqual(value["breakdown_bytes"]["fixed_margin"], 0)
        self.assertEqual(value["required_vram_bytes"], 310756352)
        self.assertEqual(value["required_vram_gb"], 0.29)
        self.assertEqual(value["breakdown_bytes"]["after_underestimate_correction"], 298959872)
        self.assertTrue(any("CUDA context" in w for w in value["warnings"]))

    def test_prefill_and_generate_still_differ_by_the_requested_kv(self):
        prefill = self.evaluate(mode="prefill")
        generate = self.evaluate(mode="generate")
        self.assertEqual(prefill["required_vram_bytes"], 298959872)
        self.assertIsNone(prefill["breakdown_bytes"]["kv_cache_estimate"])
        self.assertEqual(generate["required_vram_bytes"] - prefill["required_vram_bytes"],
                         11796480)

    def test_precision_conversion_is_preserved(self):
        fp32 = self.evaluate(dtype="fp32")
        fp16 = self.evaluate(dtype="fp16")
        bf16 = self.evaluate(dtype="bf16")
        self.assertEqual(fp32["required_vram_bytes"], 2 * fp16["required_vram_bytes"])
        self.assertEqual(bf16["required_vram_bytes"], fp16["required_vram_bytes"])


class DeploymentRegressionTests(unittest.TestCase):
    def test_ready_container_refreshes_stale_predictor_without_reinstall(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            repo, work = base / "repo", base / "work"
            repo.mkdir()
            (work / "xMem").mkdir(parents=True)
            (repo / "predict_vram.py").write_bytes(b"new-predictor")
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
