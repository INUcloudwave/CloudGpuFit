#!/usr/bin/env python3
"""Regression tests for every installed instance_type x engine_mode.

Two suites:

  smoke (default, ~1-2 min)
      For every target found under targets/ and both engine modes:
        * available_targets() lists it and the right targets/<instance>/ is used
        * runtime_manifest.json (and Compiled manifests) validate: size, sha256,
          target_instance -- i.e. no other GPU's artifacts can be picked up
        * all model files load and a first-token-scope prediction returns ok
        * every workload family (decoder_llm, text_encoder, text_encdec,
          audio_encoder, vision_cls) predicts in both modes with its own
          prediction_scope
        * Eager/Compiled branch really differs and shares one result format
        * Compiled context calibration: applied only where the target ships
          one (g7, g7e); never for gr6.4xlarge (rejected calibration)
        * G6 results are unchanged versus the pre-merge service
          (PRE_MERGE_G6_REFERENCE, recorded from the previous service code)
        * predictions match the stored snapshot (regression/baselines/*/
          smoke_snapshot.json) -- guards against unintended drift

  full (--full, tens of minutes)
      Re-runs each checkpoint's own final E2E validation (78 measured cases per
      target) with the integrated service and compares overall MAPE, bias and
      row counts with the official summaries in regression/baselines/.

Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUNTIME_ROOT = HERE.parent
PACKAGE_PARENT = RUNTIME_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

ENGINE_MODES = ("eager", "compiled")

# Every artifact was calibrated with transformers 4.45.1 (see the checkpoints'
# requirements.txt and CloudGPUFit orchestrator/requirements-runtime.txt).
# The installed transformers version changes the traced FX graph (e.g. OPT
# attention as SDPA vs matmul+softmax) and therefore the predictions, so the
# numeric checks only mean something in that environment.
REQUIRED_TRANSFORMERS = "4.45.1"

# The previous service (device code before this integration) produced these
# for g6.xlarge with the SMOKE_REQUEST below, under transformers 4.45.1.
# G6 must not regress.
PRE_MERGE_G6_REFERENCE = {
    "eager": {"predicted_median_ms": 7.681439964195674, "predicted_p90_ms": 9.579524605852791},
    "compiled": {"predicted_median_ms": 2.9200617851739787, "predicted_p90_ms": 2.9492624030257186},
}

# Instances whose Compiled context calibration was rejected during
# validation: the file must not exist and must never be applied.
REJECTED_CONTEXT_CALIBRATION = {"gr6.4xlarge"}

SMOKE_REQUEST = {
    "model": "facebook/opt-125m",
    "type": "decoder_llm",
    "dtype": "fp16",
    "batch_size": 4,
    "params": {"mode": "prefill", "seq_len": 128},
    "model_config": {
        "model_type": "opt",
        "architectures": ["OPTForCausalLM"],
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "ffn_dim": 3072,
        "vocab_size": 50272,
        "max_position_embeddings": 2048,
        "word_embed_proj_dim": 768,
    },
}

WORKLOAD_SCOPES = {
    "decoder_llm": "first_token_logits",
    "text_encdec": "first_token_logits",
    "vision_cls": "classification_forward",
    "text_encoder": "encoder_forward",
    "audio_encoder": "audio_encoder_forward",
}


def workload_requests() -> dict[str, dict[str, Any]]:
    """One small request per workload family, built from configs only (no
    weights downloaded). Vision uses the Hugging Face ResNet head
    (nn.Flatten), which is what CloudGPUFit sends for vision_cls."""
    import compiled_models
    from transformers import ResNetConfig

    def config(name: str) -> dict[str, Any]:
        return compiled_models.build_transformer_config(name).to_dict()

    resnet = ResNetConfig(
        depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 256, 512], layer_type="basic"
    ).to_dict()
    resnet["architectures"] = ["ResNetForImageClassification"]
    common = {"dtype": "fp16", "batch_size": 2}
    return {
        "decoder_llm": {**SMOKE_REQUEST, "batch_size": 2},
        "text_encoder": {**common, "model": "bert-base-uncased", "type": "text_encoder",
                         "model_config": config("bert"), "params": {"seq_len": 128}},
        "text_encdec": {**common, "model": "t5-small", "type": "text_encdec",
                        "model_config": config("t5"),
                        "params": {"encoder_seq_len": 128, "decoder_seq_len": 1}},
        "audio_encoder": {**common, "model": "facebook/wav2vec2-base", "type": "audio_encoder",
                          "model_config": config("wav2vec2"), "params": {"audio_samples": 32000}},
        "vision_cls": {**common, "model": "microsoft/resnet-18", "type": "vision_cls",
                       "model_config": resnet, "params": {"resolution": 224}},
    }


COMMON_RESULT_KEYS = {
    "status", "model", "model_type", "workload_type", "instance_type",
    "target_instance", "target_gpu", "engine_mode", "dtype", "batch_size",
    "params", "predicted_median_ms", "raw_predicted_median_ms",
    "predicted_p90_ms", "predicted_throughput_rps", "official_estimate",
    "p90_status", "profile_provenance", "prediction_scope", "warnings",
    "breakdown",
}

# |our MAPE - official MAPE| allowed in the full suite (percentage points).
MAPE_TOLERANCE_PP = 0.05
REL_TOL = 1e-6


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passed = 0

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        if ok:
            self.passed += 1
            print(f"  PASS {label}")
        else:
            self.failures.append(f"{label}: {detail}")
            print(f"  FAIL {label}: {detail}")


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=1e-12)


def smoke(report: Report, update_snapshot: bool) -> None:
    import eager_runtime as er
    from compiled_external_model import load_external_manifest
    from compiled_fusion_model import load_fusion_manifest

    targets = {mode: er.available_targets(engine_mode=mode) for mode in ENGINE_MODES}
    all_targets = sorted(set(targets["eager"]) | set(targets["compiled"]))
    print(f"installed targets: eager={targets['eager']} compiled={targets['compiled']}")

    for target in all_targets:
        print(f"\n[{target}]")
        target_dir = RUNTIME_ROOT / "targets" / target
        snapshot_path = HERE / "baselines" / target / "smoke_snapshot.json"
        snapshot = (
            json.loads(snapshot_path.read_text(encoding="utf-8"))
            if snapshot_path.is_file() and not update_snapshot
            else {}
        )
        new_snapshot: dict[str, Any] = {"request": SMOKE_REQUEST, "results": {}}

        # Manifests: validated in a subprocess so the environment variables
        # validate_manifest() sets cannot leak between targets.
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "from pathlib import Path; from runtime_manifest import validate_manifest;"
            "m = validate_manifest(Path(sys.argv[2]), expected_target=sys.argv[3], check_gpu=False);"
            "print(m['gpu_name'])"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code, str(RUNTIME_ROOT),
             str(target_dir / "runtime_manifest.json"), target],
            capture_output=True, text=True,
        )
        report.check(proc.returncode == 0, f"{target} runtime_manifest valid",
                     proc.stderr.strip().splitlines()[-1] if proc.stderr else "")
        if target in targets["compiled"]:
            try:
                load_external_manifest(RUNTIME_ROOT, target)
                load_fusion_manifest(RUNTIME_ROOT, target)
                report.check(True, f"{target} compiled manifests valid")
            except Exception as exc:  # noqa: BLE001
                report.check(False, f"{target} compiled manifests valid", repr(exc))

        context_file = target_dir / "compiled_context_calibration.json"
        if target in REJECTED_CONTEXT_CALIBRATION:
            report.check(not context_file.exists(),
                         f"{target} rejected compiled_context_calibration.json absent",
                         str(context_file))

        results = {}
        for mode in ENGINE_MODES:
            if target not in targets[mode]:
                report.check(False, f"{target} {mode} installed", "missing")
                continue
            result = er.predict_runtime(SMOKE_REQUEST, instance_type=target, engine_mode=mode)
            results[mode] = result
            label = f"{target} {mode}"
            report.check(result.get("status") == "ok", f"{label} prediction ok",
                         f"{result.get('error_type')}: {result.get('error')}")
            if result.get("status") != "ok":
                continue
            report.check(result["instance_type"] == target and result["engine_mode"] == mode,
                         f"{label} routed to targets/{target}", str((result["instance_type"], result["engine_mode"])))
            missing = sorted(COMMON_RESULT_KEYS - set(result))
            report.check(not missing, f"{label} common result format", f"missing {missing}")
            report.check(result["prediction_scope"] == "first_token_logits"
                         and result["workload_type"] == "decoder_llm",
                         f"{label} first-token scope", str(result.get("prediction_scope")))
            if mode == "compiled":
                applied = result["breakdown"]["median_pipeline"].get("context_calibration_applied", 0.0)
                expected = 1.0 if context_file.is_file() else 0.0
                report.check(applied == expected, f"{label} context calibration applied={expected}",
                             f"applied={applied}")
            new_snapshot["results"][mode] = {
                "predicted_median_ms": result["predicted_median_ms"],
                "predicted_p90_ms": result["predicted_p90_ms"],
            }
            reference = snapshot.get("results", {}).get(mode)
            if reference:
                ok = all(_close(result[k], reference[k]) for k in reference)
                report.check(ok, f"{label} matches snapshot",
                             f"now={new_snapshot['results'][mode]} snapshot={reference}")

        if len(results) == 2 and all(r.get("status") == "ok" for r in results.values()):
            report.check(results["eager"]["predicted_median_ms"] != results["compiled"]["predicted_median_ms"],
                         f"{target} eager/compiled branches differ")

        if target == "g6.xlarge":
            for mode, reference in PRE_MERGE_G6_REFERENCE.items():
                result = results.get(mode, {})
                ok = result.get("status") == "ok" and all(_close(result[k], v) for k, v in reference.items())
                report.check(ok, f"g6.xlarge {mode} unchanged vs pre-merge service",
                             f"now={[result.get(k) for k in reference]} ref={list(reference.values())}")

        # Every workload family through the public API, both modes.
        new_snapshot["workloads"] = {}
        for workload, request in workload_requests().items():
            for mode in ENGINE_MODES:
                if target not in targets[mode]:
                    continue
                result = er.predict_runtime(request, instance_type=target, engine_mode=mode)
                label = f"{target} {mode} {workload}"
                ok = (
                    result.get("status") == "ok"
                    and result.get("workload_type") == workload
                    and result.get("prediction_scope") == WORKLOAD_SCOPES[workload]
                )
                report.check(ok, f"{label} ok, scope={WORKLOAD_SCOPES[workload]}",
                             f"{result.get('status')} {result.get('workload_type')} "
                             f"{result.get('prediction_scope')} {result.get('error', '')[:160]}")
                if result.get("status") != "ok":
                    continue
                key = f"{workload}/{mode}"
                new_snapshot["workloads"][key] = result["predicted_median_ms"]
                reference = snapshot.get("workloads", {}).get(key)
                if reference is not None:
                    report.check(_close(result["predicted_median_ms"], reference),
                                 f"{label} matches snapshot",
                                 f"now={result['predicted_median_ms']} snapshot={reference}")

        if update_snapshot and new_snapshot["results"]:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            new_snapshot["note"] = (
                "Snapshot of this service's own predictions (not measurements); "
                "used only to detect unintended drift."
            )
            snapshot_path.write_text(json.dumps(new_snapshot, indent=2) + "\n", encoding="utf-8", newline="\n")
            print(f"  wrote {snapshot_path.relative_to(RUNTIME_ROOT)}")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(RUNTIME_ROOT))


def full(report: Report, only: list[str] | None) -> None:
    out_root = HERE / "results"
    for baseline_dir in sorted((HERE / "baselines").iterdir()):
        target = baseline_dir.name
        if only and target not in only:
            continue
        for mode in ENGINE_MODES:
            baseline_path = baseline_dir / f"{mode}_e2e_summary.json"
            if not baseline_path.is_file():
                continue
            print(f"\n[{target} {mode} full E2E]")
            out_dir = out_root / target
            out_dir.mkdir(parents=True, exist_ok=True)
            summary_path = out_dir / f"{mode}_e2e_summary.json"
            if mode == "compiled":
                cmd = [sys.executable, str(HERE / "validate_compiled_predictor_e2e.py"),
                       "--target", target,
                       "--output", str(out_dir / "compiled_e2e.csv"),
                       "--summary", str(summary_path)]
                data = HERE / "data" / target / "compiled_validation_data"
                if data.is_dir():
                    cmd += ["--validation-data", str(data)]
            else:
                cmd = [sys.executable, str(HERE / "validate_eager_predictor_e2e.py"),
                       "--target", target,
                       "--actual", str(HERE / "data" / target / "eager_e2e_actual.csv"),
                       "--output", str(out_dir / "eager_e2e.csv"),
                       "--summary", str(summary_path)]
            proc = _run(cmd)
            if not summary_path.is_file():
                report.check(False, f"{target} {mode} full E2E ran", proc.stderr[-800:])
                continue
            ours = json.loads(summary_path.read_text(encoding="utf-8"))
            ref = json.loads(baseline_path.read_text(encoding="utf-8"))
            if "failed_rows" in ref:
                report.check(ours.get("failed_rows") == ref["failed_rows"] == 0,
                             f"{target} {mode} failed rows 0",
                             f"ours={ours.get('failed_rows')} ref={ref['failed_rows']}")
            report.check(ours["overall"]["rows"] == ref["overall"]["rows"],
                         f"{target} {mode} row count", f"{ours['overall']['rows']} vs {ref['overall']['rows']}")
            for metric in ("mape_percent", "mean_bias_percent"):
                a, b = ours["overall"][metric], ref["overall"][metric]
                report.check(abs(a - b) <= MAPE_TOLERANCE_PP,
                             f"{target} {mode} {metric} {a:.3f} vs official {b:.3f}",
                             f"diff={a - b:+.4f}pp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also re-run the per-target E2E validations")
    parser.add_argument("--full-only", action="store_true", help="skip the smoke suite")
    parser.add_argument("--targets", nargs="+", help="limit the full suite to these targets")
    parser.add_argument("--update-snapshot", action="store_true")
    parser.add_argument("--allow-env-mismatch", action="store_true",
                        help="run anyway with a different transformers version (structural checks only are meaningful)")
    args = parser.parse_args()

    report = Report()
    import transformers

    if transformers.__version__ != REQUIRED_TRANSFORMERS:
        message = (
            f"transformers {transformers.__version__} is installed but the targets "
            f"were calibrated with {REQUIRED_TRANSFORMERS}; numeric comparisons "
            "would be meaningless."
        )
        if not args.allow_env_mismatch:
            print("ENVIRONMENT ERROR:", message)
            raise SystemExit(2)
        print("WARNING:", message)
    if not args.full_only:
        smoke(report, args.update_snapshot)
    if args.full or args.full_only:
        full(report, args.targets)

    print(f"\n{report.passed} passed, {len(report.failures)} failed")
    for failure in report.failures:
        print("  FAIL", failure)
    raise SystemExit(1 if report.failures else 0)


if __name__ == "__main__":
    main()
