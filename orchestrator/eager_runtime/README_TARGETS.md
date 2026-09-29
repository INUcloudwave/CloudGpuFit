Target-aware Eager / Compiled Runtime

Installed targets (both engine modes)

| instance_type | GPU | Eager | Compiled | Notes |
|---|---|---|---|---|
| g4dn.xlarge | Tesla T4 | yes | yes | |
| g5.xlarge | NVIDIA A10G | yes | yes | |
| g6.xlarge | NVIDIA L4 | yes | yes | GPU base for the L4 size family |
| g6e.xlarge | NVIDIA L40S | yes | yes | |
| g7.2xlarge | RTX PRO 4500 Blackwell | yes | yes | Compiled context calibration; Eager P90 not E2E-calibrated yet |
| g7e.2xlarge | RTX PRO 6000 Blackwell | yes | yes | Compiled context calibration |
| gr6.4xlarge | NVIDIA L4 | yes | yes | Size target: L4 kernel models shared from g6.xlarge; its Compiled context calibration was rejected and is not used |

Prediction scope (`prediction_scope` in every result), per workload:

| workload_type | prediction_scope | covers |
|---|---|---|
| decoder_llm, text_encdec | first_token_logits | prefill + LM head up to the first generated token's logits (no decode loop) |
| vision_cls | classification_forward | one classification forward pass |
| text_encoder | encoder_forward | one encoder forward pass |
| audio_encoder | audio_encoder_forward | one Wav2Vec2/HuBERT-style encoder forward pass |

Median (`predicted_median_ms`) is the official number; Compiled P90 is diagnostic.

Directory layout

eager_runtime/
├── __init__.py                  public API (predict_runtime, predict_runtime_targets, available_targets)
├── predict_runtime.py           engine_mode / instance_type dispatcher
├── predict_eager_runtime.py     Eager predictor
├── predict_compiled_runtime.py  Compiled (torch.compile / Inductor) predictor
├── build_size_target.py         builds size targets from size_profiles/
├── size_profiles/               GPU family -> base target, per-size overhead profiles
├── regression/                  official E2E baselines, measured data, regression runner
└── targets/
    ├── g6.xlarge/               full target: GPU kernel models + size-level files
    │   ├── runtime_manifest.json          Eager artifacts (size + sha256 pinned)
    │   ├── hardware_profile.json          GPU specs + host overheads
    │   ├── p90_calibration_factor.json
    │   ├── first_token_calibration.json   (optional)
    │   ├── *.pkl, cache_model/, sdpa_model/, ...          Eager kernel models
    │   ├── compiled_external_model/       Compiled external-kernel LightGBM models + manifest
    │   ├── compiled_fusion_model/         Compiled fusion-group LightGBM models + manifest
    │   └── compiled_context_calibration.json   (optional, model-independent)
    └── gr6.4xlarge/             size target: only size-level files; GPU artifacts
                                 referenced from ../g6.xlarge/ and sha256-pinned

Predict

from eager_runtime import predict_runtime, predict_runtime_targets, available_targets

# keyword style
result = predict_runtime(
    model="facebook/opt-125m",
    instance_type="g6.xlarge",
    engine_mode="compiled",        # default "eager"
    batch_size=8,
    sequence_length=128,
)

# request-dict style (CloudGPUFit's existing call keeps working unchanged)
request = {
    "model": "facebook/opt-125m",
    "instance_type": "g6.xlarge",  # "target_instance" is accepted too
    "engine_mode": "eager",
    "dtype": "fp16",
    "batch_size": 1,
    "params": {"seq_len": 128},
}
result = predict_runtime(request)
result = predict_runtime(request, target="g6.xlarge")   # old positional form

# instances that passed the VRAM filter, one engine mode for all
result = predict_runtime_targets(request, targets=["g5.xlarge", "g6.xlarge"], engine_mode="compiled")

available_targets()                       # Eager targets
available_targets(engine_mode="compiled") # Compiled targets

Selection and safety

- instance_type picks targets/<instance_type>/. If it is not installed for the
  requested engine_mode the call returns a structured error listing the
  installed targets -- it never falls back to another GPU's artifacts.
- Every prediction validates runtime_manifest.json (target name, file size and
  sha256 of every artifact). Compiled manifests are validated the same way.
- Eager and Compiled results share one format: status, model, workload_type,
  instance_type, target_gpu, engine_mode, predicted_median_ms,
  raw_predicted_median_ms, predicted_p90_ms, predicted_throughput_rps,
  official_estimate, p90_status, profile_provenance, prediction_scope,
  warnings, breakdown. Compiled adds its compile_* settings.
- profile_provenance says where the target's host-overhead numbers come from:
  measured / interpolated / extrapolated. It is metadata only.

Add a new GPU (new target)

Create targets/<instance_type>/ with the same generic artifact filenames and
its runtime_manifest.json (python runtime_manifest.py --target <instance> on
the matching GPU instance). Add compiled_external_model/ and
compiled_fusion_model/ for Compiled. The directory is picked up automatically
by available_targets() -- no code change. Never reuse another GPU's kernel
models, hardware profile or P90 factor.

Add another size of an existing GPU (g6.2xlarge, g6.4xlarge, ...)

Instances with the same GPU share the GPU kernel-time models; only host-side
overheads differ. Do not scale the whole GPU latency by vCPU count.

size_profiles/<gpu_family>/family.json             GPU name + base target (e.g. g6 -> g6.xlarge)
size_profiles/<gpu_family>/sizes/<instance>.json   one size: vcpu, ram_gib, overheads, provenance

A size profile carries: vcpu, ram_gib, H2D/D2H base latency and bandwidth,
dispatch light/heavy, synchronization, Python loop overhead, the Compiled
fixed overhead (compiled_context_calibration: null, "inherit_base" or values),
and provenance.status (measured / interpolated / extrapolated; the latter two
must list their measured basis sizes). Then:

python build_size_target.py --profile size_profiles/g6/sizes/g6.2xlarge.json

This writes a thin targets/g6.2xlarge/ whose manifests point at the base's GPU
artifacts (sha256-pinned) and whose hardware_profile.json takes only the
overhead fields from the size profile. If the base target is retrained, the
thin target fails validation until it is rebuilt. The builder refuses to
overwrite a full, directly calibrated target.

size_profiles/*/sizes/<base>.json documents each base size (measured) and is
not built.

Regression tests

python regression/run_regression.py          # 7 targets x 2 modes x 5 workloads smoke suite (~3 min)
python regression/run_regression.py --full   # + re-run each target's official 78-case E2E validation

The full suite compares overall MAPE and bias with the official final E2E
summaries in regression/baselines/<instance>/ (tolerance 0.05 pp).
regression/baselines/gr6.4xlarge/diagnostics/ keeps the rejected gr6 Compiled
context calibration for diagnosis only; nothing loads it.
