Target-aware Eager Runtime

Directory layout

eager_runtime/
├── predict_eager_runtime.py
├── run_eager_local.py
├── runtime_manifest.py
├── fxgraph_*.py
└── targets/
    └── g6.xlarge/
        ├── runtime_manifest.json
        ├── hardware_profile.json
        ├── p90_calibration_factor.json
        ├── gemm_eager_median.pkl
        ├── gemm_eager_90th.pkl
        ├── reduction_eager_median.pkl
        ├── reduction_eager_90th.pkl
        ├── cache_model/
        ├── conv1d_model/
        ├── linear_context_model_deep_v2/
        ├── memory_model/
        ├── pointwise_model_v4/
        └── sdpa_model/

The artifact names are GPU-neutral. The containing target directory identifies
the instance profile for which they were measured and trained.

Validate the packaged g6.xlarge profile on a CPU computer

python runtime_manifest.py \
  --target g6.xlarge \
  --check targets/g6.xlarge/runtime_manifest.json \
  --skip-gpu-check

Predict from a request

python predict_eager_runtime.py \
  --target g6.xlarge \
  --request request.json

--manifest is optional. When omitted, the selected path is automatically:

targets/<target>/runtime_manifest.json

Import from another backend

Import the package from the directory containing eager_runtime/:

from eager_runtime import predict_runtime

request = {
    "model": "facebook/opt-125m",
    "dtype": "fp16",
    "batch_size": 1,
    "params": {"seq_len": 128},
}

result = predict_runtime(request, target="g6.xlarge")

Predict only the instances that passed the VRAM filter:

from eager_runtime import predict_runtime_targets

result = predict_runtime_targets(
    request,
    targets=["g4dn.xlarge", "g5.xlarge", "g6.xlarge", "g6e.xlarge"],
)

Use available_targets() to list locally installed target profiles. API errors
are returned as structured dictionaries by default. Pass raise_on_error=True
when the caller should handle Python exceptions directly.

Run the fixed regression models

python run_eager_local.py \
  --target g6.xlarge \
  --models ResNet-18,BERT-Base \
  --batch-sizes 1

Add another GPU target

Create targets/<instance>/ with the same generic artifact filenames.

Add that target's hardware_profile.json.

Add that target's p90_calibration_factor.json.

Generate its manifest on the matching GPU instance:

python runtime_manifest.py --target <instance>

Copy the complete target directory to the CPU prediction package and run
predict_eager_runtime.py --target <instance> ....

Do not reuse the g6.xlarge PKLs, hardware profile, or P90 factor for another
GPU target. The folder structure is reusable; the learned artifacts and measured
hardware values are target-specific.