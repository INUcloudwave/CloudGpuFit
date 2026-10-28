"""Generate (Prefill + cache-aware Decode) Request E2E prediction.

Kept separate from the existing forward/Prefill predictors: nothing here is
imported by them, and the Generate-only Compiled planner lives in
generate/compiled_fusion_ir.py.
"""
from .service import (  # noqa: F401
    GenerateInputError,
    predict_generate_targets,
    resolve_generate_inputs,
    select_anchors,
)
