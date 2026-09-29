"""Importable runtime prediction API (Eager + Compiled).

The package's public entry point is now ``predict_runtime`` (the
``engine_mode``-aware dispatcher in ``predict_runtime.py``), not
``predict_eager_runtime`` directly. Every existing caller -- including
CloudGPUFit's ``orchestrator/main.py`` and ``runtime_adapter.py``, which do
``from eager_runtime import available_targets, predict_runtime_targets`` and
never pass ``engine_mode`` -- keeps working unchanged: these three functions
default to Eager mode exactly as ``predict_eager_runtime``'s own versions did,
and only opt into Compiled mode when a caller explicitly asks for it.
"""

from .predict_runtime import (
    available_targets,
    predict_runtime,
    predict_runtime_targets,
    target_provenance,
)

__all__ = [
    "available_targets",
    "predict_runtime",
    "predict_runtime_targets",
    "target_provenance",
]
