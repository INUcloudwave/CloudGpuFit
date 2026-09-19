"""Importable Eager runtime prediction API."""

from .predict_eager_runtime import (
    available_targets,
    predict_runtime,
    predict_runtime_targets,
)

__all__ = [
    "available_targets",
    "predict_runtime",
    "predict_runtime_targets",
]
