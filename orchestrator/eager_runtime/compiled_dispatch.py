#!/usr/bin/env python3
"""Shared Inductor dispatch rules used by Compiled prediction routers.

The rules in this module describe backend lowering, not model identities.  A
node classified as ``triton`` must be accounted for by the analytical fusion
planner and must not also be sent to the external-library predictor.
"""

from __future__ import annotations

from collections.abc import Sequence


def matmul_reduction_size(
    left_shape: Sequence[int] | None,
    right_shape: Sequence[int] | None,
) -> int | None:
    """Return the K dimension for a valid matrix multiplication."""

    if left_shape is None or right_shape is None:
        return None
    if len(left_shape) < 2 or len(right_shape) < 2:
        return None
    left_k = int(left_shape[-1])
    right_k = int(right_shape[-2])
    if left_k != right_k:
        return None
    return left_k


def is_degenerate_inductor_matmul(
    operation: str,
    left_shape: Sequence[int] | None,
    right_shape: Sequence[int] | None,
) -> bool:
    """Whether Inductor lowers a K=1 matmul as elementwise/Triton work.

    K=1 ``mm``/``bmm``/``matmul`` has no real reduction.  Inductor can express
    it as broadcast multiplication and fuse it with adjacent pointwise work,
    so counting it as a cuBLAS External kernel double counts the boundary.
    Bias-bearing ``addmm`` and ``linear`` are deliberately excluded from this
    rule until their lowering is verified independently.
    """

    normalized = str(operation).lower().rsplit(".", 1)[-1]
    if normalized not in {"mm", "bmm", "matmul"}:
        return False
    return matmul_reduction_size(left_shape, right_shape) == 1
