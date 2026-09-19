"""Per-workload-type rules for a job's `params` dict.

Kept in its own module so it can be tested without importing main.py (which
pulls in torch/eager_runtime), and so there is one written-down definition of
what each workload type accepts. web/lib/workloadSchema.ts mirrors these rules
for the form; this module is what actually enforces them.
"""
from __future__ import annotations

from typing import Any

# workload type -> (required param keys, optional param keys). Types missing
# here aren't checked at all -- create_job's SUPPORTED_TYPES check is what
# reports those, with an error code the UI already understands.
PARAM_SPEC: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "decoder_llm": (frozenset({"seq_len"}), frozenset({"mode", "max_new_tokens"})),
    "vision_cls": (frozenset({"resolution"}), frozenset()),
    "text_encdec": (
        frozenset({"encoder_seq_len", "decoder_seq_len"}),
        frozenset({"mode", "max_new_tokens"}),
    ),
}

MODES = frozenset({"prefill", "generate"})


def validate_params(workload_type: str, params: dict[str, Any]) -> None:
    """Raises ValueError with a user-facing message if `params` doesn't match
    the workload type's contract."""
    spec = PARAM_SPEC.get(workload_type)
    if spec is None:
        return
    required, optional = spec

    # buildPayload() sends null for fields its form hid (max_new_tokens while
    # mode=prefill, ...), so a null means "not supplied", not "invalid".
    supplied = {key: value for key, value in params.items() if value is not None}

    unknown = sorted(set(supplied) - (required | optional))
    if unknown:
        raise ValueError(f"type={workload_type}에 없는 params 키입니다: {unknown}")

    missing = sorted(required - set(supplied))
    if missing:
        raise ValueError(f"type={workload_type}에 필요한 params가 없습니다: {missing}")

    for key, value in sorted(supplied.items()):
        if key == "mode":
            if value not in MODES:
                raise ValueError(
                    f"params.mode는 {sorted(MODES)} 중 하나여야 합니다 (받은 값: {value!r})"
                )
            continue
        # bool is an int subclass in Python -- rejected explicitly so True
        # doesn't sail through as 1.
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"params.{key}는 양의 정수여야 합니다 (받은 값: {value!r})")

    if supplied.get("mode") == "generate" and "max_new_tokens" not in supplied:
        raise ValueError("mode=generate일 때는 params.max_new_tokens가 필요합니다")
