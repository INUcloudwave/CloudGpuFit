#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Any


def parse_ints(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item.strip()})
    if not values or values[0] < 1:
        raise ValueError("max-new-tokens must contain positive integers")
    return values


def interpolate(cache_len: int, anchors: list[tuple[int, float]]) -> float:
    positions = [position for position, _ in anchors]
    index = bisect.bisect_left(positions, cache_len)
    if index < len(anchors) and positions[index] == cache_len:
        return anchors[index][1]
    if index == 0 or index == len(anchors):
        raise ValueError(
            f"cache_len={cache_len} is outside Graph anchors "
            f"[{positions[0]}, {positions[-1]}]"
        )
    left_x, left_y = anchors[index - 1]
    right_x, right_y = anchors[index]
    ratio = (cache_len - left_x) / (right_x - left_x)
    return left_y + ratio * (right_y - left_y)


def validate_components(report: dict[str, Any]) -> None:
    if report.get("status") != "ok":
        raise RuntimeError("component Graph report is not complete")
    if not report.get("rows"):
        raise RuntimeError("component Graph report contains no rows")
    for row in report["rows"]:
        eager = row["eager"]
        compiled = row["compiled"]
        if eager.get("unsupported_operations"):
            raise RuntimeError(
                f"unsupported Eager operations: {eager['unsupported_operations']}"
            )
        if compiled.get("unsupported_operations"):
            raise RuntimeError(
                f"unsupported Compiled operations: {compiled['unsupported_operations']}"
            )
        if int(eager.get("host_model_applied", 0)) != 1:
            raise RuntimeError(
                f"v33 Eager host model was not applied to "
                f"{row['model']}:{row['phase']}:{row['cache_len']}"
            )


def aggregate(
    report: dict[str, Any],
    *,
    batch_size: int,
    seq_len: int,
    max_new_tokens: list[int],
    hardware: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rows = report["rows"]
    for model in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        prefill = next(row for row in model_rows if row["phase"] == "prefill")
        decode = [row for row in model_rows if row["phase"] == "decode"]
        start_cache = 1 if model in {"t5", "bart"} else seq_len
        for engine in ("eager", "compiled"):
            anchors = sorted(
                (int(row["cache_len"]), float(row[engine]["scheduled_ms"]))
                for row in decode
            )
            for tokens in max_new_tokens:
                cache_lengths = range(start_cache, start_cache + tokens - 1)
                decode_values = [interpolate(value, anchors) for value in cache_lengths]
                prefill_ms = float(prefill[engine]["scheduled_ms"])
                decode_ms = sum(decode_values)
                input_bytes = batch_size * 2 * seq_len * 8
                output_bytes = batch_size * tokens * 8
                h2d_ms = float(hardware["t_h2d_base"]) + input_bytes / (
                    float(hardware["bw_h2d_GBs"]) * 1_000_000_000
                ) * 1000
                d2h_ms = float(hardware["t_d2h_base"]) + output_bytes / (
                    float(hardware["bw_d2h_GBs"]) * 1_000_000_000
                ) * 1000
                sync_ms = float(hardware["t_sync"])
                generate_ms = prefill_ms + decode_ms
                output.append(
                    {
                        "model": model,
                        "engine_mode": engine,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "max_new_tokens": tokens,
                        "prefill_ms": prefill_ms,
                        "decode_steps": tokens - 1,
                        "decode_ms": decode_ms,
                        "kernel_and_host_generate_ms": generate_ms,
                        "generate_startup_constant_ms": 0.0,
                        "per_token_control_constant_ms": 0.0,
                        "generate_control_ms": 0.0,
                        "h2d_ms": h2d_ms,
                        "d2h_ms": d2h_ms,
                        "final_sync_ms": sync_ms,
                        "predicted_generate_ms": generate_ms,
                        "predicted_request_inference_ms": (
                            generate_ms + h2d_ms + d2h_ms + sync_ms
                        ),
                        "first_decode_cache_length": (
                            start_cache if tokens > 1 else None
                        ),
                        "last_decode_cache_length": (
                            start_cache + tokens - 2 if tokens > 1 else None
                        ),
                        "cache_anchor_min": anchors[0][0],
                        "cache_anchor_max": anchors[-1][0],
                        "aggregation": "piecewise_linear_cache_anchor_sum",
                        "eager_host_model_applied": (
                            1 if engine == "eager" else None
                        ),
                        "compiled_path_modified_by_v33": False,
                    }
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hardware-profile", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--max-new-tokens", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if min(args.batch_size, args.seq_len) < 1:
        raise ValueError("batch-size and seq-len must be positive")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    hardware = json.loads(args.hardware_profile.read_text(encoding="utf-8"))
    validate_components(report)
    if int(report.get("batch_size", args.batch_size)) != args.batch_size:
        raise ValueError("batch-size does not match component Graph report")
    if int(report.get("seq_len", args.seq_len)) != args.seq_len:
        raise ValueError("seq-len does not match component Graph report")

    rows = aggregate(
        report,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_new_tokens=parse_ints(args.max_new_tokens),
        hardware=hardware,
    )
    payload = {
        "scope": "generate_request_inference_e2e_v40",
        "target": report.get("target"),
        "source_execution": report.get("execution"),
        "prediction_structure": {
            "eager": (
                "operation_ml_kernel_time + physical_dispatch + "
                "v33_framework_host_ml_with_cpu_gpu_overlap"
            ),
            "compiled": "existing_external_and_fusion_predictor_unchanged",
        },
        "request_inference_formula": (
            "H2D + use_cache_prefill + sum(cache_aware_decode) + D2H + final_sync"
        ),
        "generate_startup_constant_ms": 0.0,
        "per_token_control_constant_ms": 0.0,
        "model_specific_scale_count": 0,
        "rows": rows,
        "status": "ok",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
