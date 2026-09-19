#!/usr/bin/env python3
"""Run a frozen target-GPU Eager predictor without requiring a local GPU."""

from __future__ import annotations

import argparse
import gc
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
import torchvision.models as models
import lightgbm as lgb
from torch.fx.proxy import TraceError
from transformers import (
    BertConfig,
    BertModel,
    OPTConfig,
    OPTModel,
    Qwen2Config,
    Qwen2Model,
    RobertaConfig,
    RobertaModel,
    T5Config,
    T5Model,
    Wav2Vec2Config,
    Wav2Vec2Model,
)

import fxgraph_conv1d_v10 as v10
import fxgraph_eager_final as final
import fxgraph_pointwise_v7 as v7
from runtime_manifest import DEFAULT_TARGET, manifest_for_target, validate_manifest


ANALYSIS_DEVICE = torch.device("meta")
RUNTIME_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = ["ResNet-18"]
DEFAULT_BATCH_SIZES = [1]


ORIGINAL_LIGHTGBM_BOOSTER = lgb.Booster


def enable_unicode_safe_lightgbm_loading() -> None:
    def load_booster(*args, **kwargs):
        model_file = kwargs.pop("model_file", None)
        if model_file is not None:
            model_path = Path(model_file)
            kwargs["model_str"] = model_path.read_text(encoding="utf-8")
        return ORIGINAL_LIGHTGBM_BOOSTER(*args, **kwargs)

    lgb.Booster = load_booster


def trace_model_compat(
    parser: v10.Conv1dContextFXGraphParser,
    model: torch.nn.Module,
    dummy_input: torch.Tensor | dict[str, torch.Tensor],
    is_hf: bool,
    model_name: str,
):
    if model_name != "Wav2Vec2":
        return parser.trace_model(model, dummy_input=dummy_input, is_hf=is_hf)

    from transformers.utils.fx import HFProxy

    original_bool = HFProxy.__bool__

    def compatible_bool(proxy):
        try:
            return original_bool(proxy)
        except TraceError:
            return False

    HFProxy.__bool__ = compatible_bool
    try:
        return parser.trace_model(model, dummy_input=dummy_input, is_hf=is_hf)
    finally:
        HFProxy.__bool__ = original_bool


def image_input(batch_size: int) -> torch.Tensor:
    return torch.empty(
        batch_size,
        3,
        224,
        224,
        dtype=torch.float16,
        device=ANALYSIS_DEVICE,
    )


def token_input(batch_size: int, vocab_size: int) -> torch.Tensor:
    return torch.zeros(
        batch_size,
        128,
        dtype=torch.long,
        device=ANALYSIS_DEVICE,
    )


def seq2seq_input(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "input_ids": token_input(batch_size, 32128),
        "decoder_input_ids": token_input(batch_size, 32128),
    }


def audio_input(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "input_values": torch.empty(
            batch_size,
            32000,
            dtype=torch.float16,
            device=ANALYSIS_DEVICE,
        )
    }


def build_model_configs() -> dict[str, dict[str, object]]:
    return {
        "MobileNetV2": {
            "is_hf": False,
            "load_fn": lambda: models.mobilenet_v2(weights=None).half(),
            "dummy_fn": image_input,
            "bytes_fn": lambda bs, precision: (
                bs * 3 * 224 * 224 * precision,
                bs * 1000 * precision,
            ),
        },
        "RoBERTa-Base": {
            "is_hf": True,
            "load_fn": lambda: RobertaModel(
                RobertaConfig(
                    vocab_size=50265,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                )
            ).half(),
            "dummy_fn": lambda bs: token_input(bs, 50265),
            "bytes_fn": lambda bs, precision: (
                bs * 128 * 4,
                bs * 128 * 768 * precision,
            ),
        },
        "Qwen2-1.5B": {
            "is_hf": True,
            "load_fn": lambda: Qwen2Model(
                Qwen2Config(
                    vocab_size=151936,
                    hidden_size=1536,
                    intermediate_size=8960,
                    num_hidden_layers=28,
                    num_attention_heads=12,
                    num_key_value_heads=2,
                    use_cache=False,
                )
            ).half(),
            "dummy_fn": lambda bs: token_input(bs, 151936),
            "bytes_fn": lambda bs, precision: (
                bs * 128 * 4,
                bs * 128 * 1536 * precision,
            ),
        },
        "ResNet-18": {
            "is_hf": False,
            "load_fn": lambda: models.resnet18(weights=None).half(),
            "dummy_fn": image_input,
            "bytes_fn": lambda bs, precision: (
                bs * 3 * 224 * 224 * precision,
                bs * 1000 * precision,
            ),
        },
        "BERT-Base": {
            "is_hf": True,
            "load_fn": lambda: BertModel(
                BertConfig(
                    vocab_size=30522,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                )
            ).half(),
            "dummy_fn": lambda bs: token_input(bs, 30522),
            "bytes_fn": lambda bs, precision: (
                bs * 128 * 4,
                bs * 128 * 768 * precision,
            ),
        },
        "OPT-125m": {
            "is_hf": True,
            "load_fn": lambda: OPTModel(
                OPTConfig(
                    vocab_size=50272,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    ffn_dim=3072,
                )
            ).half(),
            "dummy_fn": lambda bs: token_input(bs, 50272),
            "bytes_fn": lambda bs, precision: (
                bs * 128 * 4,
                bs * 128 * 768 * precision,
            ),
        },
        "T5-Small": {
            "is_hf": True,
            "load_fn": lambda: T5Model(
                T5Config(
                    vocab_size=32128,
                    d_model=512,
                    d_kv=64,
                    d_ff=2048,
                    num_layers=6,
                    num_decoder_layers=6,
                    num_heads=8,
                    use_cache=False,
                )
            ).half(),
            "dummy_fn": seq2seq_input,
            "bytes_fn": lambda bs, precision: (
                (bs * 128 * 4) * 2,
                bs * 128 * 512 * precision,
            ),
        },
        "Wav2Vec2": {
            "is_hf": True,
            "load_fn": lambda: Wav2Vec2Model(
                Wav2Vec2Config(
                    vocab_size=32,
                    hidden_size=768,
                    num_hidden_layers=6,
                    num_attention_heads=8,
                    conv_dim=(512, 512, 512, 512, 512, 512, 512),
                )
            ).half(),
            "dummy_fn": audio_input,
            "bytes_fn": lambda bs, precision: (
                bs * 32000 * precision,
                bs * (32000 // 320) * 768 * precision,
            ),
        },
    }


def parse_csv_values(value: str, cast: Callable[[str], object]) -> list[object]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    configs = build_model_configs()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated names. Available: {', '.join(configs)}",
    )
    parser.add_argument(
        "--batch-sizes",
        default=",".join(map(str, DEFAULT_BATCH_SIZES)),
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Target profile directory under targets/ (default: g6.xlarge)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional explicit manifest path; normally selected by --target",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fx_breakdown_eager_local.csv"),
    )
    parser.add_argument(
        "--construction-device",
        choices=("meta", "cpu"),
        default="meta",
        help="Device used only while constructing and tracing the model",
    )
    return parser.parse_args()


def predict_model(
    model_name: str,
    config: dict[str, object],
    batch_sizes: list[int],
    parser: v10.Conv1dContextFXGraphParser,
    router: final.CalibratedEagerSimulatorRouter,
    construction_device: str,
) -> list[dict[str, object]]:
    load_fn = config["load_fn"]
    dummy_fn = config["dummy_fn"]
    bytes_fn = config["bytes_fn"]
    assert callable(load_fn) and callable(dummy_fn) and callable(bytes_fn)

    if construction_device == "meta":
        with torch.device(ANALYSIS_DEVICE):
            model = load_fn().eval()
    else:
        model = load_fn().eval()

    trace_input = dummy_fn(1)
    gm = trace_model_compat(
        parser,
        model,
        trace_input,
        bool(config["is_hf"]),
        model_name,
    )
    gm = gm.half().to(ANALYSIS_DEVICE)
    rows: list[dict[str, object]] = []

    print(f"\n[{model_name}]")
    print(f"{'Batch':<8} | {'Median':>12} | {'P90':>12}")
    print("-" * 39)

    for batch_size in batch_sizes:
        dummy_input = dummy_fn(batch_size)
        input_bytes, output_bytes = bytes_fn(batch_size, router.precision_bytes)
        nodes = parser.extract_nodes(gm, dummy_input, engine_mode="eager")

        _, median_log = router.simulate_fx_graph(
            nodes,
            batch_size=batch_size,
            prediction_mode="median",
        )
        _, p90_log = router.simulate_fx_graph(
            nodes,
            batch_size=batch_size,
            prediction_mode="90th",
        )
        median_ms, _ = router.simulate_e2e_pipeline(
            median_log,
            input_bytes,
            output_bytes,
        )
        raw_p90_ms, _ = router.simulate_e2e_pipeline(
            p90_log,
            input_bytes,
            output_bytes,
        )
        predicted_p90_ms = router.finalize_p90_prediction(
            median_e2e_ms=median_ms,
            raw_model_p90_ms=raw_p90_ms,
        )
        record = final.build_prediction_breakdown(
            router=router,
            model_name=model_name,
            batch_size=batch_size,
            mode="eager",
            log_df=median_log,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            predicted_e2e_ms=median_ms,
        )
        existing_p90_ms = max(
            median_ms * final.P90_MEDIAN_FLOOR_MULTIPLIER,
            raw_p90_ms,
        )
        record.update(
            {
                "raw_model_p90_ms": raw_p90_ms,
                "existing_p90_ms": existing_p90_ms,
                "predicted_p90_ms": predicted_p90_ms,
            }
        )
        rows.append(record)
        print(
            f"{batch_size:<8} | {median_ms:>9.3f} ms | "
            f"{predicted_p90_ms:>9.3f} ms"
        )

    del model, gm, trace_input
    gc.collect()
    return rows


def main() -> None:
    args = parse_args()
    configs = build_model_configs()
    selected_models = parse_csv_values(args.models, str)
    batch_sizes = parse_csv_values(args.batch_sizes, int)

    unknown = sorted(set(selected_models).difference(configs))
    if unknown:
        raise ValueError(f"Unknown model names: {', '.join(unknown)}")
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("Batch sizes must be positive integers")

    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else manifest_for_target(RUNTIME_ROOT, args.target)
    )
    manifest = validate_manifest(
        manifest_path,
        expected_target=args.target,
        check_gpu=False,
    )
    os.environ["EAGER_P90_CALIBRATION_ACTIVE"] = "1"
    print(
        "Runtime manifest valid: "
        f"instance={manifest['target_instance']}, "
        f"GPU={manifest['gpu_name']}, artifacts={len(manifest['artifacts'])}"
    )
    print("Analysis device: meta (local CUDA is not required)")
    print(f"Model construction device: {args.construction_device}")

    enable_unicode_safe_lightgbm_loading()
    parser = v10.Conv1dContextFXGraphParser()
    router = final.CalibratedEagerSimulatorRouter(engine_mode="eager")
    rows: list[dict[str, object]] = []

    for model_name in selected_models:
        rows.extend(
            predict_model(
                model_name=model_name,
                config=configs[model_name],
                batch_sizes=batch_sizes,
                parser=parser,
                router=router,
                construction_device=args.construction_device,
            )
        )
        pd.DataFrame(rows).to_csv(args.output, index=False)

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"\nSaved: {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
