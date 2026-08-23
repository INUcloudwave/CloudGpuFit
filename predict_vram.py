#!/usr/bin/env python3
"""
predict_vram.py — CPU-only GPU VRAM 예측 래퍼

xMem 저장소 루트에 두고 실행한다. xMem 코어 코드는 수정하지 않으며,
합성 데이터로더를 주입해 사용자가 지정한 shape 그대로 프로파일링한다.

  python predict_vram.py --model facebook/opt-125m --batch_size 8 --seq_len 512
  python predict_vram.py --model ResNet152 --batch_size 32 --resolution 224
  python predict_vram.py --json '{"model":"distilgpt2","batch_size":4,"seq_len":256}'

출력: stdout 마지막 줄에 결과 JSON 1개.
"""
import argparse
import copy
import json
import os
import sys
import time
import uuid
import warnings

import torch

# ───────────────────────────────────────── 실측 기반 상수
# dtype 스케일: FP32 프로파일 → 목표 dtype 환산.
# 6개 모델 실측 비율 0.5015~0.5552 의 보수적 상한.
DTYPE_SCALE = {"fp32": 1.0, "fp16": 0.5, "bf16": 0.5}
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}

# tensor 예측 과소추정 보정. LLM 12조건 관측 최대 -2.37%.
UNDERESTIMATE_CONSTANT = 32 * 1024 ** 2

# reserved-allocated gap + CUDA context + 여유를 흡수하는 고정 마진.
MARGIN_BYTES = 2 * 1024 ** 3

# 검증 범위 (이 밖은 warning)
VALIDATED = {
    "llm_batch": (1, 30),
    "llm_seq": (128, 2048),
    "vision_batch": (32, 64),
    "vision_res": (86, 86),
}

PROFILE_ITERATIONS = 2      # xMem train_on_cpu 기본값과 동일
SAMPLES_PER_LOADER = 4      # iterations=2 → 최소 3배치 필요


# ───────────────────────────────────────── 합성 데이터로더
class _LLMDataset(torch.utils.data.Dataset):
    def __init__(self, n, seq_len, vocab_size):
        self.n, self.seq_len = n, seq_len
        self.vocab = max(2, min(vocab_size, 30000))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        ids = torch.randint(0, self.vocab, (self.seq_len,), generator=g)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _VisionDataset(torch.utils.data.Dataset):
    def __init__(self, n, res):
        self.n, self.res = n, res

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        x = torch.randn(3, self.res, self.res, generator=g)
        return x, torch.tensor(0)


def build_loader(kind, batch_size, *, seq_len=None, resolution=None, vocab_size=None):
    n = batch_size * SAMPLES_PER_LOADER
    ds = (_LLMDataset(n, seq_len, vocab_size) if kind == "llm"
          else _VisionDataset(n, resolution))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


# ───────────────────────────────────────── 모델 로딩
def load_model(model_name):
    """(model, kind, hf_config) 반환. kind는 'llm' 또는 'vision'."""
    from perf_estimator.models import AllModels

    if getattr(AllModels, model_name, None) is not None:
        return getattr(AllModels, model_name).value, "vision", None

    from transformers import AutoConfig
    from utils.huggingface import suggest_automodel_class

    cfg = AutoConfig.from_pretrained(model_name)
    cfg.torch_dtype = torch.float32
    model_class = suggest_automodel_class(model_name)
    model = model_class.from_config(cfg)
    model.train()          # xMem ModelPreparer와 동일 (추론 루프가 eval로 전환)
    return model, "llm", cfg


def param_bytes(model):
    return (sum(p.numel() * p.element_size() for p in model.parameters())
            + sum(b.numel() * b.element_size() for b in model.buffers()))


# ───────────────────────────────────────── CPU 프로파일링
def profile_on_cpu(model, loader, is_transformer):
    from exp.trainer import ProfilerPlugin
    from exp.trainer.trainer import ModelTrainer
    from perf_estimator.config import Config
    from utils import search_profiler_file

    conf = Config(save2tmp=False)
    conf.task_id = f"CPU_{uuid.uuid4().hex[:3]}"

    trainer = ModelTrainer(
        model=copy.deepcopy(model),
        data_loader=loader,
        on_cpu=True,
        gpu_id=0,
        iterations=PROFILE_ITERATIONS,
        optimiser=torch.optim.SGD,      # 추론 경로에서는 미사용
        config=conf,
        plugins=[ProfilerPlugin(config=conf)],
        zero_grad_mode=conf.trainer.zero_out,
    )
    trainer.train(is_transformer=is_transformer)

    files = search_profiler_file(conf.result_dir)
    if not files:
        raise RuntimeError(f"프로파일 파일을 찾지 못했습니다: {conf.result_dir}")
    return files[-1]


# ───────────────────────────────────────── xMem 추정
def estimate(profiler_file, batch_size, kind, model_name, gpu_memory_gb):
    from perf_estimator.config import Config
    from perf_estimator.log import init_logging
    from perf_estimator.xmem import XMem

    conf = Config(save2tmp=False)
    if kind == "llm":
        conf.trainer.huggingface_enable = True
        conf.trainer.huggingface_model_name = model_name
    else:
        conf.trainer.huggingface_enable = False
        conf.trainer.huggingface_model_name = None

    init_logging(level="ERROR", conf=conf)
    xmem = XMem(batch_size=batch_size,
                max_gpu_memory_in_gb=gpu_memory_gb,
                config=conf)
    return xmem.estimate(profiler_file=profiler_file)


# ───────────────────────────────────────── KV cache (실험적)
def kv_cache_bytes(cfg, batch_size, total_len, dtype):
    """generate 모드용 해석적 추정. 미검증."""
    try:
        layers = getattr(cfg, "num_hidden_layers", None) or cfg.n_layer
        heads = getattr(cfg, "num_attention_heads", None) or cfg.n_head
        hidden = getattr(cfg, "hidden_size", None) or cfg.n_embd
        kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
        head_dim = getattr(cfg, "head_dim", None) or (hidden // heads)
    except Exception:
        return None
    return 2 * layers * kv_heads * head_dim * total_len * batch_size * DTYPE_BYTES[dtype]


# ───────────────────────────────────────── 입력 검증
def collect_warnings(kind, batch_size, seq_len, resolution, mode):
    w = []
    if kind == "llm":
        lo, hi = VALIDATED["llm_batch"]
        if not lo <= batch_size <= hi:
            w.append(f"batch_size {batch_size} is outside the validated range {lo}-{hi}")
        lo, hi = VALIDATED["llm_seq"]
        if not lo <= seq_len <= hi:
            w.append(f"seq_len {seq_len} is outside the validated range {lo}-{hi}; "
                     "long sequences may switch GPU attention kernels (FlashAttention), "
                     "which the CPU trace cannot reproduce")
    else:
        lo, hi = VALIDATED["vision_batch"]
        if not lo <= batch_size <= hi:
            w.append(f"batch_size {batch_size} is outside the validated range {lo}-{hi}")
        if resolution != VALIDATED["vision_res"][0]:
            w.append(f"resolution {resolution} is outside the validated value "
                     f"{VALIDATED['vision_res'][0]}; activation memory scales "
                     "quadratically with resolution")
    if mode == "generate":
        w.append("mode=generate is EXPERIMENTAL: KV cache is estimated analytically "
                 "and has not been validated against GPU measurements")
    return w


# ───────────────────────────────────────── 메인
def run(args):
    t0 = time.time()
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)

    model, kind, hf_cfg = load_model(args.model)
    pb = param_bytes(model)

    if kind == "llm":
        loader = build_loader("llm", args.batch_size, seq_len=args.seq_len,
                              vocab_size=getattr(hf_cfg, "vocab_size", 30000))
    else:
        loader = build_loader("vision", args.batch_size, resolution=args.resolution)

    warns = collect_warnings(kind, args.batch_size, args.seq_len,
                             args.resolution, args.mode)

    os.environ["XMEM_INFER"] = "1"
    os.environ["XMEM_PARAM_BYTES"] = str(pb)      # CNN 경로에서만 사용됨

    trace = profile_on_cpu(model, loader, is_transformer=(kind == "llm"))
    raw = estimate(trace, args.batch_size, kind, args.model, args.gpu_memory_gb)

    allocated_fp32 = raw["memory"]["tensor"]
    scaled = allocated_fp32 * DTYPE_SCALE[args.dtype]
    corrected = scaled + UNDERESTIMATE_CONSTANT * DTYPE_SCALE[args.dtype]

    kv = None
    if args.mode == "generate" and kind == "llm" and hf_cfg is not None:
        kv = kv_cache_bytes(hf_cfg, args.batch_size,
                            args.seq_len + args.max_new_tokens, args.dtype)
        if kv is None:
            warns.append("KV cache could not be computed from the model config; "
                         "the result covers prefill only")

    required = corrected + (kv or 0) + MARGIN_BYTES

    return {
        "status": "ok",
        "input": {
            "model": args.model, "type": kind, "batch_size": args.batch_size,
            "seq_len": args.seq_len if kind == "llm" else None,
            "resolution": args.resolution if kind == "vision" else None,
            "dtype": args.dtype, "mode": args.mode,
            "max_new_tokens": args.max_new_tokens if args.mode == "generate" else None,
        },
        "breakdown_bytes": {
            "param_bytes_fp32": pb,
            "predicted_allocated_fp32": allocated_fp32,
            "predicted_allocated_scaled": int(scaled),
            "after_underestimate_correction": int(corrected),
            "kv_cache_estimate": kv,
            "fixed_margin": MARGIN_BYTES,
        },
        "required_vram_bytes": int(required),
        "required_vram_gb": round(required / 1024 ** 3, 2),
        "elapsed_seconds": round(time.time() - t0, 1),
        "warnings": warns,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", help="모든 인자를 JSON 문자열로 전달")
    p.add_argument("--model")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--dtype", choices=list(DTYPE_SCALE), default="fp16")
    p.add_argument("--mode", choices=["prefill", "generate"], default="prefill")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--gpu_memory_gb", type=float, default=80)
    a = p.parse_args()

    if a.json:
        for k, v in json.loads(a.json).items():
            setattr(a, k, v)
    if not a.model:
        p.error("--model is required")

    try:
        result = run(a)
    except Exception as e:
        result = {"status": "error", "error": f"{type(e).__name__}: {e}",
                  "input": {"model": a.model, "batch_size": a.batch_size}}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
