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


class _EncDecDataset(torch.utils.data.Dataset):
    """encoder_input_ids/decoder_input_ids를 따로 낸다. T5/BART 등의 forward는
    이 두 시퀀스 길이가 독립이라, LLM 로더처럼 하나의 seq_len으로는 표현이 안 된다.
    decoder_input_ids[0]에 decoder_start_token_id를 넣는 것은 실제 추론과의 형태를
    맞추기 위함이고(값 자체는 메모리 사용량에 영향 없음), attention_mask는 인코더
    입력에만 준다 — 디코더 쪽은 합성 데이터라 causal mask를 모델이 내부에서 만든다."""

    def __init__(self, n, encoder_seq_len, decoder_seq_len, vocab_size, decoder_start_token_id):
        self.n = n
        self.encoder_seq_len, self.decoder_seq_len = encoder_seq_len, decoder_seq_len
        self.vocab = max(2, min(vocab_size, 30000))
        self.decoder_start_token_id = decoder_start_token_id

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        enc_ids = torch.randint(0, self.vocab, (self.encoder_seq_len,), generator=g)
        dec_ids = torch.randint(0, self.vocab, (self.decoder_seq_len,), generator=g)
        dec_ids[0] = self.decoder_start_token_id
        return {
            "input_ids": enc_ids,
            "attention_mask": torch.ones_like(enc_ids),
            "decoder_input_ids": dec_ids,
        }


class _VisionDataset(torch.utils.data.Dataset):
    """(tensor, label) 튜플을 낸다 — conv_infer_loop이 `model(inputs)`처럼
    위치 인자 하나로 부르는 형태. HF 이미지 분류 모델도 forward의 첫 인자가
    pixel_values라 같은 호출이 그대로 맞는다."""

    def __init__(self, n, res, channels=3):
        self.n, self.res, self.channels = n, res, channels

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        x = torch.randn(self.channels, self.res, self.res, generator=g)
        return x, torch.tensor(0)


def build_loader(kind, batch_size, *, seq_len=None, resolution=None, vocab_size=None,
                 channels=3):
    n = batch_size * SAMPLES_PER_LOADER
    ds = (_LLMDataset(n, seq_len, vocab_size) if kind == "llm"
          else _VisionDataset(n, resolution, channels))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def build_encdec_loader(batch_size, encoder_seq_len, decoder_seq_len, *,
                        vocab_size, decoder_start_token_id):
    n = batch_size * SAMPLES_PER_LOADER
    ds = _EncDecDataset(n, encoder_seq_len, decoder_seq_len, vocab_size, decoder_start_token_id)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


# ───────────────────────────────────────── 모델 로딩
# workload_type -> (AutoModel 클래스 이름, kind). kind는 아래 전부를 좌우한다:
# 어떤 합성 로더를 쓸지, ModelTrainer에 is_transformer로 뭘 넘길지(=패치가 고른
# 추론 루프), 그리고 estimate()가 huggingface_enable을 켤지.
#   llm    -> 텍스트 로더 + transformer_infer_loop(dict 입력, use_cache=False)
#   vision -> 이미지 로더 + conv_infer_loop(위치 인자 하나)
WORKLOAD_MODEL_CLASS = {
    "decoder_llm": ("AutoModelForCausalLM", "llm"),
    "text_encdec": ("AutoModelForSeq2SeqLM", "llm"),
    "vision_cls": ("AutoModelForImageClassification", "vision"),
}


def load_model(model_name, model_config=None, workload_type=None):
    """(model, kind, hf_config) 반환. kind는 'llm' 또는 'vision'.

    model_config(호출측이 이미 HF에서 받아둔 config.json 원본 dict)가 있으면
    그걸로 config를 복원해서 이 함수가 또 HF에 config.json을 직접 fetch하지
    않게 한다. 없으면(CLI 단독 실행 등) 예전처럼 자체적으로 받아온다.

    workload_type이 주어지면 AutoModel 클래스도 그걸로 정한다. 그러면 xMem의
    suggest_automodel_class()를 부르지 않게 되는데, 그 함수는 클래스를 고르려고
    huggingface_hub.model_info()로 pipeline_tag를 또 네트워크에서 받아온다.
    웹은 이미 pipeline_tag로 유형을 정해서 보내주므로 다시 물어볼 이유가 없다.
    """
    import transformers

    if workload_type is None and getattr_all_models(model_name) is not None:
        # 이름만 보고 torchvision 별칭으로 바꿔치기하는 건 CLI 단독 사용일 때뿐.
        # 호출측이 workload_type을 지정했다면 그건 HF 모델을 뜻하므로 건드리지
        # 않는다 (HF ID에는 "/"가 들어가 별칭과 겹치지도 않는다).
        return getattr_all_models(model_name).value, "vision", None

    if model_config is not None:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        model_type = model_config.get("model_type")
        if not model_type:
            raise ValueError("model_config.model_type is required")
        cfg = CONFIG_MAPPING[model_type].from_dict(dict(model_config))
    else:
        cfg = transformers.AutoConfig.from_pretrained(model_name)

    cfg.torch_dtype = torch.float32

    if workload_type is not None:
        if workload_type not in WORKLOAD_MODEL_CLASS:
            raise ValueError(
                f"지원하지 않는 workload_type입니다: {workload_type!r} "
                f"(가능: {sorted(WORKLOAD_MODEL_CLASS)})"
            )
        class_name, kind = WORKLOAD_MODEL_CLASS[workload_type]
        model_class = getattr(transformers, class_name)
    else:
        from utils.huggingface import suggest_automodel_class

        model_class = suggest_automodel_class(model_name)
        kind = "llm"

    model = model_class.from_config(cfg)
    model.train()          # xMem ModelPreparer와 동일 (추론 루프가 eval로 전환)
    return model, kind, cfg


def getattr_all_models(model_name):
    from perf_estimator.models import AllModels

    return getattr(AllModels, model_name, None)


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


# ───────────────────────────────────────── 경고
def collect_warnings(kind, mode):
    w = []
    if mode == "generate":
        w.append("mode=generate is EXPERIMENTAL: the KV cache is estimated "
                 "analytically rather than traced, and tends to overestimate")
    return w


# ───────────────────────────────────────── 메인
def run(args):
    t0 = time.time()
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)

    workload_type = getattr(args, "workload_type", None)
    model, kind, hf_cfg = load_model(
        args.model,
        getattr(args, "model_config", None),
        workload_type,
    )
    pb = param_bytes(model)

    resolution = args.resolution
    if workload_type == "text_encdec":
        # decoder_start_token_id는 0(T5의 실제 값)일 수 있어서 `or` 체인으로
        # 이어붙이면 falsy로 오인해 다음 항목으로 넘어간다 — None 체크로만 판단.
        decoder_start = getattr(hf_cfg, "decoder_start_token_id", None)
        if decoder_start is None:
            decoder_start = getattr(hf_cfg, "pad_token_id", None)
        if decoder_start is None:
            decoder_start = 0
        loader = build_encdec_loader(
            args.batch_size, args.encoder_seq_len, args.decoder_seq_len,
            vocab_size=getattr(hf_cfg, "vocab_size", 30000),
            decoder_start_token_id=decoder_start,
        )
    elif kind == "llm":
        loader = build_loader("llm", args.batch_size, seq_len=args.seq_len,
                              vocab_size=getattr(hf_cfg, "vocab_size", 30000))
    else:
        # 해상도/채널은 모델 config가 있으면 거기서 가져온다. ViT 계열은
        # position embedding이 학습 해상도에 묶여 있어서 다른 값을 주면 실제로
        # 실패하고, ConvNeXt/EfficientNet처럼 image_size를 선언해도 다른 해상도로
        # 도는 계열이 있어서 여기서 막지는 않고 경고만 남긴다.
        declared_res = getattr(hf_cfg, "image_size", None) if hf_cfg is not None else None
        channels = getattr(hf_cfg, "num_channels", None) if hf_cfg is not None else None
        if resolution is None:
            resolution = declared_res or 224
        loader = build_loader("vision", args.batch_size, resolution=resolution,
                              channels=channels or 3)

    warns = collect_warnings(kind, args.mode)
    if kind == "vision" and hf_cfg is not None:
        declared_res = getattr(hf_cfg, "image_size", None)
        if declared_res is not None and resolution != declared_res:
            warns.append(
                f"resolution {resolution} differs from the model's declared "
                f"image_size {declared_res}; models with fixed position embeddings "
                "(ViT/DeiT/Swin/...) will fail outright at a different size"
            )

    os.environ["XMEM_INFER"] = "1"
    os.environ["XMEM_PARAM_BYTES"] = str(pb)      # CNN 경로에서만 사용됨

    # kind=="llm"이면 estimate()가 TrainerEstimator를 쓰는데, 그건 xMem 자체
    # 코드에서 model_memory()를 계산하려고 AutoConfig.from_pretrained()와
    # suggest_automodel_class()(pipeline_tag 조회)를 *또* 부른다 — 우리가 이미
    # load_model()에서 받아둔 것과 같은 정보를 다시 HF에서 가져오는 것. 이걸
    # 건너뛰게 하려고 core_inference.patch에 이 두 env var를 읽는 hunk를 추가
    # 해뒀다. 원본 model_config가 없거나(레거시 CLI) workload_type을 안 줬으면
    # 그냥 비워둬서 xMem 원래 동작(직접 fetch)으로 자연히 떨어진다.
    raw_model_config = getattr(args, "model_config", None)
    if kind == "llm" and raw_model_config is not None:
        os.environ["XMEM_MODEL_CONFIG_JSON"] = json.dumps(raw_model_config)
    else:
        os.environ.pop("XMEM_MODEL_CONFIG_JSON", None)
    if kind == "llm" and workload_type is not None:
        os.environ["XMEM_MODEL_CLASS"] = WORKLOAD_MODEL_CLASS[workload_type][0]
    else:
        os.environ.pop("XMEM_MODEL_CLASS", None)

    trace = profile_on_cpu(model, loader, is_transformer=(kind == "llm"))
    raw = estimate(trace, args.batch_size, kind, args.model, args.gpu_memory_gb)

    allocated_fp32 = raw["memory"]["tensor"]
    scaled = allocated_fp32 * DTYPE_SCALE[args.dtype]
    corrected = scaled + UNDERESTIMATE_CONSTANT * DTYPE_SCALE[args.dtype]

    kv = None
    if args.mode == "generate" and kind == "llm" and hf_cfg is not None:
        # text_encdec generates past decoder_seq_len, not seq_len (that arg
        # isn't even meaningful for this workload_type).
        base_len = args.decoder_seq_len if workload_type == "text_encdec" else args.seq_len
        kv = kv_cache_bytes(hf_cfg, args.batch_size,
                            base_len + args.max_new_tokens, args.dtype)
        if kv is None:
            warns.append("KV cache could not be computed from the model config; "
                         "the result covers prefill only")

    required = corrected + (kv or 0) + MARGIN_BYTES

    return {
        "status": "ok",
        "input": {
            "model": args.model, "type": kind, "batch_size": args.batch_size,
            "seq_len": args.seq_len if kind == "llm" and workload_type != "text_encdec" else None,
            "encoder_seq_len": args.encoder_seq_len if workload_type == "text_encdec" else None,
            "decoder_seq_len": args.decoder_seq_len if workload_type == "text_encdec" else None,
            "resolution": resolution if kind == "vision" else None,
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
    p.add_argument("--json", help="모든 인자를 JSON 문자열로 전달. '-'이면 stdin에서 읽는다 "
                                  "(model_config가 들어가면 수십 KB라 명령줄 길이 제한에 걸린다)")
    p.add_argument("--model")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--encoder_seq_len", type=int, default=512)
    p.add_argument("--decoder_seq_len", type=int, default=128)
    # default None: vision 경로에서 모델 config의 image_size를 기본값으로 쓰고,
    # 그것도 없으면 run()이 224로 떨어뜨린다.
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--workload_type", choices=sorted(WORKLOAD_MODEL_CLASS), default=None)
    p.add_argument("--dtype", choices=list(DTYPE_SCALE), default="fp16")
    p.add_argument("--mode", choices=["prefill", "generate"], default="prefill")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--gpu_memory_gb", type=float, default=80)
    a = p.parse_args()

    if a.json:
        raw = sys.stdin.read() if a.json == "-" else a.json
        for k, v in json.loads(raw).items():
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
