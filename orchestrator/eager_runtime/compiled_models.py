from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BertConfig,
    BertModel,
    OPTConfig,
    Qwen2Config,
    RobertaConfig,
    RobertaModel,
    T5Config,
    Wav2Vec2Config,
    Wav2Vec2Model,
)

from first_token_models import LastTokenLogits, restrict_causal_lm_to_last_token


TEXT_MODELS = {"bert", "roberta", "opt", "t5", "qwen2"}
DECODER_MODELS = {"opt", "t5", "qwen2"}
VISION_MODELS = {"resnet18", "mobilenet_v2"}
AUDIO_MODELS = {"wav2vec2"}
MODEL_NAMES = tuple(
    sorted(TEXT_MODELS | VISION_MODELS | AUDIO_MODELS)
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    measurement_scope: str
    model_id: str
    decoder_sequence_length: int | None = None


MODEL_SPECS = {
    "resnet18": ModelSpec(
        "resnet18", "vision_cls", "classification_logits", "torchvision/resnet18"
    ),
    "mobilenet_v2": ModelSpec(
        "mobilenet_v2",
        "vision_cls",
        "classification_logits",
        "torchvision/mobilenet_v2",
    ),
    "bert": ModelSpec(
        "bert", "text_encoder", "encoder_hidden_state", "google-bert/bert-base-uncased"
    ),
    "roberta": ModelSpec(
        "roberta", "text_encoder", "encoder_hidden_state", "FacebookAI/roberta-base"
    ),
    "opt": ModelSpec(
        "opt", "decoder_llm", "first_token_logits", "facebook/opt-125m"
    ),
    "t5": ModelSpec(
        "t5",
        "text_encdec",
        "first_token_logits",
        "google-t5/t5-small",
        decoder_sequence_length=1,
    ),
    "wav2vec2": ModelSpec(
        "wav2vec2",
        "audio_encoder",
        "encoder_hidden_state",
        "facebook/wav2vec2-base",
    ),
    "qwen2": ModelSpec(
        "qwen2", "decoder_llm", "first_token_logits", "Qwen/Qwen2-1.5B"
    ),
}


class HiddenStateTensor(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, **inputs):
        output = self.model(**inputs, return_dict=True)
        return output.last_hidden_state


def build_transformer_config(model_name: str):
    if model_name == "bert":
        return BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
        )
    if model_name == "roberta":
        return RobertaConfig(
            vocab_size=50265,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=514,
        )
    if model_name == "opt":
        return OPTConfig(
            vocab_size=50272,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            ffn_dim=3072,
            max_position_embeddings=2048,
            use_cache=False,
            architectures=["OPTForCausalLM"],
        )
    if model_name == "t5":
        return T5Config(
            vocab_size=32128,
            d_model=512,
            d_kv=64,
            d_ff=2048,
            num_layers=6,
            num_decoder_layers=6,
            num_heads=8,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
            use_cache=False,
            architectures=["T5ForConditionalGeneration"],
        )
    if model_name == "qwen2":
        return Qwen2Config(
            vocab_size=151936,
            hidden_size=1536,
            intermediate_size=8960,
            num_hidden_layers=28,
            num_attention_heads=12,
            num_key_value_heads=2,
            max_position_embeddings=32768,
            use_cache=False,
            architectures=["Qwen2ForCausalLM"],
        )
    if model_name == "wav2vec2":
        return Wav2Vec2Config(
            vocab_size=32,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=6,
            num_attention_heads=8,
            conv_dim=(512, 512, 512, 512, 512, 512, 512),
            conv_stride=(5, 2, 2, 2, 2, 2, 2),
            conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        )
    raise ValueError(f"No transformer config for model: {model_name}")


def _allocate_without_pretrained_weights(factory, device: str):
    with torch.device("meta"):
        model = factory().half()
    model = model.to_empty(device=device).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.01)
        for name, buffer in model.named_buffers():
            if buffer.is_floating_point():
                if name.endswith("running_var"):
                    buffer.fill_(1.0)
                else:
                    buffer.zero_()
    return model


def build_model(model_name: str, device: str = "cuda") -> nn.Module:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name == "resnet18":
        return _allocate_without_pretrained_weights(
            lambda: models.resnet18(weights=None), device
        )
    if model_name == "mobilenet_v2":
        return _allocate_without_pretrained_weights(
            lambda: models.mobilenet_v2(weights=None), device
        )

    config = build_transformer_config(model_name)
    config.torch_dtype = torch.float16
    if model_name == "bert":
        model = _allocate_without_pretrained_weights(
            lambda: BertModel(config, add_pooling_layer=False), device
        )
        return HiddenStateTensor(model).eval()
    if model_name == "roberta":
        model = _allocate_without_pretrained_weights(
            lambda: RobertaModel(config, add_pooling_layer=False), device
        )
        return HiddenStateTensor(model).eval()
    if model_name == "wav2vec2":
        model = _allocate_without_pretrained_weights(lambda: Wav2Vec2Model(config), device)
        return HiddenStateTensor(model).eval()

    model_class = AutoModelForSeq2SeqLM if model_name == "t5" else AutoModelForCausalLM
    model = _allocate_without_pretrained_weights(
        lambda: model_class.from_config(config, trust_remote_code=False),
        device,
    )
    if model_name != "t5":
        model = restrict_causal_lm_to_last_token(model)
    return LastTokenLogits(model).eval()


def remove_unused_text_pooler(model: nn.Module) -> bool:
    """Remove an auxiliary pooler when the declared output is hidden states.

    BERT-family poolers are independent of ``last_hidden_state``.  The real
    compiled wrapper returns only that tensor, so Inductor dead-code eliminates
    the pooler.  FX tracing the raw HF model needs the same output boundary.
    """

    candidates = [model]
    prefix = getattr(model, "base_model_prefix", "")
    if prefix:
        base = getattr(model, str(prefix), None)
        if isinstance(base, nn.Module) and base is not model:
            candidates.append(base)
    for candidate in candidates:
        pooler = getattr(candidate, "pooler", None)
        if isinstance(pooler, nn.Module):
            candidate.pooler = None
            return True
    return False


def make_cpu_inputs(
    model_name: str,
    batch_size: int,
    sequence_length: int,
    resolution: int = 224,
    audio_samples: int = 32000,
):
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    if model_name in VISION_MODELS:
        return torch.zeros(
            batch_size,
            3,
            resolution,
            resolution,
            dtype=torch.float16,
        )
    if model_name in AUDIO_MODELS:
        return {
            "input_values": torch.zeros(
                batch_size,
                audio_samples,
                dtype=torch.float16,
            )
        }

    config = build_transformer_config(model_name)
    input_ids = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    if model_name != "t5":
        return {"input_ids": input_ids}

    return {
        "input_ids": input_ids,
        "decoder_input_ids": torch.full(
            (batch_size, 1),
            int(config.decoder_start_token_id or 0),
            dtype=torch.long,
        ),
    }
