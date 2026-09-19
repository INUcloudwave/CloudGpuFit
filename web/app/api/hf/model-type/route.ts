import { NextRequest, NextResponse } from "next/server";
import type { ModelConfigFields, ModelTypeResponse, WorkloadType } from "@/lib/types";

export const runtime = "nodejs";

type HfConfigJson = Record<string, unknown>;

// The Hub's own task classification (a fixed vocabulary — set by the uploader in the
// model card, or auto-inferred by the Hub from the architecture if they didn't). This
// is the SAME signal xMem's own suggest_automodel_class() uses to pick an AutoModel
// class, so classifying against it here instead of guessing from config.json fields
// keeps the web gate consistent with what the execution engine actually does.
// Everything else (audio, fill-mask, sentence-similarity, ...) is just
// "unsupported" — no separate audio_encdec bucket or special-cased message.
// README lists exactly these tags as supported; anything not here is out of
// scope for the same reason, not because audio specifically is worse off.
const PIPELINE_TAG_TYPE: Record<string, Exclude<WorkloadType, "unsupported">> = {
  "text-generation": "decoder_llm",
  "image-classification": "vision_cls",
  "text2text-generation": "text_encdec",
  translation: "text_encdec",
  summarization: "text_encdec",
};

function classify(pipelineTag: string | null): WorkloadType {
  if (!pipelineTag) return "unsupported";
  return PIPELINE_TAG_TYPE[pipelineTag] ?? "unsupported";
}

function rejectionReason(type: WorkloadType, pipelineTag: string | null): string | undefined {
  if (type !== "unsupported") return undefined;
  if (!pipelineTag) {
    return "이 모델은 Hugging Face pipeline_tag 정보가 없어 유형을 판단할 수 없습니다.";
  }
  return `pipeline_tag "${pipelineTag}"는 지원하는 유형(decoder_llm / vision_cls / text_encdec)에 해당하지 않습니다.`;
}

function pickConfig(cfg: HfConfigJson): ModelConfigFields {
  const num = (v: unknown): number | null => (typeof v === "number" ? v : null);
  const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
  return {
    model_type: str(cfg.model_type),
    architectures: Array.isArray(cfg.architectures) ? (cfg.architectures as string[]) : null,
    image_size: num(cfg.image_size),
    num_mel_bins: num(cfg.num_mel_bins),
    num_attention_heads: num(cfg.num_attention_heads),
    num_key_value_heads: num(cfg.num_key_value_heads),
    num_hidden_layers: num(cfg.num_hidden_layers),
    hidden_size: num(cfg.hidden_size),
    max_position_embeddings: num(cfg.max_position_embeddings),
    is_encoder_decoder: typeof cfg.is_encoder_decoder === "boolean" ? cfg.is_encoder_decoder : null,
  };
}

export async function GET(req: NextRequest) {
  const id = req.nextUrl.searchParams.get("id")?.trim();
  if (!id) {
    return NextResponse.json({ error: "missing_id" }, { status: 400 });
  }

  const headers: Record<string, string> = {};
  if (process.env.HF_TOKEN) {
    headers.Authorization = `Bearer ${process.env.HF_TOKEN}`;
  }

  // pipeline_tag comes from the Hub's model-info API, not config.json.
  let infoRes: Response;
  try {
    infoRes = await fetch(`https://huggingface.co/api/models/${id}`, { headers });
  } catch {
    return NextResponse.json({ error: "network_error" }, { status: 502 });
  }

  if (infoRes.status === 401 || infoRes.status === 403) {
    return NextResponse.json({ error: "gated" }, { status: infoRes.status });
  }
  if (infoRes.status === 404) {
    return NextResponse.json({ error: "not_found" }, { status: 404 });
  }
  if (!infoRes.ok) {
    return NextResponse.json({ error: "hf_error" }, { status: infoRes.status });
  }

  let info: { pipeline_tag?: unknown };
  try {
    info = (await infoRes.json()) as { pipeline_tag?: unknown };
  } catch {
    return NextResponse.json({ error: "invalid_config" }, { status: 502 });
  }
  const pipelineTag = typeof info.pipeline_tag === "string" ? info.pipeline_tag : null;

  // config.json is still needed for two things that have nothing to do with
  // classification: a few UI-only defaults (isResolutionLocked/defaultResolution)
  // and the full raw object, kept as-is so a later step can hand it straight to
  // xMem/eager_runtime instead of them fetching it again themselves.
  let res: Response;
  try {
    res = await fetch(`https://huggingface.co/${id}/resolve/main/config.json`, { headers });
  } catch {
    return NextResponse.json({ error: "network_error" }, { status: 502 });
  }

  if (res.status === 401 || res.status === 403) {
    return NextResponse.json({ error: "gated" }, { status: res.status });
  }
  if (res.status === 404) {
    return NextResponse.json({ error: "not_found" }, { status: 404 });
  }
  if (!res.ok) {
    return NextResponse.json({ error: "hf_error" }, { status: res.status });
  }

  let cfg: HfConfigJson;
  try {
    cfg = (await res.json()) as HfConfigJson;
  } catch {
    return NextResponse.json({ error: "invalid_config" }, { status: 502 });
  }

  const type = classify(pipelineTag);

  const body: ModelTypeResponse = {
    type,
    modelId: id,
    pipeline_tag: pipelineTag,
    reason: rejectionReason(type, pipelineTag),
    config: pickConfig(cfg),
    raw_config: cfg,
  };
  return NextResponse.json(body);
}
