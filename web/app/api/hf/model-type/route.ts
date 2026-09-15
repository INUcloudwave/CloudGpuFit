import { NextRequest, NextResponse } from "next/server";
import type { ModelConfigFields, ModelTypeResponse, WorkloadType } from "@/lib/types";

export const runtime = "nodejs";

type HfConfigJson = Record<string, unknown>;

function classify(cfg: HfConfigJson): WorkloadType {
  const architectures = Array.isArray(cfg.architectures) ? (cfg.architectures as string[]) : [];
  const isEncoderDecoder = cfg.is_encoder_decoder === true;
  const hasMelBins = cfg.num_mel_bins !== undefined && cfg.num_mel_bins !== null;

  if (isEncoderDecoder && hasMelBins) return "audio_encdec";
  if (isEncoderDecoder) return "text_encdec";
  if (architectures.some((a) => a.endsWith("ForImageClassification"))) return "vision_cls";
  if (architectures.some((a) => a.endsWith("ForCausalLM"))) return "decoder_llm";
  return "unsupported";
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

  const body: ModelTypeResponse = {
    type: classify(cfg),
    modelId: id,
    config: pickConfig(cfg),
  };
  return NextResponse.json(body);
}
