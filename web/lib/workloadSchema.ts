import type {
  DeploymentInput,
  DeploymentNormalized,
  Dtype,
  ModelConfigFields,
  ParamValue,
  PredictPayload,
  WorkloadType,
} from "./types";

export interface NumberFieldSpec {
  key: string;
  label: string;
  kind: "number";
  default: number;
  hardMin?: number;
  showWhen?: (values: Record<string, unknown>) => boolean;
  helperText?: string;
}

export interface RadioFieldSpec {
  key: string;
  label: string;
  kind: "radio";
  default: string;
  options: { value: string; label: string }[];
  showWhen?: (values: Record<string, unknown>) => boolean;
}

export type FieldSpec = NumberFieldSpec | RadioFieldSpec;

export interface WorkloadSchema {
  type: Exclude<WorkloadType, "unsupported">;
  label: string;
  fields: FieldSpec[];
}

export const DTYPE_FIELD: RadioFieldSpec = {
  key: "dtype",
  label: "dtype",
  kind: "radio",
  default: "fp16",
  options: [
    { value: "fp32", label: "fp32" },
    { value: "fp16", label: "fp16" },
    { value: "bf16", label: "bf16" },
  ],
};

// Generative workloads are always predicted end to end (prefill + every
// generated token) -- there is no prefill-only mode to pick in the form.
const MAX_NEW_TOKENS_FIELD = (defaultMaxNewTokens: number): FieldSpec => ({
  key: "max_new_tokens",
  label: "max_new_tokens",
  kind: "number",
  default: defaultMaxNewTokens,
  hardMin: 1,
});

// batch_size is the LARGEST batch the service may run, not a fixed one: VRAM
// is predicted at it, and the recommendation (orchestrator/recommendation.py)
// picks each instance's operating batch from 1..this value. A larger cap only
// opens more options (the VRAM requirement grows with it).
const BATCH_FIELD = (defaultBatch: number): FieldSpec => ({
  key: "batch_size",
  label: "최대 배치 (batch_size)",
  kind: "number",
  default: defaultBatch,
  hardMin: 1,
  helperText: "한 번에 묶어 처리할 수 있는 최대 요청 수. 추천 시 인스턴스별로 이 값 이하에서 운영 배치를 고릅니다.",
});

export const WORKLOAD_SCHEMAS: Record<Exclude<WorkloadType, "unsupported">, WorkloadSchema> = {
  decoder_llm: {
    type: "decoder_llm",
    label: "Decoder-only LLM",
    fields: [
      BATCH_FIELD(32),
      { key: "seq_len", label: "seq_len", kind: "number", default: 512, hardMin: 1 },
      MAX_NEW_TOKENS_FIELD(256),
    ],
  },
  vision_cls: {
    type: "vision_cls",
    label: "Vision classification",
    fields: [
      BATCH_FIELD(32),
      { key: "resolution", label: "resolution", kind: "number", default: 224, hardMin: 32 },
    ],
  },
  text_encdec: {
    type: "text_encdec",
    label: "Text Encoder-Decoder",
    fields: [
      BATCH_FIELD(32),
      { key: "encoder_seq_len", label: "encoder_seq_len", kind: "number", default: 512, hardMin: 1 },
      { key: "decoder_seq_len", label: "decoder_seq_len", kind: "number", default: 128, hardMin: 1 },
      MAX_NEW_TOKENS_FIELD(128),
    ],
  },
  audio_encdec: {
    type: "audio_encdec",
    label: "Audio Encoder-Decoder",
    fields: [
      BATCH_FIELD(1),
      {
        key: "audio_chunks",
        label: "audio_chunks",
        kind: "number",
        default: 1,
        hardMin: 1,
        helperText: "30초 단위 (예: 3 = 90초 오디오)",
      },
      { key: "decoder_seq_len", label: "decoder_seq_len", kind: "number", default: 128, hardMin: 1 },
      MAX_NEW_TOKENS_FIELD(224),
    ],
  },
};

/** model_type substrings whose position embeddings are fixed to the training resolution. */
const VIT_LIKE_MODEL_TYPES = ["vit", "deit", "beit", "swin", "dinov2"];

export function isResolutionLocked(config: ModelConfigFields): boolean {
  if (config.image_size == null || !config.model_type) return false;
  const modelType = config.model_type.toLowerCase();
  return VIT_LIKE_MODEL_TYPES.some((t) => modelType.includes(t));
}

export function defaultResolution(config: ModelConfigFields): number {
  return config.image_size ?? 224;
}

export function defaultValuesFor(
  type: Exclude<WorkloadType, "unsupported">,
  config: ModelConfigFields
): Record<string, unknown> {
  const schema = WORKLOAD_SCHEMAS[type];
  const values: Record<string, unknown> = {};
  for (const field of schema.fields) {
    values[field.key] = field.default;
  }
  if (type === "vision_cls") {
    values.resolution = defaultResolution(config);
  }
  return values;
}

export interface ScenarioPreset {
  id: string;
  label: string;
  dtype: Dtype;
  /** Only the keys this scenario actually sets — merged over defaultValuesFor()'s
   * output, so e.g. vision_cls's resolution still comes from the model config,
   * never from the preset (a bigger "task size" must never imply a resolution
   * the model doesn't actually support). */
  values: Record<string, unknown>;
}

// Preset numbers, not measured results: convenience starting points for
// "small/medium/large"-shaped tasks, not validated against the prediction
// engine's accuracy or a model's actual context limit. Shown to the user as
// such, and re-derived from real evidence later.
export const SCENARIO_PRESETS: Record<Exclude<WorkloadType, "unsupported">, ScenarioPreset[]> = {
  decoder_llm: [
    {
      id: "short_chat",
      label: "짧은 대화 응답",
      dtype: "fp16",
      values: { batch_size: 1, seq_len: 128, max_new_tokens: 64 },
    },
    {
      id: "doc_summary",
      label: "문서 요약 요청",
      dtype: "fp16",
      values: { batch_size: 4, seq_len: 512, max_new_tokens: 256 },
    },
    {
      id: "batch_processing",
      label: "대량 배치 처리",
      dtype: "fp16",
      values: { batch_size: 8, seq_len: 1024, max_new_tokens: 256 },
    },
  ],
  vision_cls: [
    { id: "small_batch", label: "소량 이미지 분류", dtype: "fp16", values: { batch_size: 1 } },
    { id: "medium_batch", label: "보통 배치", dtype: "fp16", values: { batch_size: 8 } },
    { id: "large_batch", label: "대량 배치", dtype: "fp16", values: { batch_size: 32 } },
  ],
  text_encdec: [
    {
      id: "short_translation",
      label: "짧은 번역",
      dtype: "fp16",
      values: { batch_size: 1, encoder_seq_len: 64, decoder_seq_len: 64, max_new_tokens: 64 },
    },
    {
      id: "medium_summary",
      label: "중간 길이 문서 요약",
      dtype: "fp16",
      values: { batch_size: 1, encoder_seq_len: 512, decoder_seq_len: 128, max_new_tokens: 128 },
    },
    {
      id: "long_transform",
      label: "긴 문서 변환",
      dtype: "fp16",
      values: { batch_size: 1, encoder_seq_len: 1024, decoder_seq_len: 256, max_new_tokens: 256 },
    },
  ],
  audio_encdec: [],
};

export function applyScenario(
  preset: ScenarioPreset,
  type: Exclude<WorkloadType, "unsupported">,
  config: ModelConfigFields
): { dtype: Dtype; values: Record<string, unknown> } {
  return { dtype: preset.dtype, values: { ...defaultValuesFor(type, config), ...preset.values } };
}

/** Short "batch=… · 입력 …" line for a scenario card — same values a click
 * would apply, so nothing on the card is a surprise once picked. */
export function describeScenario(type: Exclude<WorkloadType, "unsupported">, preset: ScenarioPreset): string {
  const v = preset.values;
  const parts = [`batch=${v.batch_size}`];
  if (type === "decoder_llm") {
    parts.push(`입력 ${v.seq_len} 토큰`);
    parts.push(`생성 ${v.max_new_tokens} 토큰`);
  } else if (type === "text_encdec") {
    parts.push(`인코더 ${v.encoder_seq_len}`, `디코더 ${v.decoder_seq_len}`);
    parts.push(`생성 ${v.max_new_tokens} 토큰`);
  }
  parts.push(preset.dtype);
  return parts.join(" · ");
}

/**
 * Mirrors orchestrator/workload_params.py. The server re-checks all of this —
 * this exists so the user sees the problem next to the field instead of as a
 * failed request.
 */
export function errorForField(field: FieldSpec, values: Record<string, unknown>): string | null {
  if (field.kind !== "number") return null;
  if (field.showWhen && !field.showWhen(values)) return null;

  const raw = values[field.key];
  if (raw === "" || raw === null || raw === undefined) return "값을 입력하세요.";

  const num = Number(raw);
  if (!Number.isFinite(num)) return "숫자를 입력하세요.";
  if (!Number.isInteger(num)) return "정수를 입력하세요.";
  if (num <= 0) return "0보다 커야 합니다.";
  if (field.hardMin !== undefined && num < field.hardMin) return `${field.hardMin} 이상이어야 합니다.`;
  return null;
}

export function collectFieldErrors(
  type: Exclude<WorkloadType, "unsupported">,
  values: Record<string, unknown>
): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const field of WORKLOAD_SCHEMAS[type].fields) {
    const error = errorForField(field, values);
    if (error) errors[field.key] = error;
  }
  return errors;
}

/** Monthly-to-hourly conversion basis: 24h × 30 days. */
export const BUDGET_HOURS_PER_MONTH = 720;

export function normalizeDeployment(input: DeploymentInput): DeploymentNormalized {
  const budgetUsdPerHour =
    input.budget_period === "monthly" ? input.budget_amount / BUDGET_HOURS_PER_MONTH : input.budget_amount;
  return { aws_region: input.aws_region, budget_usd_per_hour: budgetUsdPerHour };
}

export function buildPayload(
  modelId: string,
  modelConfigSnapshot: Record<string, unknown>,
  type: Exclude<WorkloadType, "unsupported">,
  dtype: Dtype,
  values: Record<string, unknown>,
  deployment: DeploymentInput
): PredictPayload {
  const schema = WORKLOAD_SCHEMAS[type];
  const params: Record<string, ParamValue> = {};

  for (const field of schema.fields) {
    if (field.key === "batch_size") continue;
    const visible = !field.showWhen || field.showWhen(values);
    if (!visible) {
      params[field.key] = null;
      continue;
    }
    params[field.key] = field.kind === "number" ? Number(values[field.key]) : (values[field.key] as string);
  }
  if ("max_new_tokens" in params) params.mode = "generate";

  return {
    model: modelId,
    model_config_snapshot: modelConfigSnapshot,
    type,
    dtype,
    batch_size: Number(values.batch_size),
    params,
    deployment: normalizeDeployment(deployment),
    warnings: [],
  };
}
