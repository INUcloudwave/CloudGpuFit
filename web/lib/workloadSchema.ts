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
  /** Soft range: values outside this still submit, but render a warning. */
  validatedRange?: [number, number];
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

const isGenerate = (values: Record<string, unknown>) => values.mode === "generate";

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

const MODE_FIELD = (defaultMaxNewTokens: number): FieldSpec[] => [
  {
    key: "mode",
    label: "mode",
    kind: "radio",
    default: "prefill",
    options: [
      { value: "prefill", label: "prefill" },
      { value: "generate", label: "generate" },
    ],
  },
  {
    key: "max_new_tokens",
    label: "max_new_tokens",
    kind: "number",
    default: defaultMaxNewTokens,
    hardMin: 1,
    showWhen: isGenerate,
  },
];

export const WORKLOAD_SCHEMAS: Record<Exclude<WorkloadType, "unsupported">, WorkloadSchema> = {
  decoder_llm: {
    type: "decoder_llm",
    label: "Decoder-only LLM",
    fields: [
      { key: "batch_size", label: "batch_size", kind: "number", default: 1, hardMin: 1, validatedRange: [1, 30] },
      { key: "seq_len", label: "seq_len", kind: "number", default: 512, hardMin: 1, validatedRange: [128, 2048] },
      ...MODE_FIELD(256),
    ],
  },
  vision_cls: {
    type: "vision_cls",
    label: "Vision classification",
    fields: [
      { key: "batch_size", label: "batch_size", kind: "number", default: 32, hardMin: 1, validatedRange: [32, 64] },
      { key: "resolution", label: "resolution", kind: "number", default: 224, hardMin: 32 },
    ],
  },
  text_encdec: {
    type: "text_encdec",
    label: "Text Encoder-Decoder",
    fields: [
      { key: "batch_size", label: "batch_size", kind: "number", default: 1, hardMin: 1 },
      { key: "encoder_seq_len", label: "encoder_seq_len", kind: "number", default: 512, hardMin: 1 },
      { key: "decoder_seq_len", label: "decoder_seq_len", kind: "number", default: 128, hardMin: 1 },
      ...MODE_FIELD(128),
    ],
  },
  audio_encdec: {
    type: "audio_encdec",
    label: "Audio Encoder-Decoder",
    fields: [
      { key: "batch_size", label: "batch_size", kind: "number", default: 1, hardMin: 1 },
      {
        key: "audio_chunks",
        label: "audio_chunks",
        kind: "number",
        default: 1,
        hardMin: 1,
        helperText: "30초 단위 (예: 3 = 90초 오디오)",
      },
      { key: "decoder_seq_len", label: "decoder_seq_len", kind: "number", default: 128, hardMin: 1 },
      ...MODE_FIELD(224),
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

export function warningForField(field: FieldSpec, values: Record<string, unknown>): string | null {
  if (field.kind !== "number" || !field.validatedRange) return null;
  if (field.showWhen && !field.showWhen(values)) return null;
  const num = Number(values[field.key]);
  if (!Number.isFinite(num)) return null;
  const [lo, hi] = field.validatedRange;
  if (num < lo || num > hi) {
    return `검증 범위(${lo}-${hi})를 벗어난 값입니다. 결과를 참고용으로만 사용하세요.`;
  }
  return null;
}

export const GENERATE_MODE_WARNING =
  "generate 모드의 KV cache 추정은 아직 GPU 실측 대비 검증되지 않았습니다. 과대추정 경향이 있습니다.";

export function collectWarnings(
  type: Exclude<WorkloadType, "unsupported">,
  values: Record<string, unknown>
): string[] {
  const schema = WORKLOAD_SCHEMAS[type];
  const warnings: string[] = [];

  for (const field of schema.fields) {
    const w = warningForField(field, values);
    if (w) warnings.push(`${field.label}: ${w}`);
  }

  if (values.mode === "generate") {
    warnings.push(GENERATE_MODE_WARNING);
  }

  return warnings;
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

  return {
    model: modelId,
    type,
    dtype,
    batch_size: Number(values.batch_size),
    params,
    deployment: normalizeDeployment(deployment),
    warnings: collectWarnings(type, values),
  };
}
