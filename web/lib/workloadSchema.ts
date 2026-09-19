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
      { key: "batch_size", label: "batch_size", kind: "number", default: 1, hardMin: 1 },
      { key: "seq_len", label: "seq_len", kind: "number", default: 512, hardMin: 1 },
      ...MODE_FIELD(256),
    ],
  },
  vision_cls: {
    type: "vision_cls",
    label: "Vision classification",
    fields: [
      { key: "batch_size", label: "batch_size", kind: "number", default: 32, hardMin: 1 },
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

export const GENERATE_MODE_WARNING =
  "generate 모드의 KV cache 추정은 아직 GPU 실측 대비 검증되지 않았습니다. 과대추정 경향이 있습니다.";

export function collectWarnings(
  type: Exclude<WorkloadType, "unsupported">,
  values: Record<string, unknown>
): string[] {
  const warnings: string[] = [];

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

  return {
    model: modelId,
    model_config_snapshot: modelConfigSnapshot,
    type,
    dtype,
    batch_size: Number(values.batch_size),
    params,
    deployment: normalizeDeployment(deployment),
    warnings: collectWarnings(type, values),
  };
}
