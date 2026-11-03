export type WorkloadType =
  | "decoder_llm"
  | "vision_cls"
  | "text_encdec"
  | "audio_encdec"
  | "unsupported";

export type Dtype = "fp32" | "fp16" | "bf16";
export type InferMode = "prefill" | "generate";

export interface HfSearchResult {
  id: string;
  downloads: number | null;
  likes: number | null;
  pipeline_tag: string | null;
  tags: string[];
}

export interface ModelConfigFields {
  model_type: string | null;
  architectures: string[] | null;
  image_size: number | null;
  num_mel_bins: number | null;
  num_attention_heads: number | null;
  num_key_value_heads: number | null;
  num_hidden_layers: number | null;
  hidden_size: number | null;
  max_position_embeddings: number | null;
  is_encoder_decoder: boolean | null;
}

export interface ModelTypeResponse {
  type: WorkloadType;
  modelId: string;
  /** The Hub's own task classification (same source xMem's suggest_automodel_class()
   * uses) — the primary classification signal. null means the Hub has none on file. */
  pipeline_tag: string | null;
  /** Present only when type is "unsupported" or "audio_encdec" — why it was rejected. */
  reason?: string;
  config: ModelConfigFields;
  /** The full config.json as fetched, kept so a later step can hand it to
   * xMem/eager_runtime instead of them re-fetching it themselves. Unused for now. */
  raw_config: Record<string, unknown>;
}

/** Shape returned by GET /api/hf/model-type on failure. */
export interface ModelTypeErrorResponse {
  error: "not_found" | "gated" | "network_error" | "invalid_config" | "hf_error" | "missing_id";
}

export type ParamValue = number | string | null;

export type BudgetPeriod = "hourly" | "monthly";

/** Raw Step 2 "배포 조건" form input, before hourly normalization. */
export interface DeploymentInput {
  aws_region: string;
  budget_amount: number;
  budget_period: BudgetPeriod;
}

/** What actually goes into the payload — always normalized to an hourly rate. */
export interface DeploymentNormalized {
  aws_region: string;
  budget_usd_per_hour: number;
}

export interface PredictPayload {
  model: string;
  /** Full config.json as fetched when the model was picked — carried through
   * so xMem/eager_runtime can reuse it instead of fetching it again themselves. */
  model_config_snapshot: Record<string, unknown>;
  type: Exclude<WorkloadType, "unsupported">;
  dtype: Dtype;
  batch_size: number;
  params: Record<string, ParamValue>;
  deployment: DeploymentNormalized;
  warnings: string[];
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface XMemBreakdownBytes {
  predicted_allocated_native_dtype: number;
  predicted_reserved_native_dtype: number;
  empirical_tensor_correction: number;
  outside_gpu_estimate: number | null;
  fixed_margin: number;
}

export interface GpuMemoryRequirement {
  status: "ok" | "unsupported_dtype" | "uncalibrated_gpu";
  gpu: string;
  estimated_nvml_bytes?: number;
  required_vram_bytes?: number;
  outside_estimate_bytes?: number;
  outside_observed_upper_bytes?: number;
  calibration_id?: string;
}

export interface XMemResult {
  prediction_version: string;
  predicted_allocated_bytes: number;
  predicted_reserved_bytes: number;
  required_vram_scope: "target_process" | "maximum_among_calibrated_gpus";
  gpu_requirements: Record<string, GpuMemoryRequirement>;
  required_vram_gb: number;
  required_vram_bytes: number;
  breakdown_bytes: XMemBreakdownBytes;
  warnings: string[];
  elapsed_seconds: number;
}

/** instance_filter.filter_instances()'s output shape — one AWS single-GPU
 * G-series candidate that passed VRAM/region/budget/installed-target checks. */
export interface InstanceCandidate {
  instance_type: string;
  gpu_name: string;
  gpu_count: number;
  gpu_vram_gb: number;
  /** null means "available, price not in the catalog yet" — never a
   * stand-in for zero or unaffordable. */
  price_usd_per_hour: number | null;
  aws_region: string;
}

/** One target's result from eager_runtime.predict_runtime_targets(). */
export interface RuntimePrediction {
  status: "ok" | "error";
  model?: string;
  model_type?: string;
  workload_type?: string;
  target_instance: string;
  target_gpu?: string;
  engine_mode?: string;
  dtype?: string;
  batch_size?: number;
  params?: Record<string, ParamValue>;
  predicted_median_ms?: number;
  predicted_p90_ms?: number;
  /** Naive stub: batch_size / (predicted_median_ms / 1000), assuming
   * back-to-back single-stream batches with no queuing or concurrency.
   * An upper bound, not a measured number -- see predict_eager_runtime.py. */
  predicted_throughput_rps?: number;
  warnings?: string[];
  breakdown?: Record<string, unknown>;
  error_type?: string;
  error?: string;
}

/**
 * "ok"/"partial"/"error" come straight from eager_runtime.predict_runtime_targets().
 * "skipped" = runtime_adapter refused the request outright (bad dtype/mode/type)
 *   before ever calling eager_runtime.
 * "no_installed_targets" = every candidate target lacked an installed eager_runtime profile.
 * "no_candidates" = instance_filter found zero AWS candidates — a normal filter
 *   result (e.g. no catalog data yet, nothing fit budget/region), not an error.
 */
export type RuntimeResultStatus =
  | "ok"
  | "partial"
  | "error"
  | "skipped"
  | "no_installed_targets"
  | "no_candidates";

export interface RuntimeTargetsResult {
  status: RuntimeResultStatus;
  error_code?: string;
  error_message?: string;
  model?: string;
  requested_targets: string[];
  not_installed_targets?: string[];
  succeeded: number;
  failed: number;
  predictions: RuntimePrediction[];
}

/** orchestrator/main.py's process_job() result once Runtime prediction is wired in. */
export interface CombinedPredictionResult {
  vram: XMemResult;
  candidates: InstanceCandidate[];
  /** Set only when the live AWS price fetch itself failed (network/parse) --
   * distinct from a normal empty `candidates` (nothing fit, or this region
   * genuinely doesn't sell any of the 4 target instances). */
  catalog_error: string | null;
  runtime: RuntimeTargetsResult;
}

export type JobErrorCode = "oom" | "timeout" | "model_load_failed" | "unsupported_type" | "internal";

export interface JobError {
  code: JobErrorCode;
  message: string;
}

/** The user's acceptance bar (PATCH /jobs/{id}/slo) — never a workload input,
 * never a reason to recompute. null until submitted. */
export interface SloInput {
  max_latency_ms: number;
  min_throughput_rps: number;
}

export interface JobResponse {
  job_id: string;
  status: JobStatus;
  cached: boolean;
  progress: string | null;
  payload: PredictPayload;
  slo: SloInput | null;
  result: CombinedPredictionResult | null;
  error: JobError | null;
}

export interface CreateJobResponse {
  job_id: string;
  status: JobStatus;
  cached: boolean;
  error_code?: string;
  /** Only on error_code="invalid_payload" — which field failed and why. */
  message?: string;
}
