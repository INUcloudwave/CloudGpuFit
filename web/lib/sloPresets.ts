import type { LatencyPreset, PredictPayload, SloInput } from "@/lib/types";

/**
 * Latency SLO presets and how an SLO is described on screen. The numbers must
 * match orchestrator/recommendation.py (LATENCY_PRESETS_MS_PER_TOKEN,
 * TARGET_UTILIZATION, TARGET_UTILIZATION_MAX). TARGET_UTILIZATION is shown as
 * "목표 부하율" (demand / throughput), not GPU utilization.
 *
 * Why per output token for Generate: the predictor's latency is the whole
 * request (prefill + every generated token), so a request limit only makes
 * sense relative to how many tokens are generated. LLM-serving work states
 * interactive SLOs as time per output token (TPOT); DistServe (OSDI'24) uses
 * ~100 ms/token for chat ("match reading speed") and ~50 ms/token for
 * latency-critical apps. 100 ms/token = 10 tokens/s, about twice an average
 * reader (~5-6 tokens/s); 50 ms/token = 20 tokens/s.
 */

export const DEFAULT_TARGET_UTILIZATION = 0.8;
export const MAX_TARGET_UTILIZATION = 0.9;
export const MIN_TARGET_UTILIZATION = 0.5;

export interface LatencyPresetSpec {
  id: Exclude<LatencyPreset, "custom" | "none">;
  label: string;
  msPerToken: number;
  description: string;
}

export const GENERATE_LATENCY_PRESETS: LatencyPresetSpec[] = [
  {
    id: "realtime",
    label: "실시간",
    msPerToken: 50,
    description: "토큰당 50ms (초당 20토큰). 코딩 보조·검색처럼 빠른 응답이 필요한 서비스.",
  },
  {
    id: "standard",
    label: "일반 대화",
    msPerToken: 100,
    description: "토큰당 100ms (초당 10토큰). 사람이 읽는 속도(초당 약 5~6토큰)보다 빠른 챗봇 수준.",
  },
];

/** Generated tokens per request, or null for workloads that don't generate. */
export function generatedTokens(payload: PredictPayload): number | null {
  const v = payload.params.max_new_tokens;
  return typeof v === "number" && v > 0 ? v : null;
}

const PRESET_LABEL: Record<LatencyPreset, string> = {
  realtime: "실시간",
  standard: "일반 대화",
  custom: "직접 입력",
  none: "제한 없음",
};

function fmt(n: number): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: 1 });
}

/** One line for the latency part of an SLO, e.g.
 * "실시간 · 토큰당 50ms × 256토큰 = 요청당 12,800ms 이내". */
export function describeLatency(slo: SloInput, payload: PredictPayload): string {
  const tokens = generatedTokens(payload);
  const label = slo.latency_preset ? `${PRESET_LABEL[slo.latency_preset]} · ` : "";
  if (slo.latency_per_token_ms) {
    if (tokens) {
      return `${label}토큰당 ${fmt(slo.latency_per_token_ms)}ms × ${tokens}토큰 = 요청당 ${fmt(
        slo.latency_per_token_ms * tokens
      )}ms 이내`;
    }
    return `${label}요청당 ${fmt(slo.latency_per_token_ms)}ms 이내`;
  }
  if (slo.max_latency_ms) return `${label}요청당 ${fmt(slo.max_latency_ms)}ms 이내`;
  return "지연 제한 없음";
}

/** One line for the workload part of an SLO: the demand (online) or the
 * amount of work (offline). */
export function describeWorkload(slo: SloInput): string {
  if (slo.purpose === "offline") {
    return slo.total_requests
      ? `오프라인 일괄 처리 · 요청 ${slo.total_requests.toLocaleString()}건`
      : "오프라인 일괄 처리";
  }
  return `온라인 서빙 · 수요 ${slo.min_throughput_rps ?? "—"} req/s`;
}
