import type { JobError } from "./types";

export function errorMessageFor(error: JobError): string {
  switch (error.code) {
    case "oom":
      return "모델이 너무 커서 CPU 메모리가 부족합니다. 더 작은 모델로 시도하세요.";
    case "timeout":
      // The backend's own message already explains the (model size / batch /
      // seq_len, not network) reason and the current limit — show it as-is
      // instead of a second, possibly stale, hardcoded copy of the same info.
      return error.message || "CPU 프로파일링이 제한 시간을 초과했습니다.";
    case "model_load_failed":
      return `모델을 불러오지 못했습니다: ${error.message}`;
    case "unsupported_type":
      return "현재 Decoder-only LLM 유형만 예측을 지원합니다.";
    case "internal":
    default:
      return "예측 중 오류가 발생했습니다.";
  }
}

export const BREAKDOWN_LABELS: Record<string, string> = {
  param_bytes_fp32: "파라미터 메모리 (FP32)",
  predicted_allocated_fp32: "예측 할당량 (FP32)",
  predicted_allocated_scaled: "예측 텐서 메모리",
  after_underestimate_correction: "과소추정 보정 후",
  kv_cache_estimate: "KV 캐시 추정",
  fixed_margin: "고정 마진 (CUDA 컨텍스트 + allocator 여유)",
};
