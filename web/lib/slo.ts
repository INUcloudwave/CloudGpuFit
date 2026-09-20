import type { RuntimePrediction, SloInput } from "@/lib/types";

/**
 * "pass"/"fail" only when there's a real number on both sides of the
 * comparison. "unknown" covers every case where a verdict can't be formed --
 * no SLO submitted yet, or this target has no runtime prediction (not "ok",
 * or missing the field). Never conflated with "fail": an instance the
 * pipeline never got a number for isn't disqualified, it's un-evaluated.
 */
export type SloVerdict = "pass" | "fail" | "unknown";

export interface CandidateSloVerdict {
  latency: SloVerdict;
  throughput: SloVerdict;
  /** "fail" if either metric fails; else "unknown" if either is unknown;
   * else "pass". */
  overall: SloVerdict;
}

function combine(latency: SloVerdict, throughput: SloVerdict): SloVerdict {
  if (latency === "fail" || throughput === "fail") return "fail";
  if (latency === "unknown" || throughput === "unknown") return "unknown";
  return "pass";
}

/**
 * Pure SLO judgment for one candidate's runtime prediction -- no fetching,
 * no state, same inputs always produce the same verdict. Compares the
 * prediction's p90 (matching the "SLO: p90 ≤ ...ms" bar already shown in
 * ConditionsSummary) against slo.max_latency_ms, and the naive
 * predicted_throughput_rps stub against slo.min_throughput_rps.
 */
export function evaluateSlo(
  prediction: RuntimePrediction | undefined,
  slo: SloInput | null
): CandidateSloVerdict {
  if (!slo) {
    return { latency: "unknown", throughput: "unknown", overall: "unknown" };
  }

  const ok = prediction?.status === "ok";

  const latency: SloVerdict =
    ok && prediction!.predicted_p90_ms !== undefined
      ? prediction!.predicted_p90_ms <= slo.max_latency_ms
        ? "pass"
        : "fail"
      : "unknown";

  const throughput: SloVerdict =
    ok && prediction!.predicted_throughput_rps !== undefined
      ? prediction!.predicted_throughput_rps >= slo.min_throughput_rps
        ? "pass"
        : "fail"
      : "unknown";

  return { latency, throughput, overall: combine(latency, throughput) };
}
