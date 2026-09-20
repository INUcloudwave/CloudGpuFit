import type { InstanceCandidate, RuntimePrediction, SloInput } from "@/lib/types";
import { evaluateSlo, type SloVerdict } from "@/lib/slo";

/** Relative importance the user assigns to each factor. Only the ratio
 * between them matters -- {latency:1,throughput:1} and
 * {latency:10,throughput:10} rank identically. Price is deliberately not a
 * scoring factor (shown in the table, but not weighted) -- that's a product
 * decision, not an oversight. */
export interface ImportanceWeights {
  latency: number;
  throughput: number;
}

export const DEFAULT_WEIGHTS: ImportanceWeights = { latency: 50, throughput: 50 };

export interface RankedCandidate {
  candidate: InstanceCandidate;
  prediction: RuntimePrediction | undefined;
  /** Same overall verdict evaluateSlo() would give this candidate -- reused
   * here (not recomputed differently) so the SLO badge and the ranking tier
   * always agree. */
  sloVerdict: SloVerdict;
  /** 0 (worst) .. 1 (best) among the candidates that had a successful
   * runtime prediction, or null when this candidate has none. Null-scored
   * candidates still appear in the ranking, just always last within their
   * SLO tier. */
  score: number | null;
}

/** min-max normalize into [0,1], where 1 is always "best" for that metric
 * (inverted for metrics where lower is better). A degenerate range (every
 * candidate tied) scores everyone 1 rather than dividing by zero -- ties
 * shouldn't be read as "worst". */
function normalize(value: number, min: number, max: number, higherIsBetter: boolean): number {
  if (max === min) return 1;
  const t = (value - min) / (max - min);
  return higherIsBetter ? t : 1 - t;
}

/**
 * Pure ranking over one job's candidates: no fetching, no state. SLO
 * compliance is the primary sort key (pass > unknown > fail, matching
 * evaluateSlo's own tiering) -- a candidate that misses the SLO bar should
 * never outrank one that meets it. Within a tier, candidates are ordered by
 * a weighted score over p90 latency (lower is better) and the naive
 * throughput estimate (higher is better), each min-max normalized across
 * only the candidates that have a successful runtime prediction.
 */
export function rankCandidates(
  candidates: InstanceCandidate[],
  predictions: RuntimePrediction[],
  slo: SloInput | null,
  weights: ImportanceWeights
): RankedCandidate[] {
  const predictionByTarget = new Map(predictions.map((p) => [p.target_instance, p]));

  const rows = candidates.map((candidate) => {
    const prediction = predictionByTarget.get(candidate.instance_type);
    const sloVerdict = evaluateSlo(prediction, slo).overall;
    return { candidate, prediction, sloVerdict };
  });

  const scoreable = rows.filter(
    (r) =>
      r.prediction?.status === "ok" &&
      r.prediction.predicted_p90_ms !== undefined &&
      r.prediction.predicted_throughput_rps !== undefined
  );

  const latencies = scoreable.map((r) => r.prediction!.predicted_p90_ms!);
  const throughputs = scoreable.map((r) => r.prediction!.predicted_throughput_rps!);
  const latencyRange = [Math.min(...latencies), Math.max(...latencies)] as const;
  const throughputRange = [Math.min(...throughputs), Math.max(...throughputs)] as const;

  const totalWeight = weights.latency + weights.throughput || 1;

  const scoreableSet = new Set(scoreable);
  const ranked: RankedCandidate[] = rows.map((r) => {
    if (!scoreableSet.has(r)) {
      return { ...r, score: null };
    }
    const latencyScore = normalize(r.prediction!.predicted_p90_ms!, ...latencyRange, false);
    const throughputScore = normalize(r.prediction!.predicted_throughput_rps!, ...throughputRange, true);
    const score = (weights.latency * latencyScore + weights.throughput * throughputScore) / totalWeight;
    return { ...r, score };
  });

  const tierRank: Record<SloVerdict, number> = { pass: 0, unknown: 1, fail: 2 };
  return ranked.sort((a, b) => {
    const tierDiff = tierRank[a.sloVerdict] - tierRank[b.sloVerdict];
    if (tierDiff !== 0) return tierDiff;
    if (a.score === null && b.score === null) return 0;
    if (a.score === null) return 1;
    if (b.score === null) return -1;
    return b.score - a.score;
  });
}
