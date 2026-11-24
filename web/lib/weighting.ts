import type { RecommendationRow, WorkloadPurpose } from "@/lib/types";

/**
 * User-weighted ordering of the instances that already pass the SLOs (and the
 * budget). SLOs and budget stay hard conditions in
 * orchestrator/recommendation.py; this only re-orders the passing rows, so no
 * weighting can recommend an instance that misses an SLO.
 *
 * Two criteria per purpose:
 *  - online : cost = hourly price, other = latency T (P90 of one batch run at
 *             the operating batch). Throughput is not a criterion: every
 *             passing instance carries the demand within the target load.
 *  - offline: cost = cost per request (hourly price / throughput), other =
 *             time per request (1 / throughput at the operating batch).
 *
 * Each value becomes "x times the best passing value" (1 = best, 1.3 = 30%
 * worse) instead of a 0-1 min-max score: it keeps a meaning, never gives the
 * best-in-one-criterion instance a zero, and reads as "+30%" on screen.
 *
 * score = w x cost ratio + (1 - w) x other ratio, w in [0, 1]; lowest wins.
 * Only the order is shown, so the weights need no normalising.
 */

export type WeightPreset = "cost" | "speed" | "balanced";

/** Share of the cost criterion for each preset button. */
export const PRESET_COST_WEIGHT: Record<WeightPreset, number> = {
  cost: 1,
  speed: 0,
  balanced: 0.5,
};

export function presetLabel(preset: WeightPreset, purpose: WorkloadPurpose): string {
  if (preset === "cost") return "비용 중시";
  if (preset === "balanced") return "균형";
  return purpose === "offline" ? "처리량 중시" : "속도 중시";
}

/** Names of the two criteria, for the slider and the ratio column. */
export function criteriaLabels(purpose: WorkloadPurpose): { cost: string; other: string } {
  return purpose === "offline" ? { cost: "요청당 비용", other: "처리 시간" } : { cost: "요금", other: "지연" };
}

/** The preset a cost weight equals, or null for a fine-tuned weight. */
export function presetOf(costWeight: number): WeightPreset | null {
  const hit = (Object.keys(PRESET_COST_WEIGHT) as WeightPreset[]).find(
    (p) => Math.abs(PRESET_COST_WEIGHT[p] - costWeight) < 1e-9
  );
  return hit ?? null;
}

export interface ScoredRow {
  row: RecommendationRow;
  costRatio: number;
  otherRatio: number;
  score: number;
}

/** The two raw values a row is compared on (smaller is better for both). */
export function criteriaValues(
  row: RecommendationRow,
  purpose: WorkloadPurpose
): { cost: number; other: number } | null {
  if (purpose === "offline") {
    if (row.cost_usd_per_million == null || row.operating_rps == null || row.operating_rps <= 0) return null;
    return { cost: row.cost_usd_per_million, other: 1 / row.operating_rps };
  }
  if (row.cost_usd_per_hour == null || row.latency_ms == null) return null;
  return { cost: row.cost_usd_per_hour, other: row.latency_ms };
}

/** Passing rows within budget (or, when none fits the budget, all passing
 * rows -- same pool the backend uses for the cheapest plan), best first. */
export function scoreRows(rows: RecommendationRow[], costWeight: number, purpose: WorkloadPurpose): ScoredRow[] {
  const w = Math.min(1, Math.max(0, costWeight));
  const ok = rows
    .filter((r) => r.status === "ok")
    .map((row) => ({ row, v: criteriaValues(row, purpose) }))
    .filter((x): x is { row: RecommendationRow; v: { cost: number; other: number } } => x.v !== null);
  const inBudget = ok.filter((x) => !x.row.over_budget);
  const pool = inBudget.length > 0 ? inBudget : ok;
  if (pool.length === 0) return [];
  const bestCost = Math.min(...pool.map((x) => x.v.cost));
  const bestOther = Math.min(...pool.map((x) => x.v.other));
  return pool
    .map(({ row, v }) => {
      const costRatio = v.cost / bestCost;
      const otherRatio = v.other / bestOther;
      return { row, costRatio, otherRatio, score: w * costRatio + (1 - w) * otherRatio, v };
    })
    .sort(
      (a, b) =>
        a.score - b.score ||
        a.v.cost - b.v.cost ||
        a.v.other - b.v.other ||
        a.row.instance_type.localeCompare(b.row.instance_type)
    )
    .map(({ row, costRatio, otherRatio, score }) => ({ row, costRatio, otherRatio, score }));
}

/** "+30%" / "-40%" / "같음" for a pick compared with another row. */
export function relative(value: number, base: number): string {
  if (Math.abs(value - base) <= base * 1e-6) return "같음";
  const pct = Math.round((value / base - 1) * 100);
  return pct > 0 ? `+${pct}%` : `${pct}%`;
}
