import type { RecommendationRow } from "@/lib/types";

/**
 * User-weighted ordering of the instances that already pass both SLOs (and
 * the budget). SLOs and budget stay hard conditions in
 * orchestrator/recommendation.py; this only re-orders the passing rows, so no
 * weighting can recommend an instance that misses an SLO.
 *
 * Criteria (single instance):
 *  - cost: hourly price
 *  - latency: P90 at the operating batch (T)
 *  - throughput: NOT a criterion. Every passing instance carries the
 *    throughput SLO within the target utilization (80%), so all are treated
 *    as equally satisfying it (decided 2026-11-23).
 *
 * Each value becomes "x times the best passing value" (1 = best, 1.3 = 30%
 * worse) instead of a 0-1 min-max score: it keeps a meaning, never gives the
 * best-in-one-criterion instance a zero, and reads as "+30% cost" on screen.
 * score = w_cost x cost ratio + w_latency x latency ratio; lowest wins.
 */

export type WeightPreset = "cost" | "speed" | "balanced";

export interface Weights {
  cost: number;
  latency: number;
}

export const PRESET_WEIGHTS: Record<WeightPreset, Weights> = {
  cost: { cost: 1, latency: 0 },
  speed: { cost: 0, latency: 1 },
  balanced: { cost: 0.5, latency: 0.5 },
};

export const PRESET_LABEL: Record<WeightPreset, string> = {
  cost: "비용 중시",
  speed: "속도 중시",
  balanced: "균형",
};

export interface ScoredRow {
  row: RecommendationRow;
  costRatio: number;
  latencyRatio: number;
  score: number;
}

/** Passing rows within budget (or, when none fits the budget, all passing
 * rows -- same pool the backend uses for the cheapest plan), best first. */
export function scoreRows(rows: RecommendationRow[], weights: Weights): ScoredRow[] {
  const ok = rows.filter(
    (r) => r.status === "ok" && r.cost_usd_per_hour != null && r.latency_ms != null
  );
  const inBudget = ok.filter((r) => !r.over_budget);
  const pool = inBudget.length > 0 ? inBudget : ok;
  if (pool.length === 0) return [];
  const bestCost = Math.min(...pool.map((r) => r.cost_usd_per_hour!));
  const bestLatency = Math.min(...pool.map((r) => r.latency_ms!));
  const total = weights.cost + weights.latency || 1;
  return pool
    .map((row) => {
      const costRatio = row.cost_usd_per_hour! / bestCost;
      const latencyRatio = row.latency_ms! / bestLatency;
      const score = (weights.cost * costRatio + weights.latency * latencyRatio) / total;
      return { row, costRatio, latencyRatio, score };
    })
    .sort(
      (a, b) =>
        a.score - b.score ||
        a.row.cost_usd_per_hour! - b.row.cost_usd_per_hour! ||
        a.row.latency_ms! - b.row.latency_ms! ||
        a.row.instance_type.localeCompare(b.row.instance_type)
    );
}

/** "+30%" / "-40%" / "같음" for a pick compared with another row. */
export function relative(value: number, base: number): string {
  if (Math.abs(value - base) <= base * 1e-6) return "같음";
  const pct = Math.round((value / base - 1) * 100);
  return pct > 0 ? `+${pct}%` : `${pct}%`;
}
