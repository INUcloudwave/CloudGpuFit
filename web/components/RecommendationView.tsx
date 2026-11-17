"use client";

import { useState } from "react";
import type {
  JobResponse,
  Recommendation,
  RecommendationPlan,
  RecommendationRow,
  SloInput,
} from "@/lib/types";
import {
  DEFAULT_TARGET_UTILIZATION,
  MAX_TARGET_UTILIZATION,
  MIN_TARGET_UTILIZATION,
  describeLatency,
} from "@/lib/sloPresets";

/**
 * Result screen for orchestrator/recommendation.py: one instance type x N
 * units. The backend computes everything (units, operating batch, verdict,
 * cost); this component only formats it. Changing the target utilization
 * re-submits the SLO and the backend recomputes on read -- no new prediction.
 */

type CostUnit = "hour" | "month";

function money(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined) return "—";
  return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: digits })}`;
}

function costText(plan: Partial<RecommendationPlan>, unit: CostUnit): string {
  return unit === "hour"
    ? `${money(plan.cost_usd_per_hour)} /시간`
    : `${money(plan.cost_usd_per_month, 0)} /월`;
}

function ms(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${value.toLocaleString(undefined, { maximumFractionDigits: value < 10 ? 2 : 1 })} ms`;
}

function rps(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} req/s`;
}

function pct(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${Math.round(value * 100)}%`;
}

const VERDICT_STYLE = {
  pass: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-400",
  edge: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-400",
} as const;

const VERDICT_LABEL = {
  pass: "대기 포함 충족",
  edge: "대기 시 초과 가능",
} as const;

function VerdictBadge({ verdict }: { verdict?: "pass" | "edge" }) {
  if (!verdict) return <span className="text-neutral-400">—</span>;
  return (
    <span className={`whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ${VERDICT_STYLE[verdict]}`}>
      {VERDICT_LABEL[verdict]}
    </span>
  );
}

const ROW_STATUS_LABEL: Record<string, string> = {
  no_price: "가격 정보 없음",
  slo_impossible: "지연 SLO 불가",
  no_prediction: "예측 실패",
};

function PlanCard({
  title,
  subtitle,
  plan,
  costUnit,
  maxLatencyMs,
  tokens,
  highlight,
}: {
  title: string;
  subtitle: string;
  plan: RecommendationPlan;
  costUnit: CostUnit;
  maxLatencyMs: number | null;
  tokens: number | null;
  highlight?: boolean;
}) {
  return (
    <div
      className={`rounded-xl border p-5 ${
        highlight
          ? "border-neutral-900 dark:border-white"
          : "border-neutral-200 dark:border-neutral-700"
      }`}
    >
      <p className="text-xs font-medium text-neutral-500">{title}</p>
      <p className="mt-1 font-mono text-2xl font-semibold text-neutral-900 dark:text-neutral-100">
        {plan.instance_type} × {plan.units}대
      </p>
      <p className="mt-0.5 text-xs text-neutral-500">{subtitle}</p>
      <p className="mt-3 font-mono text-lg font-semibold text-neutral-900 dark:text-neutral-100">
        {costText(plan, costUnit)}
      </p>
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
        <dt className="text-neutral-500">운영 배치</dt>
        <dd className="font-mono">{plan.operating_batch}</dd>
        <dt className="text-neutral-500">배치 1회 지연 (P90)</dt>
        <dd className="font-mono">
          {ms(plan.latency_ms)}
          {maxLatencyMs !== null && <span className="text-neutral-400"> / {ms(maxLatencyMs)}</span>}
          {tokens !== null && (
            <span className="block text-xs text-neutral-400">토큰당 약 {ms(plan.latency_ms / tokens)}</span>
          )}
        </dd>
        <dt className="text-neutral-500">대기 포함 최악 (2T)</dt>
        <dd className="flex flex-wrap items-center gap-2 font-mono">
          {ms(plan.wait_latency_ms)} <VerdictBadge verdict={plan.verdict} />
        </dd>
        <dt className="text-neutral-500">대당 처리량 / 전체 용량</dt>
        <dd className="font-mono">
          {rps(plan.per_unit_rps)} / {rps(plan.capacity_rps)}
        </dd>
        <dt className="text-neutral-500">사용률</dt>
        <dd className="font-mono">{pct(plan.utilization)}</dd>
      </dl>
    </div>
  );
}

function StatusNote({ rec }: { rec: Recommendation }) {
  const text: Partial<Record<Recommendation["status"], string>> = {
    no_slo: "SLO가 아직 입력되지 않아 추천을 계산하지 않았습니다.",
    over_budget:
      "예산 안에서 SLO를 만족하는 구성이 없습니다. 예산을 넘지만 가장 저렴한 구성을 보여 드립니다.",
    slo_impossible:
      "어느 후보도 지연 SLO를 만족하지 못합니다. 아래 표의 '최소 지연'을 참고해 SLO를 완화해 보세요.",
    no_price: "후보 인스턴스의 가격 정보가 없어 비용을 계산할 수 없습니다.",
    no_prediction: "후보 인스턴스의 성능 예측이 모두 실패했습니다.",
    error: `추천 계산 중 오류가 발생했습니다${rec.error ? `: ${rec.error}` : "."}`,
  };
  const body = text[rec.status];
  if (!body) return null;
  const danger = rec.status === "error" || rec.status === "slo_impossible" || rec.status === "no_prediction";
  return (
    <p
      className={`rounded-lg px-4 py-3 text-sm ${
        danger
          ? "bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-400"
          : "bg-amber-50 text-amber-700 dark:bg-amber-950 dark:text-amber-400"
      }`}
    >
      {body}
    </p>
  );
}

function RowsTable({
  rows,
  cheapest,
  similar,
  costUnit,
}: {
  rows: RecommendationRow[];
  cheapest: string | null | undefined;
  similar: string[];
  costUnit: CostUnit;
}) {
  return (
    <div className="overflow-hidden rounded-xl border border-neutral-200 dark:border-neutral-700">
      <div className="border-b border-neutral-200 px-5 py-4 dark:border-neutral-700">
        <h3 className="text-sm font-semibold text-neutral-800 dark:text-neutral-200">전체 후보</h3>
        <p className="mt-0.5 text-xs text-neutral-500">
          인스턴스마다 수요를 감당하는 최소 대수와, 그 대수에서 지연이 가장 짧은 운영 배치를 계산했습니다.
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-neutral-50 text-left text-xs text-neutral-500 dark:bg-neutral-900 dark:text-neutral-400">
            <tr>
              <th className="px-4 py-3">인스턴스</th>
              <th className="px-4 py-3">대수</th>
              <th className="px-4 py-3">운영 배치</th>
              <th className="px-4 py-3">지연 P90 (T)</th>
              <th className="px-4 py-3">대기 포함 (2T)</th>
              <th className="px-4 py-3">사용률</th>
              <th className="px-4 py-3">비용</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800">
            {rows.map((r) => {
              const isCheapest = r.instance_type === cheapest;
              const isSimilar = similar.includes(r.instance_type);
              return (
                <tr
                  key={r.instance_type}
                  className={r.status !== "ok" || r.over_budget ? "text-neutral-400" : undefined}
                >
                  <td className="px-4 py-3 font-mono">
                    <div className="flex flex-wrap items-center gap-1.5">
                      {r.instance_type}
                      {isCheapest && (
                        <span className="rounded-full bg-neutral-900 px-2 py-0.5 font-sans text-[11px] text-white dark:bg-white dark:text-neutral-900">
                          추천
                        </span>
                      )}
                      {isSimilar && (
                        <span className="rounded-full bg-neutral-200 px-2 py-0.5 font-sans text-[11px] text-neutral-700 dark:bg-neutral-700 dark:text-neutral-200">
                          비슷한 비용
                        </span>
                      )}
                      {r.over_budget && (
                        <span className="rounded-full bg-amber-100 px-2 py-0.5 font-sans text-[11px] text-amber-700 dark:bg-amber-950 dark:text-amber-400">
                          예산 초과
                        </span>
                      )}
                    </div>
                    {r.latency_proxy_of && (
                      <p className="mt-0.5 font-sans text-[11px] text-neutral-400">
                        같은 GPU {r.latency_proxy_of} 예측값 사용
                      </p>
                    )}
                  </td>
                  {r.status === "ok" ? (
                    <>
                      <td className="px-4 py-3 font-mono">{r.units}</td>
                      <td className="px-4 py-3 font-mono">{r.operating_batch}</td>
                      <td className="px-4 py-3 font-mono">{ms(r.latency_ms)}</td>
                      <td className="px-4 py-3">
                        <div className="flex flex-wrap items-center gap-2 font-mono">
                          {ms(r.wait_latency_ms)} <VerdictBadge verdict={r.verdict} />
                        </div>
                      </td>
                      <td className="px-4 py-3 font-mono">{pct(r.utilization)}</td>
                      <td className="px-4 py-3 font-mono">{costText(r, costUnit)}</td>
                    </>
                  ) : (
                    <td className="px-4 py-3 text-xs" colSpan={6}>
                      {ROW_STATUS_LABEL[r.status] ?? r.status}
                      {r.status === "slo_impossible" && r.min_latency_ms !== undefined && (
                        <> — 최소 지연 {ms(r.min_latency_ms)} (배치 {r.min_latency_batch})</>
                      )}
                      {r.status === "no_prediction" && r.reason && <> — {r.reason}</>}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function RecommendationView({
  job,
  onJobUpdate,
  onReset,
}: {
  job: JobResponse;
  onJobUpdate: (job: JobResponse) => void;
  onReset: () => void;
}) {
  const rec = job.result?.recommendation;
  const slo = job.slo;
  const [costUnit, setCostUnit] = useState<CostUnit>("hour");
  const appliedTarget = Math.min(
    slo?.target_utilization ?? rec?.policy?.target_utilization ?? DEFAULT_TARGET_UTILIZATION,
    MAX_TARGET_UTILIZATION
  );
  const limit = rec?.latency_limit ?? null;
  const maxLatencyMs = limit?.max_latency_ms ?? null;
  const tokens = limit?.basis === "per_token" ? limit.tokens : null;
  const [draftTarget, setDraftTarget] = useState(Math.round(appliedTarget * 100));
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  async function applyTarget() {
    if (!slo) return;
    setSaving(true);
    setSaveError(null);
    try {
      const body: SloInput = { ...slo, target_utilization: draftTarget / 100 };
      const res = await fetch(`/api/jobs/${job.job_id}/slo`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        setSaveError("목표 사용률을 저장하지 못했습니다.");
        return;
      }
      onJobUpdate((await res.json()) as JobResponse);
    } catch {
      setSaveError("오케스트레이터에 연결할 수 없습니다.");
    } finally {
      setSaving(false);
    }
  }

  const cheapestRow = rec?.rows.find((r) => r.instance_type === rec.cheapest && r.status === "ok");
  const budget = job.payload.deployment.budget_usd_per_hour;

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold text-neutral-900 dark:text-neutral-100">GPU 인스턴스 추천 결과</h2>
          <p className="mt-1 text-sm text-neutral-500">
            한 종류의 인스턴스를 필요한 대수만큼 띄우는 구성 중, SLO를 만족하는 가장 저렴한 구성을 추천합니다.
          </p>
          <dl className="mt-3 space-y-1 text-sm text-neutral-600 dark:text-neutral-400">
            <div className="flex gap-2">
              <dt className="font-medium text-neutral-700 dark:text-neutral-300">모델</dt>
              <dd className="font-mono">
                {job.payload.model} · 최대 배치 {job.payload.batch_size}
              </dd>
            </div>
            {slo && (
              <div className="flex gap-2">
                <dt className="font-medium text-neutral-700 dark:text-neutral-300">SLO</dt>
                <dd>
                  {describeLatency(slo, job.payload)} · 수요 {slo.min_throughput_rps} req/s
                </dd>
              </div>
            )}
            <div className="flex gap-2">
              <dt className="font-medium text-neutral-700 dark:text-neutral-300">예산</dt>
              <dd>
                {costUnit === "hour" ? `${money(budget)} /시간` : `${money(budget * 720, 0)} /월`} 이하 · 리전{" "}
                {job.payload.deployment.aws_region}
              </dd>
            </div>
          </dl>
        </div>

        <div className="flex flex-col items-end gap-3">
          <div className="inline-flex rounded-lg border border-neutral-300 p-0.5 text-xs dark:border-neutral-700">
            {(["hour", "month"] as const).map((u) => (
              <button
                key={u}
                type="button"
                onClick={() => setCostUnit(u)}
                className={`rounded-md px-3 py-1.5 font-medium ${
                  costUnit === u
                    ? "bg-neutral-900 text-white dark:bg-white dark:text-neutral-900"
                    : "text-neutral-600 dark:text-neutral-400"
                }`}
              >
                {u === "hour" ? "시간당" : "월 (720시간)"}
              </button>
            ))}
          </div>

          {slo && (
            <div className="w-64 rounded-lg border border-neutral-200 p-3 dark:border-neutral-700">
              <div className="flex items-baseline justify-between">
                <p className="text-xs font-medium text-neutral-700 dark:text-neutral-300">목표 사용률</p>
                <span className="font-mono text-sm font-semibold">{draftTarget}%</span>
              </div>
              <input
                type="range"
                min={MIN_TARGET_UTILIZATION * 100}
                max={MAX_TARGET_UTILIZATION * 100}
                step={5}
                value={draftTarget}
                onChange={(e) => setDraftTarget(Number(e.target.value))}
                className="mt-1 w-full accent-neutral-800 dark:accent-white"
              />
              <p className="text-[11px] leading-snug text-neutral-400">
                대당 최대 처리량의 몇 %까지 쓸지입니다(기본 80%, 최대 90%). 사용률이 오를수록 요청이 줄을 서는
                시간이 급격히 늘어 80%에서 처리 시간의 약 4배, 90%에서 약 9배가 됩니다. 예측이 실측보다 10~20%
                빠르게 나오는 GPU도 있어 여유를 둡니다.
              </p>
              <button
                type="button"
                disabled={saving || draftTarget === Math.round(appliedTarget * 100)}
                onClick={applyTarget}
                className="mt-2 w-full rounded-lg bg-neutral-900 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40 dark:bg-white dark:text-neutral-900"
              >
                {saving ? "적용 중…" : "적용"}
              </button>
              {saveError && <p className="mt-1 text-xs text-red-600">{saveError}</p>}
            </div>
          )}
        </div>
      </div>

      {!rec ? (
        <p className="rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:bg-amber-950 dark:text-amber-400">
          추천 결과가 없습니다.
        </p>
      ) : (
        <>
          <StatusNote rec={rec} />

          {cheapestRow && slo && (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <PlanCard
                title={rec.status === "over_budget" ? "가장 저렴한 구성 (예산 초과)" : "추천 — 최저 비용"}
                subtitle="SLO를 만족하는 구성 중 가장 저렴합니다."
                plan={cheapestRow as RecommendationPlan}
                costUnit={costUnit}
                maxLatencyMs={maxLatencyMs}
                tokens={tokens}
                highlight
              />
              {rec.budget_plan && !rec.budget_plan.same_as_cheapest ? (
                <PlanCard
                  title="예산 활용 — 최저 지연"
                  subtitle="예산 안에서 대수를 늘려 지연을 가장 짧게 만든 구성입니다."
                  plan={rec.budget_plan}
                  costUnit={costUnit}
                  maxLatencyMs={maxLatencyMs}
                tokens={tokens}
                />
              ) : (
                <div className="rounded-xl border border-dashed border-neutral-200 p-5 text-sm text-neutral-500 dark:border-neutral-700">
                  <p className="text-xs font-medium">예산 활용 — 최저 지연</p>
                  <p className="mt-2">
                    {rec.budget_plan
                      ? "예산 안에서 더 빠른 구성이 없어 최저 비용 구성과 같습니다."
                      : "예산 안에 들어가는 구성이 없습니다."}
                  </p>
                </div>
              )}
            </div>
          )}

          {rec.status === "slo_impossible" && tokens !== null && (
            <p className="text-xs text-neutral-500">
              생성 토큰 수({tokens}개)를 줄이거나 지연 기준을 ‘일반 대화’나 ‘제한 없음’으로 바꾸면 선택지가 생길 수
              있습니다.
            </p>
          )}

          {rec.similar && rec.similar.length > 0 && (
            <p className="text-sm text-neutral-500">
              비용 차이 {rec.policy?.similar_pct ?? 10}% 이내의 다른 선택지:{" "}
              <span className="font-mono">{rec.similar.join(", ")}</span>
            </p>
          )}

          {rec.rows.length > 0 && (
            <RowsTable rows={rec.rows} cheapest={rec.cheapest} similar={rec.similar ?? []} costUnit={costUnit} />
          )}

          <p className="text-xs leading-relaxed text-neutral-400">
            지연은 예측기의 P90(보수적 추정)으로 판정합니다. 생성형 모델의 지연은 입력 처리부터 마지막 토큰까지
            요청 전체 시간이며, 토큰당 기준(실시간 50ms, 일반 100ms)은 LLM 서빙 연구에서 대화형 서비스에 쓰는 토큰
            간 지연(TPOT) 범위를 따랐습니다. 대기 포함(2T)은 앞 배치가 끝나기를 기다린 요청의
            최악 지연으로, ‘대기 시 초과 가능’은 배치 1회는 SLO 안이지만 대기까지 더하면 넘을 수 있다는 뜻입니다.
            각 인스턴스는 모델 전체를 따로 올리고 요청을 나눠 받습니다. 월 비용은 시간당 × 720시간입니다.
          </p>
        </>
      )}

      <button
        type="button"
        onClick={onReset}
        className="rounded-lg border border-neutral-300 px-5 py-2.5 text-sm font-medium text-neutral-700 dark:border-neutral-700 dark:text-neutral-300"
      >
        새 예측 시작
      </button>
    </div>
  );
}
