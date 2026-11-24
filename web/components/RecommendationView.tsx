"use client";

import { useState } from "react";
import type {
  JobResponse,
  Recommendation,
  RecommendationPlan,
  RecommendationRow,
  WorkloadPurpose,
} from "@/lib/types";
import {
  PRESET_COST_WEIGHT,
  criteriaLabels,
  criteriaValues,
  presetLabel,
  presetOf,
  relative,
  scoreRows,
  type ScoredRow,
  type WeightPreset,
} from "@/lib/weighting";
import { describeLatency, describeWorkload } from "@/lib/sloPresets";

/**
 * Result screen for orchestrator/recommendation.py: ONE instance. The backend
 * computes everything (operating batch, latency, cost); this component only
 * formats it and orders the passing instances by the user's weights.
 * Moving the weights is local and instant -- no new prediction.
 *
 * Operating batch: the batch the user entered, stepped down (128, 64, 32, 16,
 * 8, 1) only when it misses an SLO. BatchReason says why it stepped down.
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

function hours(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (value < 1) return `${Math.round(value * 60)}분`;
  return `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })}시간`;
}

const ROW_STATUS_LABEL: Record<string, string> = {
  throughput_short: "처리량 SLO 미달",
  no_price: "가격 정보 없음",
  slo_impossible: "지연 SLO 불가",
  no_prediction: "예측 실패",
};

/** Why the operating batch is what it is, relative to the batch the user entered. */
function batchReason(row: RecommendationRow, offline: boolean): string {
  if (row.input_batch === undefined || row.operating_batch === row.input_batch) return "입력한 배치 그대로";
  if (row.input_batch_latency_ok === false) return `입력 배치 ${row.input_batch}: 지연 SLO 초과로 내림`;
  return offline
    ? `입력 배치 ${row.input_batch}: 아래 배치의 처리량이 더 커서 내림`
    : `입력 배치 ${row.input_batch}: 처리량 SLO 미달로 내림`;
}

/** "배치 128: 14,790 ms · 9.9 req/s · 지연 SLO 충족" -- the user's own batch. */
function InputBatchNote({ row, compact }: { row: RecommendationRow; compact?: boolean }) {
  if (row.input_batch === undefined) return null;
  if (row.input_batch_latency_ms == null) {
    return <span className="text-neutral-400">배치 {row.input_batch}: 예측 없음</span>;
  }
  const ok = row.input_batch_latency_ok;
  return (
    <span>
      {!compact && <>배치 {row.input_batch}: </>}
      {ms(row.input_batch_latency_ms)} · {rps(row.input_batch_rps)}{" "}
      <span className={ok ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}>
        {ok ? "지연 SLO 충족" : "지연 SLO 초과"}
      </span>
    </span>
  );
}

function PlanCard({
  title,
  subtitle,
  plan,
  purpose,
  costUnit,
  maxLatencyMs,
  tokens,
  demand,
  totalRequests,
  highlight,
}: {
  title: string;
  subtitle: string;
  plan: RecommendationRow;
  purpose: WorkloadPurpose;
  costUnit: CostUnit;
  maxLatencyMs: number | null;
  tokens: number | null;
  demand: number | null;
  totalRequests: number | null;
  highlight?: boolean;
}) {
  const offline = purpose === "offline";
  return (
    <div
      className={`rounded-xl border p-5 ${
        highlight ? "border-neutral-900 dark:border-white" : "border-neutral-200 dark:border-neutral-700"
      }`}
    >
      <p className="text-xs font-medium text-neutral-500">{title}</p>
      <p className="mt-1 font-mono text-2xl font-semibold text-neutral-900 dark:text-neutral-100">
        {plan.instance_type} <span className="text-base font-normal text-neutral-500">× 1대</span>
      </p>
      <p className="mt-0.5 text-xs text-neutral-500">{subtitle}</p>
      <p className="mt-3 font-mono text-lg font-semibold text-neutral-900 dark:text-neutral-100">
        {offline ? `${money(plan.cost_usd_per_million)} /100만 건` : costText(plan, costUnit)}
      </p>
      {offline && <p className="font-mono text-xs text-neutral-500">{costText(plan, costUnit)}</p>}
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
        <dt className="text-neutral-500">운영 배치</dt>
        <dd className="font-mono">
          {plan.operating_batch}
          <span className="block font-sans text-xs text-neutral-400">
            {batchReason(plan, offline)}
          </span>
        </dd>
        <dt className="text-neutral-500">배치 1회 처리 시간 P90 (T)</dt>
        <dd className="font-mono">
          {ms(plan.latency_ms)}
          {maxLatencyMs !== null && <span className="text-neutral-400"> / {ms(maxLatencyMs)}</span>}
          {tokens !== null && plan.latency_ms != null && (
            <span className="block text-xs text-neutral-400">토큰당 약 {ms(plan.latency_ms / tokens)}</span>
          )}
        </dd>
        {offline ? (
          <>
            <dt className="text-neutral-500">처리량</dt>
            <dd className="font-mono">{rps(plan.operating_rps)}</dd>
            <dt className="text-neutral-500">완료 예상{totalRequests ? ` (${totalRequests.toLocaleString()}건)` : ""}</dt>
            <dd className="font-mono">
              {totalRequests ? (
                <>
                  {hours(plan.job_hours)}
                  <span className="block text-xs text-neutral-400">비용 {money(plan.job_cost_usd, 2)}</span>
                </>
              ) : (
                <span className="font-sans text-xs text-neutral-400">요청 수를 입력하면 계산</span>
              )}
            </dd>
          </>
        ) : (
          <>
            <dt className="text-neutral-500">운영 배치 처리량</dt>
            <dd className="font-mono">
              {rps(plan.operating_rps)}
              <span className="block text-xs text-neutral-400">요청량 {rps(demand)} 감당</span>
            </dd>
            <dt className="text-neutral-500">최대 처리량</dt>
            <dd className="font-mono">
              {rps(plan.capacity_rps)}
              <span className="block text-xs text-neutral-400">
                배치 {plan.capacity_batch} 기준 · 요청량 비중 {pct(plan.demand_share)}
              </span>
            </dd>
          </>
        )}
        <dt className="text-neutral-500">입력한 배치</dt>
        <dd className="font-mono text-xs">
          <InputBatchNote row={plan} />
        </dd>
      </dl>
    </div>
  );
}

function StatusNote({ rec }: { rec: Recommendation }) {
  const text: Partial<Record<Recommendation["status"], string>> = {
    no_slo: "SLO가 아직 입력되지 않아 추천을 계산하지 않았습니다.",
    over_budget:
      "예산 안에서 SLO를 만족하는 인스턴스가 없습니다. 예산을 넘지만 가장 저렴한 인스턴스를 보여 드립니다.",
    throughput_short:
      "한 대로 이 요청량을 감당하는 인스턴스가 없습니다. 표의 '최대 처리량'을 참고해 요청량을 줄이거나, 최대 배치를 늘리거나, 지연 SLO를 완화해 보세요.",
    slo_impossible:
      "어느 후보도 지연 SLO를 만족하지 못합니다. 아래 표의 '최소 지연'을 참고해 SLO를 완화해 보세요.",
    no_price: "후보 인스턴스의 가격 정보가 없어 비용을 계산할 수 없습니다.",
    no_prediction: "후보 인스턴스의 성능 예측이 모두 실패했습니다.",
    error: `추천 계산 중 오류가 발생했습니다${rec.error ? `: ${rec.error}` : "."}`,
  };
  const body = text[rec.status];
  if (!body) return null;
  const danger =
    rec.status === "error" ||
    rec.status === "slo_impossible" ||
    rec.status === "throughput_short" ||
    rec.status === "no_prediction";
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
  pickName,
  similar,
  purpose,
  costUnit,
  scored,
  inputBatch,
}: {
  rows: RecommendationRow[];
  pickName: string | null | undefined;
  similar: string[];
  purpose: WorkloadPurpose;
  costUnit: CostUnit;
  scored: ScoredRow[];
  inputBatch: number;
}) {
  // Passing rows in the user's weighted order first, then the rest as the
  // backend ordered them.
  const ratios = new Map(scored.map((s) => [s.row.instance_type, s]));
  const ordered = [...scored.map((s) => s.row), ...rows.filter((r) => !ratios.has(r.instance_type))];
  const offline = purpose === "offline";
  const labels = criteriaLabels(purpose);
  const cols = offline
    ? ["인스턴스", "운영 배치", "T (운영 배치)", "처리량 (운영 배치)", "요청 100만 건당", "완료 예상", `입력 배치 ${inputBatch}`, "시간당 요금", `최저 대비 (${labels.cost} / ${labels.other})`]
    : ["인스턴스", "운영 배치", "T (운영 배치)", "처리량 (운영 배치)", "최대 처리량 · 요청량 비중", `입력 배치 ${inputBatch}`, "비용", `최저 대비 (${labels.cost} / ${labels.other})`];
  return (
    <div className="overflow-hidden rounded-xl border border-neutral-200 dark:border-neutral-700">
      <div className="border-b border-neutral-200 px-5 py-4 dark:border-neutral-700">
        <h3 className="text-sm font-semibold text-neutral-800 dark:text-neutral-200">전체 후보</h3>
        <p className="mt-0.5 text-xs text-neutral-500">
          ‘운영 배치’ 열들은 입력하신 배치 {inputBatch}에서 시작해 SLO를 어길 때만 한 단계씩 내린 배치 기준입니다.
          ‘최대 처리량’은 지연 SLO 안에서 처리량이 가장 큰 배치 기준이고, ‘요청량 비중’은 요청량이 그 처리량의 몇
          %인지입니다. ‘입력 배치’는 입력하신 배치 그대로 돌렸을 때의 예측입니다.
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-neutral-50 text-left text-xs text-neutral-500 dark:bg-neutral-900 dark:text-neutral-400">
            <tr>
              {cols.map((c) => (
                <th key={c} className="px-4 py-3">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800">
            {ordered.map((r) => {
              const isPick = r.instance_type === pickName;
              const isSimilar = similar.includes(r.instance_type);
              const s = ratios.get(r.instance_type);
              return (
                <tr key={r.instance_type} className={r.status !== "ok" || r.over_budget ? "text-neutral-400" : undefined}>
                  <td className="px-4 py-3 font-mono">
                    <div className="flex flex-wrap items-center gap-1.5">
                      {r.instance_type}
                      {isPick && (
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
                      <p className="mt-0.5 font-sans text-[11px] text-neutral-400">같은 GPU {r.latency_proxy_of} 예측값 사용</p>
                    )}
                  </td>
                  {r.status === "ok" ? (
                    <>
                      <td className="px-4 py-3 font-mono">
                        {r.operating_batch}
                        {r.operating_batch !== r.input_batch && (
                          <span className="block font-sans text-[11px] text-neutral-400">{batchReason(r, offline)}</span>
                        )}
                      </td>
                      <td className="px-4 py-3 font-mono">{ms(r.latency_ms)}</td>
                      {offline ? (
                        <>
                          <td className="px-4 py-3 font-mono">{rps(r.operating_rps)}</td>
                          <td className="px-4 py-3 font-mono">{money(r.cost_usd_per_million)}</td>
                          <td className="px-4 py-3 font-mono">
                            {r.job_hours != null ? `${hours(r.job_hours)} · ${money(r.job_cost_usd, 2)}` : "—"}
                          </td>
                        </>
                      ) : (
                        <>
                          <td className="px-4 py-3 font-mono">{rps(r.operating_rps)}</td>
                          <td className="px-4 py-3 font-mono">
                            {rps(r.capacity_rps)} · {pct(r.demand_share)}{" "}
                            <span className="text-xs text-neutral-400">(배치 {r.capacity_batch})</span>
                          </td>
                        </>
                      )}
                      <td className="px-4 py-3 font-mono text-xs">
                        <InputBatchNote row={r} compact />
                      </td>
                      <td className="px-4 py-3 font-mono">{costText(r, costUnit)}</td>
                      <td className="px-4 py-3 font-mono text-xs">
                        {s ? `×${s.costRatio.toFixed(2)} / ×${s.otherRatio.toFixed(2)}` : "—"}
                      </td>
                    </>
                  ) : (
                    <td className="px-4 py-3 text-xs" colSpan={cols.length - 1}>
                      {ROW_STATUS_LABEL[r.status] ?? r.status}
                      {r.status === "slo_impossible" && r.min_latency_ms !== undefined && (
                        <> — 최소 지연 {ms(r.min_latency_ms)} (배치 {r.min_latency_batch})</>
                      )}
                      {r.status === "no_prediction" && r.reason && <> — {r.reason}</>}
                      {r.status === "throughput_short" && (
                        <>
                          {" "}— 최대 처리량 {rps(r.capacity_rps)} (배치 {r.capacity_batch}, 지연{" "}
                          {ms(r.capacity_latency_ms)}), 요청량 비중 {pct(r.demand_share)} · {costText(r, costUnit)}
                        </>
                      )}
                      {r.input_batch !== undefined && r.status !== "no_prediction" && (
                        <span className="block">
                          입력 배치 — <InputBatchNote row={r} />
                        </span>
                      )}
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
  onReset,
}: {
  job: JobResponse;
  onJobUpdate: (job: JobResponse) => void;
  onReset: () => void;
}) {
  const rec = job.result?.recommendation;
  const slo = job.slo;
  const purpose: WorkloadPurpose = rec?.policy?.purpose ?? slo?.purpose ?? "online";
  const offline = purpose === "offline";
  const labels = criteriaLabels(purpose);
  const [costUnit, setCostUnit] = useState<CostUnit>("hour");
  // Weight of the cost criterion (0-1); the other criterion gets 1 - this.
  const [costWeight, setCostWeight] = useState(PRESET_COST_WEIGHT.cost);
  const activePreset = presetOf(costWeight);
  const limit = rec?.latency_limit ?? null;
  const maxLatencyMs = limit?.max_latency_ms ?? null;
  const tokens = limit?.basis === "per_token" ? limit.tokens : null;
  const demand = slo?.min_throughput_rps ?? null;
  const totalRequests = slo?.total_requests ?? null;

  const rows = rec?.rows ?? [];
  const scored = scoreRows(rows, costWeight, purpose);
  const pick = scored[0] ?? null;
  // Second card: the best on the criterion the user weighs less, so the
  // trade-off stays visible.
  const compare = scoreRows(rows, costWeight >= 0.5 ? 0 : 1, purpose)[0] ?? null;
  const compareTitle = costWeight >= 0.5 ? `비교 — ${presetLabel("speed", purpose)}라면` : "비교 — 비용 중시라면";
  const budget = job.payload.deployment.budget_usd_per_hour;
  const inputBatch = job.payload.batch_size;

  function pickSubtitle(): string {
    const sloText = offline ? "지연 SLO를 만족하는 인스턴스 중" : "지연·처리량 SLO를 모두 만족하는 인스턴스 중";
    if (activePreset === "cost") return `${sloText} ${offline ? "요청당 비용이 가장 낮습니다" : "가장 저렴합니다"}.`;
    if (activePreset === "speed") return `${sloText} ${offline ? "처리량이 가장 큽니다" : "지연이 가장 짧습니다"}.`;
    return `${labels.cost} ${Math.round(costWeight * 100)} : ${labels.other} ${Math.round(
      (1 - costWeight) * 100
    )} 비중으로 본 최저 대비 배수의 가중합이 가장 작은 인스턴스입니다.`;
  }

  function compareSubtitle(c: ScoredRow, p: ScoredRow): string {
    const cv = criteriaValues(c.row, purpose)!;
    const pv = criteriaValues(p.row, purpose)!;
    return `추천 대비 ${labels.cost} ${relative(cv.cost, pv.cost)}, ${labels.other} ${relative(cv.other, pv.other)}`;
  }

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold text-neutral-900 dark:text-neutral-100">GPU 인스턴스 추천 결과</h2>
          <p className="mt-1 text-sm text-neutral-500">
            {offline
              ? "오프라인 일괄 처리: 인스턴스 한 대로 작업을 돌린다는 가정에서, 지연 SLO 안에서 요청당 비용이 낮은 인스턴스를 추천합니다."
              : "온라인 서빙: 인스턴스 한 대에 올린다는 가정에서, 지연·처리량 SLO를 모두 만족하는 인스턴스를 추천합니다."}
          </p>
          <dl className="mt-3 space-y-1 text-sm text-neutral-600 dark:text-neutral-400">
            <div className="flex gap-2">
              <dt className="font-medium text-neutral-700 dark:text-neutral-300">모델</dt>
              <dd className="font-mono">
                {job.payload.model} · 입력 배치(상한) {inputBatch}
              </dd>
            </div>
            {slo && (
              <div className="flex gap-2">
                <dt className="font-medium text-neutral-700 dark:text-neutral-300">SLO</dt>
                <dd>
                  {describeLatency(slo, job.payload)} · {describeWorkload(slo)}
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
        </div>
      </div>

      {!rec ? (
        <p className="rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:bg-amber-950 dark:text-amber-400">
          추천 결과가 없습니다.
        </p>
      ) : (
        <>
          <StatusNote rec={rec} />

          {rec.rows.length > 0 && (
            <p className="rounded-lg bg-neutral-50 px-4 py-3 text-xs leading-relaxed text-neutral-600 dark:bg-neutral-900 dark:text-neutral-400">
              <b>운영 배치 기준으로 보여 드립니다.</b> 입력하신 배치 {inputBatch}에서 시작해,{" "}
              {offline
                ? "지연 SLO를 넘는 인스턴스만 한 단계씩(128→64→32→16→8→1) 내려 계산했습니다. 아래 배치의 처리량이 더 크면 그 배치를 씁니다."
                : "지연 SLO나 처리량 SLO를 어기는 인스턴스만 한 단계씩(128→64→32→16→8→1) 내려 계산했습니다."}{" "}
              SLO를 지키는 인스턴스는 입력하신 배치 그대로이고, 배치를 내린 인스턴스는 카드와 표의 ‘운영 배치’에 이유를
              표시합니다.
            </p>
          )}

          {pick && slo && (
            <>
              <div className="space-y-2 rounded-xl border border-neutral-200 p-4 dark:border-neutral-700">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-medium text-neutral-500">무엇을 중시하나요?</span>
                  {(["cost", "speed", "balanced"] as const).map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => setCostWeight(PRESET_COST_WEIGHT[p as WeightPreset])}
                      className={`rounded-full border px-3 py-1 text-xs font-medium ${
                        activePreset === p
                          ? "border-neutral-900 bg-neutral-900 text-white dark:border-white dark:bg-white dark:text-neutral-900"
                          : "border-neutral-300 text-neutral-600 dark:border-neutral-700 dark:text-neutral-400"
                      }`}
                    >
                      {presetLabel(p, purpose)}
                    </button>
                  ))}
                  {activePreset === null && (
                    <span className="rounded-full border border-dashed border-neutral-400 px-3 py-1 text-xs text-neutral-500">
                      직접 조정
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3 text-xs text-neutral-500">
                  <span className="w-28 text-right">
                    {labels.cost} {Math.round(costWeight * 100)}%
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={Math.round((1 - costWeight) * 100)}
                    onChange={(e) => setCostWeight(1 - Number(e.target.value) / 100)}
                    className="flex-1 accent-neutral-800 dark:accent-white"
                    aria-label={`${labels.cost}과 ${labels.other}의 비중`}
                  />
                  <span className="w-28">
                    {labels.other} {Math.round((1 - costWeight) * 100)}%
                  </span>
                </div>
                <p className="text-[11px] text-neutral-400">
                  SLO를 모두 만족하는 인스턴스 안에서만 순서를 바꿉니다. 점수 = {labels.cost} 비중 × (최저 대비 배수) +{" "}
                  {labels.other} 비중 × (최저 대비 배수), 낮을수록 위.
                  {offline
                    ? " 처리 시간은 요청 1건당 시간(1 ÷ 처리량)입니다."
                    : " 처리량은 SLO를 통과하면(요청량을 감당하면) 모두 같게 봅니다."}
                </p>
              </div>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <PlanCard
                  title={
                    rec.status === "over_budget"
                      ? `${activePreset ? presetLabel(activePreset, purpose) : "직접 조정"} (예산 초과)`
                      : `추천 — ${activePreset ? presetLabel(activePreset, purpose) : "직접 조정"}`
                  }
                  subtitle={pickSubtitle()}
                  plan={pick.row}
                  purpose={purpose}
                  costUnit={costUnit}
                  maxLatencyMs={maxLatencyMs}
                  tokens={tokens}
                  demand={demand}
                  totalRequests={totalRequests}
                  highlight
                />
                {compare && compare.row.instance_type !== pick.row.instance_type ? (
                  <PlanCard
                    title={compareTitle}
                    subtitle={compareSubtitle(compare, pick)}
                    plan={compare.row}
                    purpose={purpose}
                    costUnit={costUnit}
                    maxLatencyMs={maxLatencyMs}
                    tokens={tokens}
                    demand={demand}
                    totalRequests={totalRequests}
                  />
                ) : (
                  <div className="rounded-xl border border-dashed border-neutral-200 p-5 text-sm text-neutral-500 dark:border-neutral-700">
                    <p className="text-xs font-medium">비교</p>
                    <p className="mt-2">
                      {labels.cost}과 {labels.other} 모두 같은 인스턴스가 가장 좋아, 어느 비중으로 봐도 같은 인스턴스가
                      추천됩니다.
                    </p>
                  </div>
                )}
              </div>
            </>
          )}

          {rec.status === "slo_impossible" && tokens !== null && (
            <p className="text-xs text-neutral-500">
              생성 토큰 수({tokens}개)를 줄이거나 지연 기준을 ‘일반 대화’나 ‘제한 없음’으로 바꾸면 선택지가 생길 수
              있습니다.
            </p>
          )}

          {costWeight === 1 && rec.similar && rec.similar.length > 0 && (
            <p className="text-sm text-neutral-500">
              {offline ? "요청당 비용" : "비용"} 차이 {rec.policy?.similar_pct ?? 10}% 이내의 다른 선택지:{" "}
              <span className="font-mono">{rec.similar.join(", ")}</span>
            </p>
          )}

          {rec.rows.length > 0 && (
            <RowsTable
              rows={rec.rows}
              pickName={pick?.row.instance_type ?? rec.cheapest}
              similar={costWeight === 1 ? rec.similar ?? [] : []}
              purpose={purpose}
              costUnit={costUnit}
              scored={scored}
              inputBatch={inputBatch}
            />
          )}

          <p className="text-xs leading-relaxed text-neutral-400">
            지연 SLO는 T로 판정합니다. T는 예측기가 낸 배치 1회 처리 시간의 P90(보수적 추정)으로, 처리 시간만 봅니다.
            요청이 줄을 서거나 배치가 차기를 기다리는 시간은 포함하지 않고, 운영 배치를 꽉 채운 경우의 시간이라 요청이
            적어 배치가 덜 차면 실제로는 더 짧을 수 있습니다. 생성형 모델의 T는 입력 처리부터 마지막 토큰까지 요청 전체 시간이며, 토큰당 기준(실시간
            50ms, 일반 100ms)은 LLM 서빙 연구에서 대화형 서비스에 쓰는 토큰 간 지연(TPOT) 범위를 따랐습니다.
            {offline
              ? " 요청당 비용은 시간당 요금 ÷ 처리량이고, 완료 예상은 요청 수 ÷ 처리량입니다."
              : " 처리량 SLO는 운영 배치의 예측 처리량이 요청량 이상인지로 판정합니다. 처리량은 예측값이라 실제와 차이가 날 수 있으니, 요청량 비중이 100%에 가까운 인스턴스는 여유가 적다는 점에 유의하세요."}{" "}
            월 비용은 시간당 × 720시간입니다.
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
