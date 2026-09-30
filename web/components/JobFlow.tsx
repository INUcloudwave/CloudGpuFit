"use client";

import { useEffect, useRef, useState } from "react";
import type { InstanceCandidate, JobResponse, PredictPayload, RuntimePrediction, SloInput } from "@/lib/types";
import { errorMessageFor, BREAKDOWN_LABELS } from "@/lib/jobErrors";
import { formatBytesToGb } from "@/lib/format";
import { WORKLOAD_SCHEMAS } from "@/lib/workloadSchema";
import type { SloVerdict } from "@/lib/slo";
import { rankCandidates, DEFAULT_WEIGHTS, type ImportanceWeights } from "@/lib/ranking";

const POLL_INTERVAL_MS = 2000;

interface Props {
  jobId: string;
  onReset: () => void;
}

function ConditionsSummary({ payload, slo }: { payload: PredictPayload; slo: SloInput | null }) {
  const schema = WORKLOAD_SCHEMAS[payload.type];
  const paramEntries = schema.fields
    .filter((f) => f.key !== "batch_size")
    .map((f) => [f.label, payload.params[f.key]] as const)
    .filter(([, value]) => value !== null && value !== undefined);

  return (
    <div className="rounded-lg border border-neutral-200 p-4 text-sm dark:border-neutral-700">
      <div className="mb-2 flex flex-wrap items-baseline gap-2">
        <span className="font-mono font-medium text-neutral-800 dark:text-neutral-200">{payload.model}</span>
        <span className="rounded-full bg-neutral-900 px-2 py-0.5 text-xs font-medium text-white dark:bg-white dark:text-neutral-900">
          {schema.label}
        </span>
      </div>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-neutral-600 dark:text-neutral-400 sm:grid-cols-3">
        <div>
          <dt className="inline text-neutral-400">dtype: </dt>
          <dd className="inline font-mono">{payload.dtype}</dd>
        </div>
        <div>
          <dt className="inline text-neutral-400">batch_size: </dt>
          <dd className="inline font-mono">{payload.batch_size}</dd>
        </div>
        {paramEntries.map(([label, value]) => (
          <div key={label}>
            <dt className="inline text-neutral-400">{label}: </dt>
            <dd className="inline font-mono">{String(value)}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 border-t border-neutral-100 pt-2 text-neutral-500 dark:border-neutral-800">
        리전: {payload.deployment.aws_region} · 예산: ${payload.deployment.budget_usd_per_hour.toFixed(4)}/시간
      </p>
      {slo && (
        <p className="mt-1 text-neutral-500">
          SLO: Latency ≤ {slo.max_latency_ms}ms · Throughput ≥ {slo.min_throughput_rps}req/s
        </p>
      )}
    </div>
  );
}

const SLO_BADGE_STYLE: Record<SloVerdict, string> = {
  pass: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-400",
  fail: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-400",
  unknown: "bg-neutral-100 text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400",
};

const SLO_BADGE_LABEL: Record<SloVerdict, string> = {
  pass: "충족",
  fail: "미충족",
  unknown: "미판정",
};

function SloBadge({ verdict }: { verdict: SloVerdict }) {
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${SLO_BADGE_STYLE[verdict]}`}>
      {SLO_BADGE_LABEL[verdict]}
    </span>
  );
}

function formatMonthlyCost(pricePerHour: number | null): string {
  if (pricePerHour === null) return "비용 미확인";
  const HOURS_PER_MONTH = 720; // 30일 * 24시간, orchestrator 쪽 예산 입력과 동일한 환산 기준
  const monthly = pricePerHour * HOURS_PER_MONTH;
  return `$${monthly.toLocaleString(undefined, { maximumFractionDigits: 2 })} /월`;
}

function RankingView({
  job,
  candidates,
  predictions,
  onReset,
}: {
  job: JobResponse;
  candidates: InstanceCandidate[];
  predictions: RuntimePrediction[];
  onReset: () => void;
}) {
  // Draft slider position vs. the weights actually used for ranking -- the
  // table only re-sorts when "적용" is clicked, not on every drag tick.
  const [draftLatencyShare, setDraftLatencyShare] = useState(DEFAULT_WEIGHTS.latency);
  const [appliedWeights, setAppliedWeights] = useState<ImportanceWeights>(DEFAULT_WEIGHTS);
  const draftThroughputShare = 100 - draftLatencyShare;

  const ranked = rankCandidates(candidates, predictions, job.slo, appliedWeights);

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_260px]">
        <div>
          <h2 className="text-2xl font-semibold text-neutral-900 dark:text-neutral-100">GPU 인스턴스 추천 결과</h2>
          <p className="mt-1 text-sm text-neutral-500">설정한 조건에 맞는 인스턴스를 비교해 보세요.</p>
          <dl className="mt-4 space-y-1 text-sm">
            <div className="flex gap-2">
              <dt className="font-medium text-neutral-700 dark:text-neutral-300">리전</dt>
              <dd className="text-neutral-600 dark:text-neutral-400">{job.payload.deployment.aws_region}</dd>
            </div>
            {job.slo && (
              <div className="flex gap-2">
                <dt className="font-medium text-neutral-700 dark:text-neutral-300">SLO</dt>
                <dd className="text-neutral-600 dark:text-neutral-400">
                  Latency ≤ {job.slo.max_latency_ms}ms · Throughput ≥ {job.slo.min_throughput_rps}req/s
                </dd>
              </div>
            )}
          </dl>
        </div>

        <div className="self-start rounded-lg border border-neutral-200 p-3 dark:border-neutral-700">
          <p className="text-xs font-medium text-neutral-700 dark:text-neutral-300">Importance 조정</p>
          <div className="mt-2 flex items-center justify-between text-xs text-neutral-500">
            <span>Latency</span>
            <span>Throughput</span>
          </div>
          <div className="flex items-baseline justify-between">
            <span className="font-mono text-base font-semibold text-neutral-900 dark:text-neutral-100">
              {(draftLatencyShare / 100).toFixed(2)}
            </span>
            <span className="font-mono text-base font-semibold text-neutral-900 dark:text-neutral-100">
              {(draftThroughputShare / 100).toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={draftLatencyShare}
            onChange={(e) => setDraftLatencyShare(Number(e.target.value))}
            className="mt-1 w-full accent-neutral-800 dark:accent-white"
          />
          <p className="mt-1 text-center text-[11px] text-neutral-400">가중치 합계 1.00</p>
          <button
            type="button"
            onClick={() => setAppliedWeights({ latency: draftLatencyShare, throughput: draftThroughputShare })}
            className="mt-2 w-full rounded-lg bg-neutral-900 px-3 py-1.5 text-xs font-medium text-white dark:bg-white dark:text-neutral-900"
          >
            적용
          </button>
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-neutral-200 dark:border-neutral-700">
        <div className="border-b border-neutral-200 px-5 py-4 dark:border-neutral-700">
          <h3 className="text-sm font-semibold text-neutral-800 dark:text-neutral-200">추천 인스턴스</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-neutral-50 text-left text-xs uppercase text-neutral-500 dark:bg-neutral-900 dark:text-neutral-400">
              <tr>
                <th className="px-5 py-3">순위</th>
                <th className="px-5 py-3">인스턴스</th>
                <th className="px-5 py-3">예상 Latency</th>
                <th className="px-5 py-3">예상 Throughput</th>
                <th className="px-5 py-3">SLO 상태</th>
                <th className="px-5 py-3">예상 비용</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800">
              {ranked.map((r, i) => {
                const p = r.prediction;
                const ok = p?.status === "ok";
                return (
                  <tr key={r.candidate.instance_type}>
                    <td className="px-5 py-3 font-mono text-neutral-400">{i + 1}</td>
                    <td className="px-5 py-3 font-mono">{r.candidate.instance_type}</td>
                    <td className="px-5 py-3 font-mono">
                      {ok && p.predicted_p90_ms !== undefined ? `${p.predicted_p90_ms.toFixed(2)} ms` : "—"}
                    </td>
                    <td className="px-5 py-3 font-mono">
                      {ok && p.predicted_throughput_rps !== undefined
                        ? `${p.predicted_throughput_rps.toFixed(2)} req/s`
                        : "—"}
                    </td>
                    <td className="px-5 py-3">
                      <SloBadge verdict={r.sloVerdict} />
                    </td>
                    <td className="px-5 py-3 font-mono">{formatMonthlyCost(r.candidate.price_usd_per_hour)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

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

/** Explains why there are 0 (or partial) runtime predictions — a normal
 * filter/adapter outcome (§10/§17), not necessarily an error to alarm over. */
function RuntimeStatusNote({ status, message }: { status: string; message?: string }) {
  const text: Record<string, string> = {
    no_candidates:
      "조건(VRAM/리전/예산)을 만족하는 AWS 인스턴스 후보가 없습니다. 인스턴스 가격 데이터가 아직 등록되지 않았을 수 있습니다.",
    no_installed_targets: "후보 인스턴스에 대해 설치된 eager_runtime 프로파일이 없습니다.",
    skipped: message ?? "이 요청 조합은 현재 Runtime 예측 대상이 아닙니다.",
    error: message ?? "Runtime 예측 중 오류가 발생했습니다.",
  };
  const body = text[status];
  if (!body) return null;
  return (
    <p className="rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:bg-amber-950 dark:text-amber-400">
      {body}
    </p>
  );
}

export default function JobFlow({ jobId, onReset }: Props) {
  const [job, setJob] = useState<JobResponse | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  // Distinct from fetchError: a 404 means this job_id is gone for good (e.g.
  // jobs.db was reset) -- retrying forever can't fix that, so polling stops
  // and the UI offers a way out instead of spinning silently.
  const [notFound, setNotFound] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [showRanking, setShowRanking] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function poll() {
      try {
        const res = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
        const data = (await res.json()) as JobResponse;
        if (cancelled) return;

        if (res.status === 404) {
          setNotFound(true);
          return;
        }

        if (!res.ok) {
          setFetchError("작업 상태를 불러오지 못했습니다.");
          timer = setTimeout(poll, POLL_INTERVAL_MS);
          return;
        }

        setFetchError(null);
        setJob(data);
        if (data.status === "queued" || data.status === "running") {
          timer = setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch {
        if (!cancelled) {
          setFetchError("작업 상태를 불러오지 못했습니다.");
          timer = setTimeout(poll, POLL_INTERVAL_MS);
        }
      }
    }

    poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  const runningSince = useRef<number | null>(null);
  useEffect(() => {
    if (job?.status !== "running") {
      runningSince.current = null;
      return;
    }
    if (runningSince.current === null) runningSince.current = Date.now();
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - (runningSince.current ?? Date.now())) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [job?.status]);

  if (!job) {
    if (notFound) {
      return (
        <div className="mx-auto w-full max-w-xl space-y-4 text-center">
          <p className="text-sm text-neutral-500">
            이 작업을 찾을 수 없습니다. 삭제되었거나 만료된 job_id입니다.
          </p>
          <button
            type="button"
            onClick={onReset}
            className="rounded-lg bg-neutral-900 px-5 py-2.5 text-sm font-medium text-white dark:bg-white dark:text-neutral-900"
          >
            새 예측 시작
          </button>
        </div>
      );
    }
    return (
      <div className="mx-auto w-full max-w-xl text-center text-sm text-neutral-500">
        {fetchError ?? "작업 정보를 불러오는 중…"}
      </div>
    );
  }

  if (job.status === "queued" || job.status === "running") {
    return (
      <div className="mx-auto w-full max-w-xl space-y-4">
        <ConditionsSummary payload={job.payload} slo={job.slo} />
        <div className="space-y-4 text-center">
          <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-neutral-300 border-t-neutral-800 dark:border-neutral-700 dark:border-t-white" />
          <p className="text-sm text-neutral-700 dark:text-neutral-300">
            {job.status === "queued" ? "대기 중… (앞선 작업 완료를 기다리는 중)" : job.progress ?? "CPU 프로파일링 실행 중…"}
          </p>
          {job.status === "running" && <p className="text-xs text-neutral-500">{elapsed}초 경과</p>}
          <p className="text-xs text-neutral-500">
            네트워크가 아니라 모델 크기·배치 크기·시퀀스 길이에 비례해 CPU 연산 시간이 늘어납니다 — 큰 모델은 수십
            분 걸릴 수 있습니다.
          </p>
        </div>
      </div>
    );
  }

  if (job.status === "failed") {
    const error = job.error ?? { code: "internal" as const, message: "" };
    return (
      <div className="mx-auto w-full max-w-xl space-y-4">
        <ConditionsSummary payload={job.payload} slo={job.slo} />
        <p className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-400">
          {errorMessageFor(error)}
        </p>
        {error.code === "internal" && error.message && (
          <details className="text-xs text-neutral-500">
            <summary className="cursor-pointer">상세 로그</summary>
            <pre className="mt-2 overflow-x-auto whitespace-pre-wrap rounded-lg bg-neutral-900 p-3 text-neutral-100 dark:bg-black">
              {error.message}
            </pre>
          </details>
        )}
        <button
          type="button"
          onClick={onReset}
          className="rounded-lg bg-neutral-900 px-5 py-2.5 text-sm font-medium text-white dark:bg-white dark:text-neutral-900"
        >
          새 예측 시작
        </button>
      </div>
    );
  }

  const result = job.result!;
  const { vram, candidates, catalog_error, runtime } = result;

  if (showRanking) {
    return (
      <RankingView job={job} candidates={candidates} predictions={runtime.predictions} onReset={onReset} />
    );
  }

  return (
    <div className="mx-auto w-full max-w-xl space-y-6">
      <ConditionsSummary payload={job.payload} slo={job.slo} />

      {job.cached && (
        <span className="inline-block rounded-full bg-blue-100 px-2.5 py-1 text-xs font-medium text-blue-800 dark:bg-blue-950 dark:text-blue-300">
          이전 실행 결과를 재사용했습니다
        </span>
      )}

      <div className="rounded-xl bg-neutral-900 p-6 text-center dark:bg-white">
        <p className="text-xs text-neutral-400 dark:text-neutral-600">required_vram_gb</p>
        <p className="text-4xl font-bold text-white dark:text-neutral-900">{vram.required_vram_gb} GB</p>
      </div>

      <details className="rounded-lg border border-neutral-200 p-3 dark:border-neutral-700">
        <summary className="cursor-pointer text-sm font-medium text-neutral-700 dark:text-neutral-300">
          VRAM 예측 상세 내역
        </summary>
        <dl className="mt-3 space-y-1.5 text-sm">
          {Object.entries(vram.breakdown_bytes).map(([key, value]) => (
            <div key={key} className="flex justify-between gap-4">
              <dt className="text-neutral-500">{BREAKDOWN_LABELS[key] ?? key}</dt>
              <dd className="font-mono text-neutral-800 dark:text-neutral-200">
                {value === null ? "해당 없음" : formatBytesToGb(value)}
              </dd>
            </div>
          ))}
        </dl>
      </details>

      {vram.warnings.length > 0 && (
        <div className="space-y-1 rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:bg-amber-950 dark:text-amber-400">
          {vram.warnings.map((w, i) => (
            <p key={i}>{w}</p>
          ))}
        </div>
      )}

      {candidates.length > 0 ? (
        <>
          <button
            type="button"
            onClick={() => setShowRanking(true)}
            className="w-full rounded-lg bg-neutral-900 px-5 py-2.5 text-sm font-medium text-white dark:bg-white dark:text-neutral-900"
          >
            추천 순위 결과 보러가기
          </button>
          {runtime.status !== "ok" && (
            <RuntimeStatusNote status={runtime.status} message={runtime.error_message} />
          )}
        </>
      ) : catalog_error ? (
        // A real fetch failure, not "nothing fit" -- shown instead of the
        // generic no_candidates note so a network/AWS-side problem isn't
        // mistaken for "this region just doesn't sell these instances".
        <p className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-400">
          AWS 가격 조회에 실패했습니다: {catalog_error}
        </p>
      ) : (
        <RuntimeStatusNote status={runtime.status} message={runtime.error_message} />
      )}
    </div>
  );
}
