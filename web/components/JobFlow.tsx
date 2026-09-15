"use client";

import { useEffect, useRef, useState } from "react";
import type { JobResponse, PredictPayload } from "@/lib/types";
import { errorMessageFor, BREAKDOWN_LABELS } from "@/lib/jobErrors";
import { formatBytesToGb } from "@/lib/format";
import { WORKLOAD_SCHEMAS } from "@/lib/workloadSchema";

const POLL_INTERVAL_MS = 2000;

interface Props {
  jobId: string;
  onReset: () => void;
}

function ConditionsSummary({ payload }: { payload: PredictPayload }) {
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
    </div>
  );
}

export default function JobFlow({ jobId, onReset }: Props) {
  const [job, setJob] = useState<JobResponse | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function poll() {
      try {
        const res = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
        const data = (await res.json()) as JobResponse;
        if (cancelled) return;

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
    return (
      <div className="mx-auto w-full max-w-xl text-center text-sm text-neutral-500">
        {fetchError ?? "작업 정보를 불러오는 중…"}
      </div>
    );
  }

  if (job.status === "queued" || job.status === "running") {
    return (
      <div className="mx-auto w-full max-w-xl space-y-4">
        <ConditionsSummary payload={job.payload} />
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
        <ConditionsSummary payload={job.payload} />
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
  return (
    <div className="mx-auto w-full max-w-xl space-y-6">
      <ConditionsSummary payload={job.payload} />

      {job.cached && (
        <span className="inline-block rounded-full bg-blue-100 px-2.5 py-1 text-xs font-medium text-blue-800 dark:bg-blue-950 dark:text-blue-300">
          이전 실행 결과를 재사용했습니다
        </span>
      )}

      <div className="rounded-xl bg-neutral-900 p-6 text-center dark:bg-white">
        <p className="text-xs text-neutral-400 dark:text-neutral-600">required_vram_gb</p>
        <p className="text-4xl font-bold text-white dark:text-neutral-900">{result.required_vram_gb} GB</p>
      </div>

      <details className="rounded-lg border border-neutral-200 p-3 dark:border-neutral-700">
        <summary className="cursor-pointer text-sm font-medium text-neutral-700 dark:text-neutral-300">
          상세 내역
        </summary>
        <dl className="mt-3 space-y-1.5 text-sm">
          {Object.entries(result.breakdown_bytes).map(([key, value]) => (
            <div key={key} className="flex justify-between gap-4">
              <dt className="text-neutral-500">{BREAKDOWN_LABELS[key] ?? key}</dt>
              <dd className="font-mono text-neutral-800 dark:text-neutral-200">
                {value === null ? "해당 없음" : formatBytesToGb(value)}
              </dd>
            </div>
          ))}
        </dl>
      </details>

      {result.warnings.length > 0 && (
        <div className="space-y-1 rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:bg-amber-950 dark:text-amber-400">
          {result.warnings.map((w, i) => (
            <p key={i}>{w}</p>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between">
        <button type="button" onClick={onReset} className="text-sm text-neutral-500 underline underline-offset-2">
          새 예측 시작
        </button>
        <button
          type="button"
          disabled
          title="Phase 3에서 구현"
          className="rounded-lg bg-neutral-900 px-5 py-2.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40 dark:bg-white dark:text-neutral-900"
        >
          다음: 인스턴스 후보 조회
        </button>
      </div>
    </div>
  );
}
