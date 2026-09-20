"use client";

import { useState } from "react";
import type { CreateJobResponse } from "@/lib/types";

interface Props {
  jobId: string;
  onSubmitted: () => void;
}

/** Shown right after job creation, before the progress/result screen. The
 * numbers entered here are an acceptance bar for the result, not a workload
 * input — prediction is already running in the background regardless of
 * what's typed here or whether this screen has been submitted yet. */
export default function SloStep({ jobId, onSubmitted }: Props) {
  const [maxLatencyRaw, setMaxLatencyRaw] = useState("500");
  const [minThroughputRaw, setMinThroughputRaw] = useState("1");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const maxLatency = Number(maxLatencyRaw);
  const minThroughput = Number(minThroughputRaw);
  const maxLatencyValid = Number.isFinite(maxLatency) && maxLatency > 0;
  const minThroughputValid = Number.isFinite(minThroughput) && minThroughput > 0;
  const valid = maxLatencyValid && minThroughputValid;

  async function handleSubmit() {
    if (!valid) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`/api/jobs/${jobId}/slo`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ max_latency_ms: maxLatency, min_throughput_rps: minThroughput }),
      });
      const data = (await res.json()) as CreateJobResponse;
      if (!res.ok) {
        setError(data.message ?? "SLO 저장에 실패했습니다.");
        return;
      }
      onSubmitted();
    } catch {
      setError("오케스트레이터에 연결할 수 없습니다.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto w-full max-w-xl space-y-6">
      <div>
        <p className="text-sm font-semibold text-neutral-800 dark:text-neutral-200">SLO(허용 기준) 입력</p>
        <p className="mt-1 text-xs text-neutral-500">
          여기 입력한 값은 결과를 평가하는 허용 기준입니다. 작업 부하가 아니며, 예측 계산은 이미 이 값과 무관하게
          백그라운드에서 진행되고 있습니다.
        </p>
      </div>

      <div className="space-y-1">
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">
          max_latency_ms — 허용 최대 p90 Latency
        </label>
        <input
          type="number"
          min={0}
          step="1"
          value={maxLatencyRaw}
          onChange={(e) => setMaxLatencyRaw(e.target.value)}
          className="w-full rounded-lg border border-neutral-300 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-500 focus:ring-2 focus:ring-neutral-200 dark:border-neutral-700 dark:bg-neutral-900 dark:focus:ring-neutral-700"
        />
        {!maxLatencyValid && <p className="text-xs text-red-600">0보다 큰 값을 입력하세요.</p>}
      </div>

      <div className="space-y-1">
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">
          min_throughput_rps — 최소 허용 Throughput
        </label>
        <input
          type="number"
          min={0}
          step="0.1"
          value={minThroughputRaw}
          onChange={(e) => setMinThroughputRaw(e.target.value)}
          className="w-full rounded-lg border border-neutral-300 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-500 focus:ring-2 focus:ring-neutral-200 dark:border-neutral-700 dark:bg-neutral-900 dark:focus:ring-neutral-700"
        />
        {!minThroughputValid && <p className="text-xs text-red-600">0보다 큰 값을 입력하세요.</p>}
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <div className="flex justify-end">
        <button
          type="button"
          disabled={submitting || !valid}
          onClick={handleSubmit}
          className="rounded-lg bg-neutral-900 px-5 py-2.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40 dark:bg-white dark:text-neutral-900"
        >
          {submitting ? "저장 중…" : "다음"}
        </button>
      </div>
    </div>
  );
}
