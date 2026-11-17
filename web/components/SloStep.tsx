"use client";

import { useState } from "react";
import type { CreateJobResponse } from "@/lib/types";

interface Props {
  jobId: string;
  onSubmitted: () => void;
}

/** Shown right after job creation, before the progress/result screen. The
 * numbers entered here never feed prediction (already running in the
 * background). max_latency_ms is the latency bar each instance must meet;
 * min_throughput_rps is the demand the whole configuration must carry -- it
 * sets how many units of the recommended instance are needed. */
export default function SloStep({ jobId, onSubmitted }: Props) {
  const [maxLatencyRaw, setMaxLatencyRaw] = useState("100");
  const [minThroughputRaw, setMinThroughputRaw] = useState("10");
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
        <p className="text-sm font-semibold text-neutral-800 dark:text-neutral-200">SLO 입력</p>
        <p className="mt-1 text-xs text-neutral-500">
          추천 구성을 고르는 기준입니다. 성능 예측은 이 값과 무관하게 이미 백그라운드에서 진행 중이며, 결과
          화면에서 추천이 이 값으로 계산됩니다.
        </p>
      </div>

      <div className="space-y-1">
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">
          최대 허용 지연 (max_latency_ms)
        </label>
        <p className="text-xs text-neutral-500">요청 하나가 처리되는 데 허용하는 최대 시간입니다.</p>
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
          예상 수요 (min_throughput_rps)
        </label>
        <p className="text-xs text-neutral-500">
          서비스 전체에 들어올 초당 요청 수입니다. 인스턴스 한 대가 감당하지 못하면 같은 인스턴스를 여러 대로 나눠
          처리하도록 대수를 계산합니다.
        </p>
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
