"use client";

import { useState } from "react";
import type { CreateJobResponse, LatencyPreset, PredictPayload, SloInput } from "@/lib/types";
import { GENERATE_LATENCY_PRESETS, generatedTokens } from "@/lib/sloPresets";

interface Props {
  jobId: string;
  payload: PredictPayload;
  onSubmitted: () => void;
}

const INPUT_CLASS =
  "w-full rounded-lg border border-neutral-300 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-500 focus:ring-2 focus:ring-neutral-200 dark:border-neutral-700 dark:bg-neutral-900 dark:focus:ring-neutral-700";

function OptionCard({
  selected,
  onSelect,
  title,
  description,
  children,
}: {
  selected: boolean;
  onSelect: () => void;
  title: string;
  description: string;
  children?: React.ReactNode;
}) {
  return (
    <label
      className={`block cursor-pointer rounded-lg border p-3 ${
        selected
          ? "border-neutral-900 bg-neutral-50 dark:border-white dark:bg-neutral-900"
          : "border-neutral-200 dark:border-neutral-700"
      }`}
    >
      <div className="flex items-start gap-2">
        <input type="radio" checked={selected} onChange={onSelect} className="mt-1 accent-neutral-800 dark:accent-white" />
        <div className="flex-1">
          <p className="text-sm font-medium text-neutral-800 dark:text-neutral-200">{title}</p>
          <p className="text-xs text-neutral-500">{description}</p>
          {selected && children}
        </div>
      </div>
    </label>
  );
}

/** Shown right after job creation, before the progress/result screen. The
 * numbers entered here never feed prediction (already running in the
 * background); the recommendation is computed from them on the result
 * screen. Latency: Generate workloads choose a per-output-token limit (the
 * request limit is that x max_new_tokens, see lib/sloPresets.ts), others a
 * per-request limit; both can be "no limit". Throughput: the demand the whole
 * single recommended instance must carry within its target utilization. */
export default function SloStep({ jobId, payload, onSubmitted }: Props) {
  const tokens = generatedTokens(payload);
  const isGenerate = tokens !== null;
  const [preset, setPreset] = useState<LatencyPreset>(isGenerate ? "standard" : "custom");
  const [customRaw, setCustomRaw] = useState(isGenerate ? "100" : "100");
  const [minThroughputRaw, setMinThroughputRaw] = useState("10");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const custom = Number(customRaw);
  const customValid = Number.isFinite(custom) && custom > 0;
  const minThroughput = Number(minThroughputRaw);
  const minThroughputValid = Number.isFinite(minThroughput) && minThroughput > 0;
  const latencyValid = preset !== "custom" || customValid;
  const valid = latencyValid && minThroughputValid;

  function buildSlo(): SloInput {
    const base: SloInput = { min_throughput_rps: minThroughput, latency_preset: preset };
    if (preset === "none") return base;
    if (isGenerate) {
      const spec = GENERATE_LATENCY_PRESETS.find((p) => p.id === preset);
      return { ...base, latency_per_token_ms: spec ? spec.msPerToken : custom };
    }
    return { ...base, max_latency_ms: custom };
  }

  async function handleSubmit() {
    if (!valid) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`/api/jobs/${jobId}/slo`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildSlo()),
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

  const selectedPerToken = isGenerate
    ? GENERATE_LATENCY_PRESETS.find((p) => p.id === preset)?.msPerToken ?? (preset === "custom" && customValid ? custom : null)
    : null;

  return (
    <div className="mx-auto w-full max-w-xl space-y-6">
      <div>
        <p className="text-sm font-semibold text-neutral-800 dark:text-neutral-200">SLO 입력</p>
        <p className="mt-1 text-xs text-neutral-500">
          추천 구성을 고르는 기준입니다. 성능 예측은 이 값과 무관하게 이미 백그라운드에서 진행 중이며, 결과
          화면에서 추천이 이 값으로 계산됩니다.
        </p>
      </div>

      <div className="space-y-2">
        <p className="text-sm font-medium text-neutral-700 dark:text-neutral-300">허용 지연</p>
        {isGenerate ? (
          <>
            <p className="text-xs text-neutral-500">
              생성형 모델은 응답을 한 토큰씩 내보내므로 <b>토큰 하나당 걸리는 시간</b>으로 정합니다. 요청 전체
              허용 시간은 토큰당 시간 × 생성 토큰 수({tokens}개)입니다.
            </p>
            {GENERATE_LATENCY_PRESETS.map((p) => (
              <OptionCard
                key={p.id}
                selected={preset === p.id}
                onSelect={() => setPreset(p.id)}
                title={p.label}
                description={p.description}
              />
            ))}
          </>
        ) : (
          <p className="text-xs text-neutral-500">요청 하나를 처리하는 데 허용하는 최대 시간입니다.</p>
        )}
        <OptionCard
          selected={preset === "custom"}
          onSelect={() => setPreset("custom")}
          title="직접 입력"
          description={isGenerate ? "토큰당 허용 시간(ms)을 직접 정합니다." : "요청당 허용 시간(ms)을 정합니다."}
        >
          <div className="mt-2 flex items-center gap-2">
            <input
              type="number"
              min={0}
              step="1"
              value={customRaw}
              onChange={(e) => setCustomRaw(e.target.value)}
              className={INPUT_CLASS}
            />
            <span className="whitespace-nowrap text-xs text-neutral-500">{isGenerate ? "ms / 토큰" : "ms / 요청"}</span>
          </div>
          {!customValid && <p className="mt-1 text-xs text-red-600">0보다 큰 값을 입력하세요.</p>}
        </OptionCard>
        <OptionCard
          selected={preset === "none"}
          onSelect={() => setPreset("none")}
          title="제한 없음"
          description="배치·오프라인 처리처럼 응답 시간이 중요하지 않은 경우. 수요만 맞추는 가장 저렴한 구성을 찾습니다."
        />
        {selectedPerToken !== null && tokens !== null && (
          <p className="rounded-lg bg-neutral-50 px-3 py-2 text-xs text-neutral-600 dark:bg-neutral-900 dark:text-neutral-400">
            → 요청당 {(selectedPerToken * tokens).toLocaleString()}ms 이내 (토큰당 {selectedPerToken}ms × {tokens}토큰,
            입력 처리 시간 포함)
          </p>
        )}
      </div>

      <div className="space-y-1">
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">
          예상 수요 (초당 요청 수)
        </label>
        <p className="text-xs text-neutral-500">
          서비스 전체에 들어올 초당 요청 수입니다. 추천하는 인스턴스 한 대가 이 요청량을 최대 처리량의
          80%(목표 사용률) 안에서 감당해야 합니다.
        </p>
        <input
          type="number"
          min={0}
          step="0.1"
          value={minThroughputRaw}
          onChange={(e) => setMinThroughputRaw(e.target.value)}
          className={INPUT_CLASS}
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
