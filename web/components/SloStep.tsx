"use client";

import { useState } from "react";
import type { CreateJobResponse, LatencyPreset, PredictPayload, SloInput, WorkloadPurpose } from "@/lib/types";
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

/** Throughput SLO calculator. Two standard ways to turn what a user knows
 * into requests/s; the result fills the throughput input, which stays
 * editable. Neither feeds prediction.
 *  - Concurrent users (interactive services): throughput = N / Z, where Z is
 *    how often one user sends a new request (their whole cycle: waiting for
 *    the answer + reading + typing). This is the interactive response time
 *    law, a form of Little's law (L = lambda x W).
 *  - Daily requests: throughput = daily / 86,400 s x peak factor (peak hour
 *    vs. the daily average; service-specific, so the user supplies it). */
function ThroughputHelper({ onApply }: { onApply: (rps: number) => void }) {
  const [mode, setMode] = useState<"users" | "daily">("users");
  const [usersRaw, setUsersRaw] = useState("100");
  const [intervalRaw, setIntervalRaw] = useState("60");
  const [dailyRaw, setDailyRaw] = useState("100000");
  const [peakRaw, setPeakRaw] = useState("3");

  const users = Number(usersRaw);
  const interval = Number(intervalRaw);
  const daily = Number(dailyRaw);
  const peak = Number(peakRaw);
  const result =
    mode === "users"
      ? users > 0 && interval > 0
        ? users / interval
        : null
      : daily > 0 && peak >= 1
        ? (daily / 86400) * peak
        : null;
  const shown = result === null ? null : Math.round(result * 100) / 100;

  return (
    <details className="rounded-lg border border-neutral-200 p-3 text-sm dark:border-neutral-700">
      <summary className="cursor-pointer text-xs font-medium text-neutral-700 dark:text-neutral-300">
        초당 요청 수를 모르겠다면 — 계산 도우미
      </summary>
      <div className="mt-3 space-y-3">
        <div className="inline-flex rounded-lg border border-neutral-300 p-0.5 text-xs dark:border-neutral-700">
          {(
            [
              ["users", "동시 사용자로"],
              ["daily", "하루 요청 수로"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setMode(id)}
              className={`rounded-md px-3 py-1.5 font-medium ${
                mode === id
                  ? "bg-neutral-900 text-white dark:bg-white dark:text-neutral-900"
                  : "text-neutral-600 dark:text-neutral-400"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {mode === "users" ? (
          <div className="grid grid-cols-2 gap-3">
            <label className="space-y-1">
              <span className="block text-xs text-neutral-600 dark:text-neutral-400">동시 사용자 수 (명)</span>
              <input type="number" min={0} value={usersRaw} onChange={(e) => setUsersRaw(e.target.value)} className={INPUT_CLASS} />
            </label>
            <label className="space-y-1">
              <span className="block text-xs text-neutral-600 dark:text-neutral-400">한 사람의 요청 간격 (초)</span>
              <input type="number" min={0} value={intervalRaw} onChange={(e) => setIntervalRaw(e.target.value)} className={INPUT_CLASS} />
            </label>
            <p className="col-span-2 text-xs text-neutral-500">
              요청 간격 = 한 사람이 질문을 보내고, 답을 받아 읽고, 다음 질문을 보내기까지의 시간. 초당 요청 수 =
              동시 사용자 수 ÷ 요청 간격 (Little의 법칙).
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            <label className="space-y-1">
              <span className="block text-xs text-neutral-600 dark:text-neutral-400">하루 요청 수 (건)</span>
              <input type="number" min={0} value={dailyRaw} onChange={(e) => setDailyRaw(e.target.value)} className={INPUT_CLASS} />
            </label>
            <label className="space-y-1">
              <span className="block text-xs text-neutral-600 dark:text-neutral-400">피크 배율 (배)</span>
              <input type="number" min={1} step="0.5" value={peakRaw} onChange={(e) => setPeakRaw(e.target.value)} className={INPUT_CLASS} />
            </label>
            <p className="col-span-2 text-xs text-neutral-500">
              피크 배율 = 가장 바쁜 시간의 요청량이 하루 평균의 몇 배인지 (서비스마다 다름). 초당 요청 수 = 하루
              요청 수 ÷ 86,400초 × 피크 배율.
            </p>
          </div>
        )}

        <div className="flex items-center justify-between gap-3 rounded-lg bg-neutral-50 px-3 py-2 dark:bg-neutral-900">
          <span className="text-xs text-neutral-600 dark:text-neutral-400">
            {shown === null
              ? mode === "users"
                ? "0보다 큰 값을 입력하세요."
                : "하루 요청 수는 0보다, 피크 배율은 1 이상이어야 합니다."
              : `→ 초당 약 ${shown.toLocaleString()}건`}
          </span>
          <button
            type="button"
            disabled={shown === null || shown <= 0}
            onClick={() => shown && onApply(shown)}
            className="rounded-md bg-neutral-900 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40 dark:bg-white dark:text-neutral-900"
          >
            이 값 사용
          </button>
        </div>
      </div>
    </details>
  );
}

/** Shown right after job creation, before the progress/result screen. The
 * numbers entered here never feed prediction (already running in the
 * background); the recommendation is computed from them on the result
 * screen. Latency: Generate workloads choose a per-output-token limit (the
 * request limit is that x max_new_tokens, see lib/sloPresets.ts), others a
 * per-request limit; both can be "no limit". Purpose: online serving takes a
 * throughput SLO (the demand the single recommended instance must carry within
 * its target load); offline batch processing takes the amount of work
 * (optional) instead, and the instance runs at its largest-throughput batch. */
export default function SloStep({ jobId, payload, onSubmitted }: Props) {
  const tokens = generatedTokens(payload);
  const isGenerate = tokens !== null;
  const [preset, setPreset] = useState<LatencyPreset>(isGenerate ? "standard" : "custom");
  const [customRaw, setCustomRaw] = useState(isGenerate ? "100" : "100");
  const [purpose, setPurpose] = useState<WorkloadPurpose>("online");
  const [minThroughputRaw, setMinThroughputRaw] = useState("10");
  const [totalRequestsRaw, setTotalRequestsRaw] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const custom = Number(customRaw);
  const customValid = Number.isFinite(custom) && custom > 0;
  const minThroughput = Number(minThroughputRaw);
  const minThroughputValid = Number.isFinite(minThroughput) && minThroughput > 0;
  const totalRequests = Number(totalRequestsRaw);
  const totalRequestsValid = totalRequestsRaw.trim() === "" || (Number.isFinite(totalRequests) && totalRequests > 0);
  const latencyValid = preset !== "custom" || customValid;
  const valid = latencyValid && (purpose === "online" ? minThroughputValid : totalRequestsValid);

  function buildSlo(): SloInput {
    const base: SloInput =
      purpose === "online"
        ? { purpose, min_throughput_rps: minThroughput, latency_preset: preset }
        : {
            purpose,
            total_requests: totalRequestsRaw.trim() === "" ? null : totalRequests,
            latency_preset: preset,
          };
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
        <p className="text-sm font-medium text-neutral-700 dark:text-neutral-300">워크로드 목적</p>
        <OptionCard
          selected={purpose === "online"}
          onSelect={() => setPurpose("online")}
          title="온라인 서빙"
          description="요청이 계속 들어오는 서비스(챗봇, API 등). 입력한 배치부터 시작해 지연·처리량 SLO를 어길 때만 배치를 내리고, 시간당 비용과 응답 시간으로 비교합니다."
        />
        <OptionCard
          selected={purpose === "offline"}
          onSelect={() => setPurpose("offline")}
          title="오프라인 일괄 처리"
          description="정해진 양의 데이터를 한꺼번에 처리하는 작업(대량 분류·요약 등). 입력한 배치부터 시작해 지연 SLO를 어길 때만 배치를 내리고(아래 배치의 처리량이 더 크면 그 배치), 요청당 비용과 처리량으로 비교합니다."
        />
      </div>

      <div className="space-y-2">
        <p className="text-sm font-medium text-neutral-700 dark:text-neutral-300">지연 SLO</p>
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
          description="응답 시간이 중요하지 않은 경우. 입력한 배치까지 모든 배치를 쓸 수 있습니다."
        />
        {selectedPerToken !== null && tokens !== null && (
          <p className="rounded-lg bg-neutral-50 px-3 py-2 text-xs text-neutral-600 dark:bg-neutral-900 dark:text-neutral-400">
            → 요청당 {(selectedPerToken * tokens).toLocaleString()}ms 이내 (토큰당 {selectedPerToken}ms × {tokens}토큰,
            입력 처리 시간 포함)
          </p>
        )}
      </div>

      {purpose === "online" ? (
      <div className="space-y-1">
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">
          처리량 SLO (초당 요청 수)
        </label>
        <p className="text-xs text-neutral-500">
          서비스 전체에 1초 동안 들어올 요청 수입니다. 추천하는 인스턴스 한 대의 예측 처리량이 이 요청량
          이상이어야 합니다.
          {isGenerate
            ? ` 요청 1건 = 프롬프트 1개에 답변 ${tokens}토큰 생성.`
            : " 요청 1건 = 입력 1개(예: 이미지 1장) 처리."}
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
        <ThroughputHelper onApply={(v) => setMinThroughputRaw(String(v))} />
      </div>
      ) : (
      <div className="space-y-1">
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">
          처리할 요청 수 (선택)
        </label>
        <p className="text-xs text-neutral-500">
          작업 전체의 요청 수입니다. 입력하면 인스턴스마다 완료까지 걸리는 시간과 비용을 함께 보여 드립니다. 비워
          두면 요청당 비용으로만 비교합니다.
          {isGenerate
            ? ` 요청 1건 = 프롬프트 1개에 답변 ${tokens}토큰 생성.`
            : " 요청 1건 = 입력 1개(예: 이미지 1장) 처리."}
        </p>
        <input
          type="number"
          min={0}
          step="1"
          placeholder="예: 1000000"
          value={totalRequestsRaw}
          onChange={(e) => setTotalRequestsRaw(e.target.value)}
          className={INPUT_CLASS}
        />
        {!totalRequestsValid && <p className="text-xs text-red-600">0보다 큰 값을 입력하거나 비워 두세요.</p>}
      </div>
      )}

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
