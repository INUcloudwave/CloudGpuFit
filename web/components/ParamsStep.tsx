"use client";

import { useState } from "react";
import type { BudgetPeriod, CreateJobResponse, Dtype, ModelConfigFields, WorkloadType } from "@/lib/types";
import { AWS_REGIONS, DEFAULT_AWS_REGION } from "@/lib/awsRegions";
import {
  BUDGET_HOURS_PER_MONTH,
  DTYPE_FIELD,
  GENERATE_MODE_WARNING,
  WORKLOAD_SCHEMAS,
  buildPayload,
  collectFieldErrors,
  defaultResolution,
  defaultValuesFor,
  isResolutionLocked,
  type FieldSpec,
} from "@/lib/workloadSchema";

type SetupState = "idle" | "provisioning" | "ready" | "error";

interface Props {
  modelId: string;
  type: Exclude<WorkloadType, "unsupported">;
  config: ModelConfigFields;
  rawConfig: Record<string, unknown>;
  onBack: () => void;
  xmemHealthy: boolean | null;
  setupState: SetupState;
  setupErrorMessage: string | null;
  onJobCreated: (jobId: string) => void;
}

// Mirrors orchestrator/main.py SUPPORTED_TYPES — the types every stage of
// the pipeline can actually run, not just the ones with an input form.
const SUPPORTED_FOR_EXECUTION: WorkloadType[] = ["decoder_llm", "vision_cls", "text_encdec"];

export default function ParamsStep({
  modelId,
  type,
  config,
  rawConfig,
  onBack,
  xmemHealthy,
  setupState,
  setupErrorMessage,
  onJobCreated,
}: Props) {
  const schema = WORKLOAD_SCHEMAS[type];
  const [dtype, setDtype] = useState<Dtype>(DTYPE_FIELD.default as Dtype);
  const [values, setValues] = useState<Record<string, unknown>>(() => defaultValuesFor(type, config));

  const [awsRegion, setAwsRegion] = useState(DEFAULT_AWS_REGION);
  // Raw text, not a number: a controlled number input bound to a coerced
  // number re-renders "0" back in on every keystroke, so clearing the field
  // to type a fresh value produces "0500" instead of "500".
  const [budgetAmountRaw, setBudgetAmountRaw] = useState("500");
  const [budgetPeriod, setBudgetPeriod] = useState<BudgetPeriod>("monthly");
  const budgetAmount = Number(budgetAmountRaw);

  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const resolutionLocked = type === "vision_cls" && isResolutionLocked(config);

  const deploymentInput = { aws_region: awsRegion, budget_amount: budgetAmount, budget_period: budgetPeriod };
  const perHourPreview = budgetPeriod === "monthly" ? budgetAmount / BUDGET_HOURS_PER_MONTH : budgetAmount;
  const budgetValid = Number.isFinite(budgetAmount) && budgetAmount > 0;
  const fieldErrors = collectFieldErrors(type, values);
  const valuesValid = Object.keys(fieldErrors).length === 0;
  const executionSupported = SUPPORTED_FOR_EXECUTION.includes(type);

  function updateValue(key: string, value: unknown) {
    setValues((v) => ({ ...v, [key]: value }));
  }

  async function handleSubmit() {
    if (!budgetValid || !valuesValid || !executionSupported || xmemHealthy === false) return;

    setSubmitting(true);
    setSubmitError(null);
    const payload = buildPayload(modelId, rawConfig, type, dtype, values, deploymentInput);

    try {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = (await res.json()) as CreateJobResponse;

      if (!res.ok) {
        if (data.error_code === "unsupported_type") {
          setSubmitError("이 유형은 아직 예측 실행을 지원하지 않습니다.");
        } else {
          // invalid_payload carries the server's own message; showing it beats
          // a generic failure when the form and the server disagree.
          setSubmitError(data.message ?? "예측 요청을 시작하지 못했습니다.");
        }
        return;
      }

      onJobCreated(data.job_id);
    } catch {
      setSubmitError("오케스트레이터에 연결할 수 없습니다.");
    } finally {
      setSubmitting(false);
    }
  }

  function renderField(field: FieldSpec) {
    const visible = !field.showWhen || field.showWhen(values);
    if (!visible) return null;

    if (field.key === "resolution" && resolutionLocked) {
      const locked = defaultResolution(config);
      return (
        <div key={field.key} className="space-y-1">
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">{field.label}</label>
          <input
            type="number"
            value={locked}
            readOnly
            disabled
            className="w-full cursor-not-allowed rounded-lg border border-neutral-300 bg-neutral-100 px-3 py-2 text-sm text-neutral-500 dark:border-neutral-700 dark:bg-neutral-800"
          />
          <p className="text-xs text-amber-600 dark:text-amber-400">
            이 모델은 position embedding 길이가 고정되어 있어 학습 해상도({locked})만 사용할 수 있습니다.
          </p>
        </div>
      );
    }

    if (field.kind === "radio") {
      return (
        <div key={field.key} className="space-y-1.5">
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">{field.label}</label>
          <div className="flex gap-2">
            {field.options.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => updateValue(field.key, opt.value)}
                className={`rounded-lg border px-3 py-1.5 text-sm ${
                  values[field.key] === opt.value
                    ? "border-neutral-900 bg-neutral-900 text-white dark:border-white dark:bg-white dark:text-neutral-900"
                    : "border-neutral-300 dark:border-neutral-700"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      );
    }

    const fieldError = fieldErrors[field.key];
    return (
      <div key={field.key} className="space-y-1">
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">{field.label}</label>
        <input
          type="number"
          min={field.hardMin}
          value={values[field.key] as string | number}
          onChange={(e) => updateValue(field.key, e.target.value)}
          aria-invalid={fieldError ? true : undefined}
          className={`w-full rounded-lg border bg-white px-3 py-2 text-sm outline-none focus:ring-2 dark:bg-neutral-900 ${
            fieldError
              ? "border-red-500 focus:border-red-500 focus:ring-red-200 dark:focus:ring-red-900"
              : "border-neutral-300 focus:border-neutral-500 focus:ring-neutral-200 dark:border-neutral-700 dark:focus:ring-neutral-700"
          }`}
        />
        {field.helperText && <p className="text-xs text-neutral-500">{field.helperText}</p>}
        {fieldError && <p className="text-xs text-red-600">{fieldError}</p>}
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="font-mono text-sm text-neutral-500">{modelId}</p>
          <span className="mt-1 inline-block rounded-full bg-neutral-900 px-2.5 py-1 text-xs font-medium text-white dark:bg-white dark:text-neutral-900">
            {schema.label}
          </span>
        </div>
        <button type="button" onClick={onBack} className="text-sm text-neutral-500 underline underline-offset-2">
          모델 다시 선택
        </button>
      </div>

      {xmemHealthy !== true && setupState === "provisioning" && (
        <p className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-700 dark:bg-amber-950 dark:text-amber-400">
          xMem 환경을 준비하는 중입니다. 최초 실행 시 도커 이미지 다운로드 때문에 수 분 이상 걸릴 수 있습니다 — 잠시
          후 자동으로 활성화됩니다.
        </p>
      )}

      {xmemHealthy !== true && setupState === "error" && (
        <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-400">
          xMem 환경 준비에 실패했습니다: {setupErrorMessage}
          <br />
          Docker Desktop이 실행 중인지 확인 후 &quot;모델 다시 선택&quot;으로 재시도하거나, 직접{" "}
          <code className="font-mono">bash setup.sh</code>를 실행하세요.
        </p>
      )}

      {xmemHealthy === false && setupState !== "provisioning" && setupState !== "error" && (
        <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-400">
          xMem 컨테이너가 실행 중이 아닙니다. Docker Desktop이 실행 중인지 확인하세요.
        </p>
      )}

      {!executionSupported && (
        <p className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-700 dark:bg-amber-950 dark:text-amber-400">
          이 유형은 아직 예측 실행을 지원하지 않습니다. 입력값 구성은 가능하지만 실행은 준비 중입니다.
        </p>
      )}

      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">{DTYPE_FIELD.label}</label>
        <div className="flex gap-2">
          {DTYPE_FIELD.options.map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={() => setDtype(opt.value as Dtype)}
              className={`rounded-lg border px-3 py-1.5 text-sm ${
                dtype === opt.value
                  ? "border-neutral-900 bg-neutral-900 text-white dark:border-white dark:bg-white dark:text-neutral-900"
                  : "border-neutral-300 dark:border-neutral-700"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-4">{schema.fields.map(renderField)}</div>

      {values.mode === "generate" && (
        <p className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:bg-amber-950 dark:text-amber-400">
          {GENERATE_MODE_WARNING}
        </p>
      )}

      <div className="space-y-4 rounded-xl border border-dashed border-neutral-300 p-4 dark:border-neutral-700">
        <div>
          <p className="text-sm font-semibold text-neutral-800 dark:text-neutral-200">배포 조건</p>
          <p className="text-xs text-neutral-500">
            이 값들은 VRAM 예측에는 쓰이지 않고, 다음 단계(인스턴스 후보 필터링)에서만 사용됩니다.
          </p>
        </div>

        <div className="space-y-1">
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">aws_region</label>
          <select
            value={awsRegion}
            onChange={(e) => setAwsRegion(e.target.value)}
            className="w-full rounded-lg border border-neutral-300 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-500 focus:ring-2 focus:ring-neutral-200 dark:border-neutral-700 dark:bg-neutral-900 dark:focus:ring-neutral-700"
          >
            {AWS_REGIONS.map((r) => (
              <option key={r.code} value={r.code}>
                {r.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-1">
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">budget_amount</label>
          <div className="flex items-center gap-2">
            <input
              type="number"
              min={0}
              step="0.01"
              value={budgetAmountRaw}
              onChange={(e) => setBudgetAmountRaw(e.target.value)}
              className="w-full rounded-lg border border-neutral-300 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-500 focus:ring-2 focus:ring-neutral-200 dark:border-neutral-700 dark:bg-neutral-900 dark:focus:ring-neutral-700"
            />
            <div className="flex shrink-0 gap-2">
              {(["hourly", "monthly"] as BudgetPeriod[]).map((period) => (
                <button
                  key={period}
                  type="button"
                  onClick={() => setBudgetPeriod(period)}
                  className={`rounded-lg border px-3 py-1.5 text-sm ${
                    budgetPeriod === period
                      ? "border-neutral-900 bg-neutral-900 text-white dark:border-white dark:bg-white dark:text-neutral-900"
                      : "border-neutral-300 dark:border-neutral-700"
                  }`}
                >
                  {period === "hourly" ? "시간당" : "월간"}
                </button>
              ))}
            </div>
          </div>
          <p className="text-xs text-neutral-500">
            GPU 인스턴스 비용만 입력하세요. 스토리지·네트워크 등 다른 클라우드 비용은 제외합니다.
          </p>
          {budgetPeriod === "monthly" && (
            <p className="text-xs text-neutral-500" title="24시간 × 30일 = 720시간 기준">
              ÷ 720시간 (24시간 × 30일 기준) = 시간당 ${perHourPreview.toFixed(4)}
            </p>
          )}
          {!budgetValid && <p className="text-xs text-red-600">budget_amount는 0보다 커야 합니다.</p>}
        </div>
      </div>

      {submitError && <p className="text-sm text-red-600">{submitError}</p>}

      <div className="flex justify-end">
        <button
          type="button"
          disabled={submitting || !budgetValid || !valuesValid || !executionSupported || xmemHealthy === false}
          onClick={handleSubmit}
          className="rounded-lg bg-neutral-900 px-5 py-2.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40 dark:bg-white dark:text-neutral-900"
        >
          {submitting ? "요청 중…" : "예측 실행"}
        </button>
      </div>

    </div>
  );
}
