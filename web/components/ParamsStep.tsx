"use client";

import type { BudgetPeriod, Dtype, ModelConfigFields, WorkloadType } from "@/lib/types";
import DeploymentFields from "./DeploymentFields";
import {
  DTYPE_FIELD,
  WORKLOAD_SCHEMAS,
  defaultResolution,
  isResolutionLocked,
  type FieldSpec,
} from "@/lib/workloadSchema";

type SetupState = "idle" | "provisioning" | "ready" | "error";

interface Props {
  modelId: string;
  type: Exclude<WorkloadType, "unsupported">;
  config: ModelConfigFields;
  dtype: Dtype;
  onDtypeChange: (dtype: Dtype) => void;
  values: Record<string, unknown>;
  onUpdateValue: (key: string, value: unknown) => void;
  fieldErrors: Record<string, string>;
  onBack: () => void;
  onShowScenarios: () => void;
  awsRegion: string;
  onAwsRegionChange: (region: string) => void;
  budgetAmountRaw: string;
  onBudgetAmountRawChange: (raw: string) => void;
  budgetPeriod: BudgetPeriod;
  onBudgetPeriodChange: (period: BudgetPeriod) => void;
  budgetValid: boolean;
  xmemHealthy: boolean | null;
  setupState: SetupState;
  setupErrorMessage: string | null;
  executionSupported: boolean;
  valuesValid: boolean;
  submitting: boolean;
  submitError: string | null;
  onSubmit: () => void;
}

export default function ParamsStep({
  modelId,
  type,
  config,
  dtype,
  onDtypeChange,
  values,
  onUpdateValue,
  fieldErrors,
  onBack,
  onShowScenarios,
  awsRegion,
  onAwsRegionChange,
  budgetAmountRaw,
  onBudgetAmountRawChange,
  budgetPeriod,
  onBudgetPeriodChange,
  budgetValid,
  xmemHealthy,
  setupState,
  setupErrorMessage,
  executionSupported,
  valuesValid,
  submitting,
  submitError,
  onSubmit,
}: Props) {
  const schema = WORKLOAD_SCHEMAS[type];
  const resolutionLocked = type === "vision_cls" && isResolutionLocked(config);

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
                onClick={() => onUpdateValue(field.key, opt.value)}
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
          onChange={(e) => onUpdateValue(field.key, e.target.value)}
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
              onClick={() => onDtypeChange(opt.value as Dtype)}
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

      <div className="flex justify-end">
        <button type="button" onClick={onShowScenarios} className="text-sm text-neutral-500 underline underline-offset-2">
          잘 모르겠어요
        </button>
      </div>

      <DeploymentFields
        awsRegion={awsRegion}
        onAwsRegionChange={onAwsRegionChange}
        budgetAmountRaw={budgetAmountRaw}
        onBudgetAmountRawChange={onBudgetAmountRawChange}
        budgetPeriod={budgetPeriod}
        onBudgetPeriodChange={onBudgetPeriodChange}
        budgetValid={budgetValid}
      />

      {submitError && <p className="text-sm text-red-600">{submitError}</p>}

      <div className="flex justify-end">
        <button
          type="button"
          disabled={submitting || !budgetValid || !valuesValid || !executionSupported || xmemHealthy === false}
          onClick={onSubmit}
          className="rounded-lg bg-neutral-900 px-5 py-2.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40 dark:bg-white dark:text-neutral-900"
        >
          {submitting ? "요청 중…" : "예측 실행"}
        </button>
      </div>
    </div>
  );
}
