"use client";

import type { BudgetPeriod, WorkloadType } from "@/lib/types";
import { SCENARIO_PRESETS, describeScenario, type ScenarioPreset } from "@/lib/workloadSchema";
import DeploymentFields from "./DeploymentFields";

type SetupState = "idle" | "provisioning" | "ready" | "error";

interface Props {
  modelId: string;
  type: Exclude<WorkloadType, "unsupported">;
  selectedScenarioId: string | null;
  onSelectScenario: (preset: ScenarioPreset) => void;
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
  submitting: boolean;
  submitError: string | null;
  onSubmit: () => void;
  onBack: () => void;
}

export default function ScenarioStep({
  modelId,
  type,
  selectedScenarioId,
  onSelectScenario,
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
  submitting,
  submitError,
  onSubmit,
  onBack,
}: Props) {
  const scenarios = SCENARIO_PRESETS[type];

  return (
    <div className="mx-auto w-full max-w-xl space-y-6">
      <div className="flex items-center justify-between">
        <p className="font-mono text-sm text-neutral-500">{modelId}</p>
        <button type="button" onClick={onBack} className="text-sm text-neutral-500 underline underline-offset-2">
          직접 입력으로 돌아가기
        </button>
      </div>

      <div>
        <p className="text-sm font-semibold text-neutral-800 dark:text-neutral-200">작업 규모를 골라주세요</p>
        <p className="mt-1 text-xs text-neutral-500">
          아래 수치는 예측 엔진이 검증한 값이 아니라 사용 편의를 위한 예시입니다. 선택 후에도 &quot;직접 입력으로
          돌아가기&quot;에서 값을 그대로 수정할 수 있습니다.
        </p>
      </div>

      <div className="space-y-2">
        {scenarios.map((preset) => {
          const selected = preset.id === selectedScenarioId;
          return (
            <button
              key={preset.id}
              type="button"
              onClick={() => onSelectScenario(preset)}
              className={`w-full rounded-lg border px-4 py-3 text-left ${
                selected
                  ? "border-neutral-900 ring-1 ring-neutral-900 dark:border-white dark:ring-white"
                  : "border-neutral-300 dark:border-neutral-700"
              }`}
            >
              <p className="text-sm font-medium text-neutral-800 dark:text-neutral-200">{preset.label}</p>
              <p className="mt-0.5 text-xs text-neutral-500">{describeScenario(type, preset)}</p>
            </button>
          );
        })}
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
        </p>
      )}
      {!executionSupported && (
        <p className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-700 dark:bg-amber-950 dark:text-amber-400">
          이 유형은 아직 예측 실행을 지원하지 않습니다.
        </p>
      )}

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

      <div className="flex justify-between">
        <button
          type="button"
          onClick={onBack}
          className="rounded-lg border border-neutral-300 px-5 py-2.5 text-sm font-medium text-neutral-700 dark:border-neutral-700 dark:text-neutral-300"
        >
          이전
        </button>
        <button
          type="button"
          disabled={
            submitting || !selectedScenarioId || !budgetValid || !executionSupported || xmemHealthy === false
          }
          onClick={onSubmit}
          className="rounded-lg bg-neutral-900 px-5 py-2.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40 dark:bg-white dark:text-neutral-900"
        >
          {submitting ? "요청 중…" : "다음"}
        </button>
      </div>
    </div>
  );
}
