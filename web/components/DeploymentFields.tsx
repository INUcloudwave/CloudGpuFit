"use client";

import type { BudgetPeriod } from "@/lib/types";
import { AWS_REGIONS } from "@/lib/awsRegions";
import { BUDGET_HOURS_PER_MONTH } from "@/lib/workloadSchema";

interface Props {
  awsRegion: string;
  onAwsRegionChange: (region: string) => void;
  budgetAmountRaw: string;
  onBudgetAmountRawChange: (raw: string) => void;
  budgetPeriod: BudgetPeriod;
  onBudgetPeriodChange: (period: BudgetPeriod) => void;
  budgetValid: boolean;
}

/** 직접 입력 화면과 시나리오 화면 둘 다에서 쓰는 리전/예산 입력 — 같은 컴포넌트를
 * 공유하는 것 자체가 두 화면을 오가도 값이 유지되는 장치다(상태는 Wizard가 들고 있음). */
export default function DeploymentFields({
  awsRegion,
  onAwsRegionChange,
  budgetAmountRaw,
  onBudgetAmountRawChange,
  budgetPeriod,
  onBudgetPeriodChange,
  budgetValid,
}: Props) {
  const budgetAmount = Number(budgetAmountRaw);
  const perHourPreview = budgetPeriod === "monthly" ? budgetAmount / BUDGET_HOURS_PER_MONTH : budgetAmount;

  return (
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
          onChange={(e) => onAwsRegionChange(e.target.value)}
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
            onChange={(e) => onBudgetAmountRawChange(e.target.value)}
            className="w-full rounded-lg border border-neutral-300 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-500 focus:ring-2 focus:ring-neutral-200 dark:border-neutral-700 dark:bg-neutral-900 dark:focus:ring-neutral-700"
          />
          <div className="flex shrink-0 gap-2">
            {(["hourly", "monthly"] as BudgetPeriod[]).map((period) => (
              <button
                key={period}
                type="button"
                onClick={() => onBudgetPeriodChange(period)}
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
  );
}
