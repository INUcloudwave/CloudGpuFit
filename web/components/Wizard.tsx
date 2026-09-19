"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import type { BudgetPeriod, CreateJobResponse, Dtype, ModelConfigFields, WorkloadType } from "@/lib/types";
import { DEFAULT_AWS_REGION } from "@/lib/awsRegions";
import {
  DTYPE_FIELD,
  applyScenario,
  buildPayload,
  collectFieldErrors,
  defaultValuesFor,
  type ScenarioPreset,
} from "@/lib/workloadSchema";
import ModelSearchStep from "./ModelSearchStep";
import ParamsStep from "./ParamsStep";
import ScenarioStep from "./ScenarioStep";
import SloGate from "./SloGate";

interface ClassifiedModel {
  modelId: string;
  type: Exclude<WorkloadType, "unsupported" | "audio_encdec">;
  config: ModelConfigFields;
  rawConfig: Record<string, unknown>;
}

type SetupState = "idle" | "provisioning" | "ready" | "error";
type ParamsView = "manual" | "scenario";

const HEALTH_POLL_MS = 5000;

// How long to keep re-polling a 409 "in_progress" /setup response before
// giving up and showing an error — bounds an otherwise-unbounded retry loop
// if the OTHER in-flight setup call never finishes.
const SETUP_POLL_INTERVAL_MS = 2000;
const SETUP_POLL_MAX_ATTEMPTS = 30; // ~60s

// Mirrors orchestrator/main.py SUPPORTED_TYPES — the types every stage of
// the pipeline can actually run, not just the ones with an input form.
const SUPPORTED_FOR_EXECUTION: WorkloadType[] = ["decoder_llm", "vision_cls", "text_encdec"];

export default function Wizard() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [classified, setClassified] = useState<ClassifiedModel | null>(null);
  const [jobId, setJobId] = useState<string | null>(searchParams.get("job"));
  const [xmemHealthy, setXmemHealthy] = useState<boolean | null>(null);
  const [setupState, setSetupState] = useState<SetupState>("idle");
  const [setupErrorMessage, setSetupErrorMessage] = useState<string | null>(null);

  // Params-phase state, lifted out of ParamsStep/ScenarioStep so switching
  // between "직접 입력" and "잘 모르겠어요" keeps whatever was already set —
  // in-memory only (React state), not saved anywhere, gone on page reload.
  const [view, setView] = useState<ParamsView>("manual");
  const [dtype, setDtype] = useState<Dtype>(DTYPE_FIELD.default as Dtype);
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [selectedScenarioId, setSelectedScenarioId] = useState<string | null>(null);

  // Deployment (region/budget) is environment context, not a model property —
  // it survives picking a different model, unlike dtype/values/view above.
  const [awsRegion, setAwsRegion] = useState(DEFAULT_AWS_REGION);
  // Raw text, not a number: a controlled number input bound to a coerced
  // number re-renders "0" back in on every keystroke, so clearing the field
  // to type a fresh value produces "0500" instead of "500".
  const [budgetAmountRaw, setBudgetAmountRaw] = useState("500");
  const [budgetPeriod, setBudgetPeriod] = useState<BudgetPeriod>("monthly");

  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  function checkHealth() {
    return fetch("/api/health", { cache: "no-store" })
      .then((res) => res.json())
      .then((data) => setXmemHealthy(Boolean(data.ok)))
      .catch(() => setXmemHealthy(false));
  }

  useEffect(() => {
    checkHealth();
  }, []);

  // Kick off container provisioning as soon as the page loads, instead of
  // waiting for a model to be picked — the first run on a fresh machine can
  // take minutes (docker pull), so start that clock as early as possible
  // rather than after Step 1. The endpoint is idempotent and cheap once the
  // container already exists, so calling it eagerly costs nothing on every
  // later visit.
  useEffect(() => {
    if (jobId) return;

    let cancelled = false;

    // Deferred one microtask so the "provisioning" flip isn't a synchronous
    // setState call at the top of the effect body (imperceptible delay; the
    // fetch below still dominates the actual wait).
    Promise.resolve().then(() => {
      if (!cancelled) {
        setSetupState("provisioning");
        setSetupErrorMessage(null);
      }
    });

    async function callSetup(attempt: number): Promise<void> {
      let res: Response;
      let data: { status?: string; message?: string };
      try {
        res = await fetch("/api/setup", { method: "POST" });
        data = await res.json();
      } catch {
        if (!cancelled) {
          setSetupState("error");
          setSetupErrorMessage("오케스트레이터에 연결할 수 없습니다.");
        }
        return;
      }
      if (cancelled) return;

      // 409 means another /setup call (a second effect run under React
      // StrictMode's dev double-invoke, another tab, ...) is already
      // provisioning — that's not a failure, just wait for it and re-check
      // instead of surfacing a false "환경 준비 실패".
      if (res.status === 409 && data.status === "in_progress") {
        if (attempt >= SETUP_POLL_MAX_ATTEMPTS) {
          setSetupState("error");
          setSetupErrorMessage("환경 준비가 예상보다 오래 걸리고 있습니다. 잠시 후 새로고침해보세요.");
          return;
        }
        setTimeout(() => callSetup(attempt + 1), SETUP_POLL_INTERVAL_MS);
        return;
      }

      if (!res.ok || data.status === "error") {
        setSetupState("error");
        setSetupErrorMessage(data.message ?? "환경 준비 중 오류가 발생했습니다.");
        return;
      }

      setSetupState("ready");
      checkHealth();
    }

    callSetup(0);

    return () => {
      cancelled = true;
    };
    // Mount-only: the whole point is to fire once, as early as possible.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reactively clear the "not ready" banner once provisioning finishes,
  // without requiring the user to reload the page.
  useEffect(() => {
    if (jobId || xmemHealthy === true) return;
    const id = setInterval(checkHealth, HEALTH_POLL_MS);
    return () => clearInterval(id);
  }, [jobId, xmemHealthy]);

  function handleJobCreated(id: string) {
    setJobId(id);
    router.replace(`?job=${id}`);
  }

  function handleReset() {
    setJobId(null);
    setClassified(null);
    router.replace("?");
  }

  function updateValue(key: string, value: unknown) {
    setValues((v) => ({ ...v, [key]: value }));
  }

  function handleSelectScenario(preset: ScenarioPreset) {
    if (!classified) return;
    const applied = applyScenario(preset, classified.type, classified.config);
    setDtype(applied.dtype);
    setValues(applied.values);
    setSelectedScenarioId(preset.id);
  }

  const budgetAmount = Number(budgetAmountRaw);
  const budgetValid = Number.isFinite(budgetAmount) && budgetAmount > 0;
  const deploymentInput = { aws_region: awsRegion, budget_amount: budgetAmount, budget_period: budgetPeriod };
  const fieldErrors = classified ? collectFieldErrors(classified.type, values) : {};
  const valuesValid = Object.keys(fieldErrors).length === 0;
  const executionSupported = classified ? SUPPORTED_FOR_EXECUTION.includes(classified.type) : false;

  async function handleSubmit() {
    if (!classified) return;
    if (!budgetValid || !valuesValid || !executionSupported || xmemHealthy === false) return;

    setSubmitting(true);
    setSubmitError(null);
    const payload = buildPayload(classified.modelId, classified.rawConfig, classified.type, dtype, values, deploymentInput);

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

      handleJobCreated(data.job_id);
    } catch {
      setSubmitError("오케스트레이터에 연결할 수 없습니다.");
    } finally {
      setSubmitting(false);
    }
  }

  if (jobId) {
    return <SloGate jobId={jobId} onReset={handleReset} />;
  }

  if (!classified) {
    return (
      <ModelSearchStep
        onClassified={(modelId, type, config, rawConfig) => {
          setClassified({ modelId, type, config, rawConfig });
          // A new model can be a different workload type with a different
          // field shape, so params-phase state resets here — but deployment
          // (region/budget) is untouched, on purpose.
          setDtype(DTYPE_FIELD.default as Dtype);
          setValues(defaultValuesFor(type, config));
          setSelectedScenarioId(null);
          setView("manual");
          setSubmitError(null);
        }}
      />
    );
  }

  const deploymentFieldProps = {
    awsRegion,
    onAwsRegionChange: setAwsRegion,
    budgetAmountRaw,
    onBudgetAmountRawChange: setBudgetAmountRaw,
    budgetPeriod,
    onBudgetPeriodChange: setBudgetPeriod,
    budgetValid,
  };

  if (view === "scenario") {
    return (
      <ScenarioStep
        modelId={classified.modelId}
        type={classified.type}
        selectedScenarioId={selectedScenarioId}
        onSelectScenario={handleSelectScenario}
        {...deploymentFieldProps}
        xmemHealthy={xmemHealthy}
        setupState={setupState}
        setupErrorMessage={setupErrorMessage}
        executionSupported={executionSupported}
        submitting={submitting}
        submitError={submitError}
        onSubmit={handleSubmit}
        onBack={() => setView("manual")}
      />
    );
  }

  return (
    <ParamsStep
      modelId={classified.modelId}
      type={classified.type}
      config={classified.config}
      dtype={dtype}
      onDtypeChange={setDtype}
      values={values}
      onUpdateValue={updateValue}
      fieldErrors={fieldErrors}
      onBack={() => setClassified(null)}
      onShowScenarios={() => setView("scenario")}
      {...deploymentFieldProps}
      xmemHealthy={xmemHealthy}
      setupState={setupState}
      setupErrorMessage={setupErrorMessage}
      executionSupported={executionSupported}
      valuesValid={valuesValid}
      submitting={submitting}
      submitError={submitError}
      onSubmit={handleSubmit}
    />
  );
}
