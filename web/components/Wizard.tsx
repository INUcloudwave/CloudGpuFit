"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import type { ModelConfigFields, WorkloadType } from "@/lib/types";
import ModelSearchStep from "./ModelSearchStep";
import ParamsStep from "./ParamsStep";
import JobFlow from "./JobFlow";

interface ClassifiedModel {
  modelId: string;
  type: Exclude<WorkloadType, "unsupported" | "audio_encdec">;
  config: ModelConfigFields;
  /** Unused for now — carried through so a later step can send it on to
   * xMem/eager_runtime instead of them re-fetching config.json themselves. */
  rawConfig: Record<string, unknown>;
}

type SetupState = "idle" | "provisioning" | "ready" | "error";

const HEALTH_POLL_MS = 5000;

// How long to keep re-polling a 409 "in_progress" /setup response before
// giving up and showing an error — bounds an otherwise-unbounded retry loop
// if the OTHER in-flight setup call never finishes.
const SETUP_POLL_INTERVAL_MS = 2000;
const SETUP_POLL_MAX_ATTEMPTS = 30; // ~60s

export default function Wizard() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [classified, setClassified] = useState<ClassifiedModel | null>(null);
  const [jobId, setJobId] = useState<string | null>(searchParams.get("job"));
  const [xmemHealthy, setXmemHealthy] = useState<boolean | null>(null);
  const [setupState, setSetupState] = useState<SetupState>("idle");
  const [setupErrorMessage, setSetupErrorMessage] = useState<string | null>(null);

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

  if (jobId) {
    return <JobFlow jobId={jobId} onReset={handleReset} />;
  }

  if (!classified) {
    return (
      <ModelSearchStep
        onClassified={(modelId, type, config, rawConfig) => {
          setClassified({ modelId, type, config, rawConfig });
        }}
      />
    );
  }

  return (
    <ParamsStep
      modelId={classified.modelId}
      type={classified.type}
      config={classified.config}
      rawConfig={classified.rawConfig}
      onBack={() => setClassified(null)}
      xmemHealthy={xmemHealthy}
      setupState={setupState}
      setupErrorMessage={setupErrorMessage}
      onJobCreated={handleJobCreated}
    />
  );
}
