"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import type { ModelConfigFields, WorkloadType } from "@/lib/types";
import ModelSearchStep from "./ModelSearchStep";
import ParamsStep from "./ParamsStep";
import JobFlow from "./JobFlow";

interface ClassifiedModel {
  modelId: string;
  type: Exclude<WorkloadType, "unsupported">;
  config: ModelConfigFields;
}

type SetupState = "idle" | "provisioning" | "ready" | "error";

const HEALTH_POLL_MS = 5000;

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

    // Deferred one microtask so the "provisioning" flip isn't a synchronous
    // setState call at the top of the effect body (imperceptible delay; the
    // fetch below still dominates the actual wait).
    Promise.resolve().then(() => {
      setSetupState("provisioning");
      setSetupErrorMessage(null);
    });

    fetch("/api/setup", { method: "POST" })
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok || data.status === "error") {
          setSetupState("error");
          setSetupErrorMessage(data.message ?? "환경 준비 중 오류가 발생했습니다.");
          return;
        }
        setSetupState("ready");
        checkHealth();
      })
      .catch(() => {
        setSetupState("error");
        setSetupErrorMessage("오케스트레이터에 연결할 수 없습니다.");
      });
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
        onClassified={(modelId, type, config) => {
          if (type === "unsupported") return;
          setClassified({ modelId, type, config });
        }}
      />
    );
  }

  return (
    <ParamsStep
      modelId={classified.modelId}
      type={classified.type}
      config={classified.config}
      onBack={() => setClassified(null)}
      xmemHealthy={xmemHealthy}
      setupState={setupState}
      setupErrorMessage={setupErrorMessage}
      onJobCreated={handleJobCreated}
    />
  );
}
