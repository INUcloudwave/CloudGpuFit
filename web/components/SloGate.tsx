"use client";

import { useEffect, useState } from "react";
import type { JobResponse, PredictPayload } from "@/lib/types";
import SloStep from "./SloStep";
import JobFlow from "./JobFlow";

interface Props {
  jobId: string;
  onReset: () => void;
}

/** Decides, once per jobId, whether to show the SLO input screen or go
 * straight to progress/results — a fresh job has no slo yet (submit it
 * first); reloading a page whose SLO was already submitted (or a cached
 * job, which never needed one to begin with) skips straight to JobFlow
 * instead of asking again. */
export default function SloGate({ jobId, onReset }: Props) {
  const [checked, setChecked] = useState(false);
  const [hasSlo, setHasSlo] = useState(false);
  const [payload, setPayload] = useState<PredictPayload | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/jobs/${jobId}`, { cache: "no-store" })
      .then((res) => res.json())
      .then((data: JobResponse) => {
        if (cancelled) return;
        setHasSlo(data.slo !== null);
        setPayload(data.payload);
        setChecked(true);
      })
      .catch(() => {
        if (!cancelled) {
          setFetchError("작업 정보를 불러오지 못했습니다.");
          setChecked(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  if (!checked) {
    return (
      <div className="mx-auto w-full max-w-xl text-center text-sm text-neutral-500">
        {fetchError ?? "작업 정보를 불러오는 중…"}
      </div>
    );
  }

  if (!hasSlo && payload) {
    return <SloStep jobId={jobId} payload={payload} onSubmitted={() => setHasSlo(true)} />;
  }

  return <JobFlow jobId={jobId} onReset={onReset} />;
}
