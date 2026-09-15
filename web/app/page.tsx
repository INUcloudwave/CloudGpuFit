import { Suspense } from "react";
import Wizard from "@/components/Wizard";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center bg-zinc-50 px-6 py-16 font-sans dark:bg-black">
      <div className="mb-10 text-center">
        <h1 className="text-2xl font-semibold text-black dark:text-zinc-50">CloudGPUFit</h1>
        <p className="mt-1 text-sm text-zinc-500">추론 워크로드 GPU 인스턴스 추천 시뮬레이터 — 입력 마법사</p>
      </div>
      <Suspense fallback={<p className="text-sm text-neutral-400">불러오는 중…</p>}>
        <Wizard />
      </Suspense>
    </div>
  );
}
