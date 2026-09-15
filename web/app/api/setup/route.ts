import { NextResponse } from "next/server";

const ORCHESTRATOR_URL = process.env.ORCHESTRATOR_URL ?? "http://localhost:8000";

export async function POST() {
  try {
    const res = await fetch(`${ORCHESTRATOR_URL}/setup`, { method: "POST", cache: "no-store" });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json({ status: "error", message: "오케스트레이터에 연결할 수 없습니다." }, { status: 502 });
  }
}
