import { NextResponse } from "next/server";

const ORCHESTRATOR_URL = process.env.ORCHESTRATOR_URL ?? "http://localhost:8000";

export async function GET() {
  try {
    const res = await fetch(`${ORCHESTRATOR_URL}/health`, { cache: "no-store" });
    if (!res.ok) return NextResponse.json({ ok: false });
    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ ok: false });
  }
}
