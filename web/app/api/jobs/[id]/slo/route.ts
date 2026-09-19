import { NextRequest, NextResponse } from "next/server";

const ORCHESTRATOR_URL = process.env.ORCHESTRATOR_URL ?? "http://localhost:8000";

export async function PATCH(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error_code: "invalid_body" }, { status: 400 });
  }

  try {
    const res = await fetch(`${ORCHESTRATOR_URL}/jobs/${encodeURIComponent(id)}/slo`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json({ error_code: "orchestrator_unreachable" }, { status: 502 });
  }
}
