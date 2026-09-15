import { NextRequest, NextResponse } from "next/server";
import type { HfSearchResult } from "@/lib/types";

export const runtime = "nodejs";

interface HfApiModel {
  id: string;
  downloads?: number;
  likes?: number;
  pipeline_tag?: string;
  tags?: string[];
}

export async function GET(req: NextRequest) {
  const q = req.nextUrl.searchParams.get("q")?.trim();
  if (!q) {
    return NextResponse.json([]);
  }

  const url = new URL("https://huggingface.co/api/models");
  url.searchParams.set("search", q);
  url.searchParams.set("sort", "downloads");
  url.searchParams.set("direction", "-1");
  url.searchParams.set("limit", "5");

  const headers: Record<string, string> = {};
  if (process.env.HF_TOKEN) {
    headers.Authorization = `Bearer ${process.env.HF_TOKEN}`;
  }

  try {
    const res = await fetch(url.toString(), { headers, signal: req.signal });
    if (!res.ok) {
      return NextResponse.json({ error: "hf_search_failed" }, { status: res.status });
    }
    const data = (await res.json()) as HfApiModel[];
    const results: HfSearchResult[] = (Array.isArray(data) ? data : []).slice(0, 5).map((m) => ({
      id: m.id,
      downloads: m.downloads ?? null,
      likes: m.likes ?? null,
      pipeline_tag: m.pipeline_tag ?? null,
      tags: Array.isArray(m.tags) ? m.tags : [],
    }));
    return NextResponse.json(results);
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      return NextResponse.json({ error: "aborted" }, { status: 499 });
    }
    return NextResponse.json({ error: "hf_search_error" }, { status: 502 });
  }
}
