import { NextResponse } from "next/server";

import { runBackendRetrieve, shouldUseMockApi } from "@/lib/scholarrag-api";
import { mockQueryResponse } from "@/lib/mock-scholarrag";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

export async function POST(request: Request) {
  const body = (await request.json()) as {
    question?: unknown;
    top_k?: unknown;
    filters?: unknown;
  };
  const question = typeof body.question === "string" ? body.question.trim() : "";
  if (!question) {
    return NextResponse.json({ detail: "question is required" }, { status: 400 });
  }
  const topK = typeof body.top_k === "number" ? Math.max(1, Math.min(50, body.top_k)) : 50;
  const filters =
    body.filters && typeof body.filters === "object" && !Array.isArray(body.filters)
      ? (body.filters as Record<string, unknown>)
      : {};

  if (shouldUseMockApi()) {
    const mock = mockQueryResponse(question, topK);
    return NextResponse.json({
      sources: mock.sources,
      retrieval: mock.retrieval,
      models: mock.models,
    });
  }

  try {
    const result = await runBackendRetrieve({ question, top_k: topK, filters });
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Retrieval failed" },
      { status: 502 },
    );
  }
}
