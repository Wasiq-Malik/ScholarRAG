import { NextResponse } from "next/server";

import { fetchBackendHealth, shouldUseMockApi } from "@/lib/scholarrag-api";
import { mockHealthResponse } from "@/lib/mock-scholarrag";

export const dynamic = "force-dynamic";

export async function GET() {
  if (shouldUseMockApi()) {
    return NextResponse.json(mockHealthResponse());
  }
  try {
    const health = await fetchBackendHealth();
    return NextResponse.json(health);
  } catch (error) {
    return NextResponse.json(
      {
        status: "error",
        vector_backend: "unavailable",
        embedding_model: "",
        embedding_dimension: 0,
        llm_model: "",
        detail: error instanceof Error ? error.message : "Backend health check failed",
      },
      { status: 502 },
    );
  }
}
