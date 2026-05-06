import { NextResponse } from "next/server";

import { getBackendBaseUrl, shouldUseMockApi } from "@/lib/scholarrag-api";
import { mockQueryResponse } from "@/lib/mock-scholarrag";
import type { QuerySource } from "@/types/scholarrag";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

function sseChunk(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

const SSE_HEADERS = {
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache, no-transform",
  Connection: "keep-alive",
  "X-Accel-Buffering": "no",
};

export async function POST(request: Request) {
  const body = (await request.json()) as {
    question?: unknown;
    sources?: unknown;
  };
  const question = typeof body.question === "string" ? body.question.trim() : "";
  const sources = Array.isArray(body.sources) ? (body.sources as QuerySource[]) : [];
  if (!question) {
    return NextResponse.json({ detail: "question is required" }, { status: 400 });
  }

  if (shouldUseMockApi()) {
    const text = mockQueryResponse(question, Math.max(5, sources.length || 5)).answer;
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        for (let i = 0; i < text.length; i += 160) {
          controller.enqueue(encoder.encode(sseChunk("chunk", { text: text.slice(i, i + 160) })));
        }
        controller.enqueue(encoder.encode(sseChunk("done", {})));
        controller.close();
      },
    });
    return new Response(stream, {
      headers: SSE_HEADERS,
    });
  }

  try {
    const upstream = await fetch(`${getBackendBaseUrl()}/answer/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question, sources }),
      cache: "no-store",
    });
    if (!upstream.ok || !upstream.body) {
      throw new Error(await upstream.text());
    }
    return new Response(upstream.body, {
      headers: SSE_HEADERS,
    });
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Answer streaming failed" },
      { status: 502 },
    );
  }
}
