import type { HealthResponse, QueryResponse, RetrieveResponse } from "@/types/scholarrag";

const DEFAULT_API_BASE_URL = "https://complete-jay-strictly.ngrok-free.app";
const MOCK_API_VALUES = new Set(["1", "true", "yes", "on"]);

export function getBackendBaseUrl(): string {
  return (process.env.SCHOLARRAG_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/+$/, "");
}

export function shouldUseMockApi(): boolean {
  const value = process.env.SCHOLARRAG_USE_MOCK_API;
  return value ? MOCK_API_VALUES.has(value.toLowerCase()) : false;
}

async function readError(response: Response): Promise<string> {
  const text = await response.text();
  if (!text) {
    return `${response.status} ${response.statusText}`;
  }
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    if (parsed.detail) {
      return typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail);
    }
  } catch {
    // Keep the raw text.
  }
  return text;
}

export async function fetchBackendHealth(): Promise<HealthResponse> {
  const response = await fetch(`${getBackendBaseUrl()}/health`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json() as Promise<HealthResponse>;
}

export async function runBackendQuery(input: {
  question: string;
  top_k: number;
  filters?: Record<string, unknown>;
}): Promise<QueryResponse> {
  const response = await fetch(`${getBackendBaseUrl()}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question: input.question,
      top_k: input.top_k,
      filters: input.filters || {},
    }),
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json() as Promise<QueryResponse>;
}

export async function runBackendRetrieve(input: {
  question: string;
  top_k: number;
  filters?: Record<string, unknown>;
}): Promise<RetrieveResponse> {
  const response = await fetch(`${getBackendBaseUrl()}/retrieve`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question: input.question,
      top_k: input.top_k,
      filters: input.filters || {},
    }),
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json() as Promise<RetrieveResponse>;
}
