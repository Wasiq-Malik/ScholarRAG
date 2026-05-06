"use client";

import {
  Activity,
  AlertCircle,
  ArrowUpRight,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock3,
  ExternalLink,
  FileText,
  GraduationCap,
  LibraryBig,
  Loader2,
  Search,
  Sparkles,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { HealthResponse, QuerySource, RetrieveResponse } from "@/types/scholarrag";

const EXAMPLE_QUERIES = [
  "is Agentic AI being used for scientific discovery?",
  "are vision language models tensor parallelized during inference?",
  "have VLMs been used for Tuberculosis detection?",
];

const PAGE_SIZE = 10;
const RETRIEVE_TOP_K = 50;
const ANSWER_CONTEXT_K = 10;
const STREAM_STEP_MS = 18;

type ResultsState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: RetrieveResponse }
  | { status: "error"; message: string };

type OverviewState =
  | { status: "idle"; text: string }
  | { status: "loading"; text: string }
  | { status: "success"; text: string }
  | { status: "error"; text: string; message: string };

function confidenceLabel(source: QuerySource): string {
  return source.confidence_score.toFixed(2);
}

function healthPillState(health: HealthResponse | null): {
  label: string;
  className: string;
} {
  if (!health) {
    return {
      label: "Checking backend",
      className:
        "border-[#f3d58a] bg-[#fff8e6] text-[#8a6110] shadow-[0_10px_24px_rgba(245,158,11,0.14)]",
    };
  }
  if (health.status !== "ok" || health.vector_backend === "unavailable") {
    return {
      label: health.detail || "Backend unavailable",
      className:
        "border-[#efb1b1] bg-[#fff1f1] text-[#a12626] shadow-[0_10px_24px_rgba(239,68,68,0.12)]",
    };
  }
  if (health.vector_backend === "mock") {
    return {
      label: "Mock dataset active",
      className:
        "border-[#f3d58a] bg-[#fff8e6] text-[#8a6110] shadow-[0_10px_24px_rgba(245,158,11,0.14)]",
    };
  }
  return {
    label: `${health.vector_backend.toUpperCase()} index online`,
    className:
      "border-[#b6e1c0] bg-[#edf9f0] text-[#1f6b3a] shadow-[0_10px_24px_rgba(34,197,94,0.12)]",
  };
}

function formatLatencyMs(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return "Latency unavailable";
  }
  if (value < 1) {
    return "Retrieval <1 ms";
  }
  return `Retrieval ${Math.round(value)} ms`;
}

function pdfUrl(source: QuerySource): string | null {
  if (!source.paper_id) {
    return null;
  }
  return `https://arxiv.org/pdf/${source.paper_id}`;
}

function splitStreamText(text: string): string[] {
  const pieces = text.match(/\S+\s*|\s+/g);
  return pieces && pieces.length ? pieces : [text];
}

function linkifyCitations(text: string, maxCitation: number): string {
  if (!text || maxCitation < 1) {
    return text;
  }
  return text.replace(/\[((?:\d+\s*,\s*)*\d+)\]/g, (fullMatch, rawGroup: string) => {
    const parts = rawGroup.split(",").map((part) => part.trim());
    const rendered = parts.map((part) => {
      const citationIndex = Number(part);
      if (!Number.isInteger(citationIndex) || citationIndex < 1 || citationIndex > maxCitation) {
        return part;
      }
      return `[${citationIndex}](#citation-${citationIndex})`;
    });
    return rendered.join(", ");
  });
}

function friendlyOverviewError(detail: string): string {
  const normalized = detail.toLowerCase();
  if (
    normalized.includes("503") ||
    normalized.includes("rate limit") ||
    normalized.includes("high demand") ||
    normalized.includes("unavailable")
  ) {
    return "Answer generation is temporarily unavailable due to model capacity or rate limits. Retrieval succeeded, so you can still review the ranked papers below.";
  }
  if (normalized.includes("api key") || normalized.includes("permission") || normalized.includes("403")) {
    return "Answer generation is currently unavailable. Retrieval succeeded, so you can still review the ranked papers below.";
  }
  return "Answer generation is currently unavailable. Retrieval succeeded, so you can still review the ranked papers below.";
}

async function streamOverview(
  question: string,
  sources: QuerySource[],
  onChunk: (text: string) => void,
): Promise<void> {
  const response = await fetch("/api/answer-stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, sources }),
  });
  if (!response.ok || !response.body) {
    let detail = "Overview streaming failed";
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // Ignore JSON parse failure.
    }
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const pendingPieces: string[] = [];
  let streamClosed = false;

  const smoothDrain = (async () => {
    while (!streamClosed || pendingPieces.length > 0) {
      if (!pendingPieces.length) {
        await new Promise((resolve) => setTimeout(resolve, STREAM_STEP_MS));
        continue;
      }
      const piece = pendingPieces.shift();
      if (!piece) {
        continue;
      }
      onChunk(piece);
      await new Promise((resolve) => setTimeout(resolve, STREAM_STEP_MS));
    }
  })();

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const separatorIndex = buffer.indexOf("\n\n");
      if (separatorIndex === -1) {
        break;
      }
      const rawEvent = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);

      const eventLine = rawEvent
        .split("\n")
        .find((line) => line.startsWith("event:"));
      const dataLine = rawEvent
        .split("\n")
        .find((line) => line.startsWith("data:"));

      const eventName = eventLine?.slice("event:".length).trim();
      const dataText = dataLine?.slice("data:".length).trim() || "{}";
      const payload = JSON.parse(dataText) as { text?: string; detail?: string };

      if (eventName === "chunk" && payload.text) {
        pendingPieces.push(...splitStreamText(payload.text));
      }
      if (eventName === "error") {
        streamClosed = true;
        await smoothDrain;
        throw new Error(payload.detail || "Overview streaming failed");
      }
      if (eventName === "done") {
        streamClosed = true;
        await smoothDrain;
        return;
      }
    }
  }

  streamClosed = true;
  await smoothDrain;
}

function ScholarRagLogo() {
  return (
    <div className="flex items-center gap-4">
      <div className="relative grid size-14 place-items-center rounded-2xl bg-[#1558d6] text-white shadow-[0_20px_40px_rgba(21,88,214,0.26)]">
        <LibraryBig size={26} />
        <div className="absolute -right-1.5 -top-1.5 grid size-6 place-items-center rounded-full bg-[#f6c453] text-[#4d3b00] shadow-sm">
          <GraduationCap size={13} />
        </div>
      </div>
      <div>
        <div className="text-2xl font-semibold tracking-[0] text-[#132033] md:text-3xl">ScholaRAG</div>
      </div>
    </div>
  );
}

function SearchForm({
  query,
  setQuery,
  onSubmit,
  isLoading,
  compact,
  showSuggestions,
  setShowSuggestions,
  onPickSuggestion,
}: {
  query: string;
  setQuery: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  isLoading: boolean;
  compact?: boolean;
  showSuggestions: boolean;
  setShowSuggestions: (value: boolean) => void;
  onPickSuggestion: (value: string) => void;
}) {
  return (
    <form
      onSubmit={onSubmit}
      className={`flex w-full items-center gap-2 ${compact ? "" : "mx-auto max-w-4xl flex-col sm:flex-row"}`}
    >
      <div className="relative min-w-0 flex-1">
        <div
          className={`flex min-w-0 items-center gap-3 rounded-full border border-[#cfd6e4] bg-white px-4 shadow-sm transition focus-within:border-[#7ca7ee] focus-within:shadow-md ${
            compact ? "h-12" : "h-[3.75rem] md:h-16"
          }`}
        >
          <Search className="shrink-0 text-[var(--muted)]" size={compact ? 20 : 22} />
          <input
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              if (!compact) {
                setShowSuggestions(true);
              }
            }}
            onFocus={() => {
              if (!compact) {
                setShowSuggestions(true);
              }
            }}
            onBlur={() => {
              if (!compact) {
                setTimeout(() => setShowSuggestions(false), 120);
              }
            }}
            placeholder="Search scientific claims, methods, systems, or papers"
            className={`h-full min-w-0 flex-1 bg-transparent text-[var(--foreground)] outline-none placeholder:text-[#7b8495] ${
              compact ? "text-[15px]" : "text-[15px] md:text-lg"
            }`}
          />
          {!compact ? (
            <button
              type="button"
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => setShowSuggestions(!showSuggestions)}
              className="grid size-8 place-items-center rounded-full text-[var(--muted)] transition hover:bg-[#f2f5fa]"
              aria-label="Toggle suggestions"
            >
              <ChevronDown size={18} />
            </button>
          ) : null}
        </div>
        {!compact && showSuggestions ? (
          <div className="absolute left-0 right-0 top-[calc(100%+10px)] z-20 overflow-hidden rounded-2xl border border-[#d9dee8] bg-white shadow-[0_22px_48px_rgba(23,32,51,0.14)]">
            {EXAMPLE_QUERIES.map((example) => (
              <button
                key={example}
                type="button"
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => onPickSuggestion(example)}
                className="block w-full border-b border-[#edf1f7] px-4 py-3 text-left text-sm leading-5 text-[#2d3748] transition hover:bg-[#f7f9fc] last:border-b-0"
              >
                {example}
              </button>
            ))}
          </div>
        ) : null}
      </div>
      <button
        type="submit"
        disabled={!query.trim() || isLoading}
        className={`inline-flex items-center justify-center gap-2 rounded-full bg-[var(--accent)] font-medium text-white shadow-sm transition hover:bg-[var(--accent-strong)] disabled:cursor-not-allowed disabled:opacity-55 ${
          compact ? "h-11 px-5 text-sm" : "h-10 w-full px-5 text-sm sm:h-14 sm:w-auto sm:px-6"
        }`}
      >
        {isLoading ? <Loader2 className="animate-spin" size={17} /> : <Search size={17} />}
        Search
      </button>
    </form>
  );
}

export function SearchExperience() {
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [resultsState, setResultsState] = useState<ResultsState>({ status: "idle" });
  const [overviewState, setOverviewState] = useState<OverviewState>({ status: "idle", text: "" });
  const [retrieveLatencyMs, setRetrieveLatencyMs] = useState<number | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [showSuggestions, setShowSuggestions] = useState(false);

  useEffect(() => {
    let ignore = false;
    fetch("/api/health")
      .then((response) => (response.ok ? response.json() : null))
      .then((data: HealthResponse | null) => {
        if (!ignore) {
          setHealth(data);
        }
      })
      .catch(() => {
        if (!ignore) {
          setHealth(null);
        }
      });
    return () => {
      ignore = true;
    };
  }, []);

  const sources = resultsState.status === "success" ? resultsState.data.sources : [];
  const totalPages = Math.max(1, Math.ceil(sources.length / PAGE_SIZE));
  const paginatedSources = sources.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const answerSources = sources.slice(0, ANSWER_CONTEXT_K);
  const selectedSource = useMemo(() => {
    if (!sources.length) {
      return null;
    }
    return sources.find((source) => source.point_id === selectedId) || sources[0];
  }, [selectedId, sources]);

  function resetSearch() {
    setQuery("");
    setPage(1);
    setSelectedId(null);
    setResultsState({ status: "idle" });
    setOverviewState({ status: "idle", text: "" });
    setRetrieveLatencyMs(null);
    setShowSuggestions(false);
  }

  async function runSearch(nextQuery = query) {
    const normalized = nextQuery.trim();
    if (!normalized) {
      return;
    }
    setQuery(normalized);
    setPage(1);
    setSelectedId(null);
    setShowSuggestions(false);
    setResultsState({ status: "loading" });
    setOverviewState({ status: "loading", text: "" });
    setRetrieveLatencyMs(null);

    try {
      const retrieveStart = performance.now();
      const retrieveResponse = await fetch("/api/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: normalized,
          top_k: RETRIEVE_TOP_K,
          filters: {},
        }),
      });
      const retrieveElapsedMs = performance.now() - retrieveStart;
      const retrieveData = await retrieveResponse.json();
      if (!retrieveResponse.ok) {
        throw new Error(retrieveData.detail || "Retrieval failed");
      }
      setRetrieveLatencyMs(retrieveElapsedMs);

      setResultsState({ status: "success", data: retrieveData });
      setSelectedId(retrieveData.sources?.[0]?.point_id || null);

      let streamedText = "";
      setOverviewState({ status: "loading", text: "" });
      try {
        await streamOverview(normalized, retrieveData.sources.slice(0, ANSWER_CONTEXT_K), (piece) => {
          streamedText += piece;
          setOverviewState({ status: "loading", text: streamedText });
        });
        setOverviewState({ status: "success", text: streamedText.trim() });
      } catch (error) {
      setOverviewState({
        status: "error",
        text: streamedText.trim(),
        message: friendlyOverviewError(error instanceof Error ? error.message : "Overview failed"),
      });
    }
  } catch (error) {
    setResultsState({
      status: "error",
        message: error instanceof Error ? error.message : "Search failed",
      });
    setOverviewState({
      status: "error",
      text: "",
      message: friendlyOverviewError(error instanceof Error ? error.message : "Search failed"),
    });
  }
}

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void runSearch();
  }

  const hasResults = resultsState.status === "success";
  const isLoading = resultsState.status === "loading";
  const resultError = resultsState.status === "error" ? resultsState.message : null;
  const backendPill = healthPillState(health);

  return (
    <main className="min-h-screen">
      {hasResults || isLoading ? (
        <header className="sticky top-0 z-30 border-b border-[var(--line)] bg-white/92 backdrop-blur">
          <div className="mx-auto flex w-full max-w-[1500px] items-center gap-3 px-4 py-3 sm:px-5">
            <div className="hidden shrink-0 md:block">
              <button
                type="button"
                onClick={resetSearch}
                className="cursor-pointer text-left"
                aria-label="Go back to search home"
              >
                <ScholarRagLogo />
              </button>
            </div>
            <div className="md:hidden">
              <button
                type="button"
                onClick={resetSearch}
                className="grid size-10 cursor-pointer place-items-center rounded-xl bg-[#1558d6] text-white shadow-[0_16px_32px_rgba(21,88,214,0.24)]"
                aria-label="Go back to search home"
              >
                <LibraryBig size={18} />
              </button>
            </div>
            <div className="min-w-0 flex-1">
              <SearchForm
                query={query}
                setQuery={setQuery}
                onSubmit={onSubmit}
                isLoading={isLoading}
                compact
                showSuggestions={false}
                setShowSuggestions={() => {}}
                onPickSuggestion={(value) => void runSearch(value)}
              />
            </div>
          </div>
        </header>
      ) : null}

      {!hasResults && !isLoading ? (
        <section className="mx-auto grid min-h-screen w-full max-w-6xl content-center px-5 py-12 md:py-16">
          <div className="mb-8 flex justify-center">
            <ScholarRagLogo />
          </div>
          <div className="mb-8 text-center">
            <div
              className={`mb-3 inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${backendPill.className}`}
            >
              <Activity size={14} />
              {backendPill.label}
            </div>
            <div className="mx-auto max-w-3xl text-sm text-[var(--muted)]">
              Search over 1.5 million arXiv papers from 2020 onward.
            </div>
          </div>
          <div className="mb-8">
            <SearchForm
              query={query}
              setQuery={setQuery}
              onSubmit={onSubmit}
              isLoading={isLoading}
              showSuggestions={showSuggestions}
              setShowSuggestions={setShowSuggestions}
              onPickSuggestion={(value) => void runSearch(value)}
            />
          </div>
        </section>
      ) : (
        <section className="mx-auto grid w-full max-w-[1500px] grid-cols-1 gap-5 px-4 py-5 sm:px-5 xl:grid-cols-[minmax(0,1fr)_420px]">
          <div className="min-w-0">
            <SearchMeta
              health={health}
              resultCount={sources.length}
              loading={isLoading}
              latencyMs={retrieveLatencyMs}
            />
            <AiOverview
              state={overviewState}
              citationSources={answerSources}
            />
            <ResultsToolbar
              page={page}
              totalPages={totalPages}
              resultCount={sources.length}
              onPrev={() => setPage((value) => Math.max(1, value - 1))}
              onNext={() => setPage((value) => Math.min(totalPages, value + 1))}
            />
            <ResultsList
              loading={isLoading}
              sources={paginatedSources}
              selectedId={selectedSource?.point_id || null}
              offset={(page - 1) * PAGE_SIZE}
              onSelect={setSelectedId}
            />
            {resultError ? (
              <div className="mt-4 flex items-start gap-3 rounded-lg border border-[#f0c9c9] bg-[#fff7f7] p-4 text-sm text-[#9b1c1c]">
                <AlertCircle className="mt-0.5 shrink-0" size={18} />
                <span>{resultError}</span>
              </div>
            ) : null}
          </div>
          <PaperPreviewPanel source={selectedSource} loading={isLoading} />
        </section>
      )}
    </main>
  );
}

function SearchMeta({
  health,
  resultCount,
  loading,
  latencyMs,
}: {
  health: HealthResponse | null;
  resultCount: number;
  loading: boolean;
  latencyMs: number | null;
}) {
  return (
    <div className="mb-4 flex flex-wrap items-center gap-2 text-sm text-[var(--muted)]">
      <span>{loading ? "Searching" : `${resultCount} search results`}</span>
      <span>·</span>
      <span>{loading ? "Measuring latency" : formatLatencyMs(latencyMs)}</span>
      <span>·</span>
      <span>
        {health?.vector_backend
          ? health.vector_backend === "mock"
            ? "Mock backend"
            : `${health.vector_backend.toUpperCase()} backend`
          : "Backend status unknown"}
      </span>
    </div>
  );
}

function AiOverview({
  state,
  citationSources,
}: {
  state: OverviewState;
  citationSources: QuerySource[];
}) {
  const citationCount = citationSources.length;
  const renderedMarkdown = useMemo(
    () => linkifyCitations(state.text, citationCount),
    [state.text, citationCount],
  );

  return (
    <section className="relative mb-6 overflow-visible rounded-2xl border border-[#bfd5fb] bg-white shadow-[0_14px_36px_rgba(26,115,232,0.12)]">
      <div className="pointer-events-none absolute inset-x-4 -inset-y-2 -z-10 rounded-[28px] bg-[rgba(26,115,232,0.10)] blur-2xl" />
      <div className="flex items-center justify-between border-b border-[#e5e9f1] px-5 py-4">
        <div className="flex items-center gap-2">
          <div className="grid size-8 place-items-center rounded-md bg-[#e7f0ff] text-[var(--accent)]">
            <Sparkles size={18} />
          </div>
          <div>
            <h2 className="text-base font-semibold">AI Overview</h2>
            <p className="text-xs text-[var(--muted)]">Streamed from the retrieved abstracts</p>
          </div>
        </div>
        {state.status === "loading" ? (
          <Loader2 className="animate-spin text-[var(--accent)]" size={20} />
        ) : null}
      </div>
      <div className="px-5 py-4">
        {state.status === "idle" ? (
          <p className="text-sm text-[var(--muted)]">
            Search to retrieve papers first. The overview will stream here once evidence is in.
          </p>
        ) : null}
        {state.status === "loading" && !state.text ? (
          <div className="space-y-3">
            <div className="h-4 w-11/12 rounded bg-[#edf1f7]" />
            <div className="h-4 w-10/12 rounded bg-[#edf1f7]" />
            <div className="h-4 w-7/12 rounded bg-[#edf1f7]" />
          </div>
        ) : null}
        {state.text ? (
          <div className="scholarrag-markdown text-[15px] leading-7 text-[#253044]">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ children }) => <p className="mb-4 last:mb-0">{children}</p>,
                ul: ({ children }) => <ul className="mb-4 list-disc space-y-2 pl-6 last:mb-0">{children}</ul>,
                ol: ({ children }) => <ol className="mb-4 list-decimal space-y-2 pl-6 last:mb-0">{children}</ol>,
                li: ({ children }) => <li>{children}</li>,
                strong: ({ children }) => <strong className="font-semibold text-[#172033]">{children}</strong>,
                a: ({ href, children }) => {
                  if (href?.startsWith("#citation-")) {
                    const citationIndex = Number(href.slice("#citation-".length));
                    const source = citationSources[citationIndex - 1];
                    if (!source?.arxiv_url) {
                      return <span>{children}</span>;
                    }
                    return (
                      <a
                        href={source.arxiv_url}
                        target="_blank"
                        rel="noreferrer"
                        className="mx-0.5 inline-flex h-6 items-center rounded-full border border-[#c9dcfb] bg-[#eef5ff] px-2 text-[12px] font-semibold leading-none text-[#1558d6] no-underline transition hover:border-[#9fc2fb] hover:bg-[#e4efff] hover:text-[#0b57d0]"
                      >
                        [{children}]
                      </a>
                    );
                  }
                  return (
                    <a
                      href={href}
                      target="_blank"
                      rel="noreferrer"
                      className="font-medium text-[var(--accent)] underline decoration-[#9fc2fb] underline-offset-2"
                    >
                      {children}
                    </a>
                  );
                },
              }}
            >
              {renderedMarkdown}
            </ReactMarkdown>
          </div>
        ) : null}
        {state.status === "error" ? (
          <div className="mt-3 flex items-start gap-3 text-sm text-[#9b1c1c]">
            <AlertCircle className="mt-0.5 shrink-0" size={18} />
            <span>{state.message}</span>
          </div>
        ) : null}
      </div>
    </section>
  );
}

function ResultsToolbar({
  page,
  totalPages,
  resultCount,
  onPrev,
  onNext,
}: {
  page: number;
  totalPages: number;
  resultCount: number;
  onPrev: () => void;
  onNext: () => void;
}) {
  return (
    <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <div className="text-sm font-medium text-[#2d3748]">Ranked papers</div>
      <div className="flex items-center gap-2 text-sm text-[var(--muted)]">
        <span>
          Page {page} of {totalPages} · {resultCount} results
        </span>
        <button
          onClick={onPrev}
          disabled={page <= 1}
          className="grid size-8 place-items-center rounded-full border border-[#d8dfeb] bg-white disabled:opacity-40"
          aria-label="Previous page"
        >
          <ChevronLeft size={17} />
        </button>
        <button
          onClick={onNext}
          disabled={page >= totalPages}
          className="grid size-8 place-items-center rounded-full border border-[#d8dfeb] bg-white disabled:opacity-40"
          aria-label="Next page"
        >
          <ChevronRight size={17} />
        </button>
      </div>
    </div>
  );
}

function ResultsList({
  loading,
  sources,
  selectedId,
  offset,
  onSelect,
}: {
  loading: boolean;
  sources: QuerySource[];
  selectedId: string | null;
  offset: number;
  onSelect: (id: string) => void;
}) {
  if (loading) {
    return (
      <div className="space-y-3">
        {Array.from({ length: 6 }).map((_, index) => (
          <div key={index} className="rounded-lg border border-[#e0e5ef] bg-white p-5 shadow-sm">
            <div className="mb-3 h-5 w-3/4 rounded bg-[#edf1f7]" />
            <div className="mb-2 h-4 w-1/2 rounded bg-[#edf1f7]" />
            <div className="h-4 w-full rounded bg-[#edf1f7]" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {sources.map((source, index) => (
        <article
          key={source.point_id}
          id={`result-${source.point_id}`}
          onClick={() => onSelect(source.point_id)}
          className={`rounded-lg border bg-white p-4 shadow-sm transition hover:border-[#aeb9cc] hover:shadow-md sm:p-5 xl:cursor-pointer ${
            selectedId === source.point_id ? "border-[#84aef0]" : "border-[#dfe5ef]"
          }`}
        >
          <div className="mb-2 flex flex-wrap items-start justify-between gap-3">
            <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs text-[var(--muted)]">
              <span className="font-semibold text-[var(--accent)]">[{offset + index + 1}]</span>
              <span>{source.paper_id || "unknown arXiv id"}</span>
              <span>·</span>
              <span>{source.update_date || "unknown date"}</span>
              <span>·</span>
              <span>confidence {confidenceLabel(source)}</span>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              {source.arxiv_url ? (
                <a
                  href={source.arxiv_url}
                  target="_blank"
                  rel="noreferrer"
                  onClick={(event) => event.stopPropagation()}
                  className="inline-flex h-8 items-center justify-center gap-1 rounded-full border border-[#d8dfeb] bg-white px-3 text-xs font-medium text-[#253044] transition hover:border-[#aeb9cc] hover:bg-[#f8faff] xl:hidden"
                  aria-label={`Open ${source.title || "paper"} on arXiv`}
                >
                  arXiv
                  <ExternalLink size={14} />
                </a>
              ) : null}
              {pdfUrl(source) ? (
                <a
                  href={pdfUrl(source) || undefined}
                  target="_blank"
                  rel="noreferrer"
                  onClick={(event) => event.stopPropagation()}
                  className="inline-flex h-8 items-center justify-center gap-1 rounded-full border border-[#d8dfeb] bg-white px-3 text-xs font-medium text-[#253044] transition hover:border-[#aeb9cc] hover:bg-[#f8faff] xl:hidden"
                  aria-label={`Open PDF for ${source.title || "paper"}`}
                >
                  PDF
                  <ArrowUpRight size={14} />
                </a>
              ) : null}
            </div>
          </div>
          <h3 className="mb-2 text-base font-medium leading-6 text-[#102a5c] sm:text-lg">
            {source.arxiv_url ? (
              <a
                href={source.arxiv_url}
                target="_blank"
                rel="noreferrer"
                onClick={(event) => event.stopPropagation()}
                className="transition hover:text-[#1558d6] hover:underline"
              >
                {source.title || "Untitled paper"}
              </a>
            ) : (
              source.title || "Untitled paper"
            )}
          </h3>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            {source.categories.map((category) => (
              <span key={category} className="rounded-full bg-[#eef2f8] px-2 py-1 text-xs text-[#40516a]">
                {category}
              </span>
            ))}
          </div>
          <p className="line-clamp-3 text-sm leading-6 text-[#3f4c61]">{source.text_preview}</p>
        </article>
      ))}
    </div>
  );
}

function PaperPreviewPanel({
  source,
  loading,
}: {
  source: QuerySource | null;
  loading: boolean;
}) {
  return (
    <aside className="hidden h-fit rounded-lg border border-[#d8dfeb] bg-white shadow-sm xl:sticky xl:top-[88px] xl:block">
      <div className="border-b border-[#e5e9f1] px-5 py-4">
        <div className="flex items-center gap-2">
          <FileText size={18} className="text-[var(--accent)]" />
          <h2 className="text-base font-semibold">Paper preview</h2>
        </div>
      </div>
      <div className="p-5">
        {loading ? (
          <div className="space-y-3">
            <div className="h-5 w-4/5 rounded bg-[#edf1f7]" />
            <div className="h-4 w-2/3 rounded bg-[#edf1f7]" />
            <div className="h-32 rounded bg-[#edf1f7]" />
          </div>
        ) : null}
        {!loading && !source ? (
          <p className="text-sm text-[var(--muted)]">Select a retrieved paper to inspect it here.</p>
        ) : null}
        {!loading && source ? (
          <div>
            <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-[var(--muted)]">
              <span>{source.paper_id}</span>
              <span>·</span>
              <Clock3 size={13} />
              <span>{source.update_date || "unknown date"}</span>
              <span>·</span>
              <span>confidence {confidenceLabel(source)}</span>
            </div>
            <h3 className="mb-3 text-xl font-semibold leading-7 text-[#172033]">
              {source.title || "Untitled paper"}
            </h3>
            <div className="mb-4 flex flex-wrap gap-2">
              {source.categories.map((category) => (
                <span key={category} className="rounded-full bg-[#eef2f8] px-2 py-1 text-xs text-[#40516a]">
                  {category}
                </span>
              ))}
            </div>
            <p className="mb-5 max-h-64 overflow-auto pr-2 text-sm leading-6 text-[#3f4c61] scrollbar-thin">
              {source.text || source.text_preview}
            </p>
            <div className="grid gap-2">
              {source.arxiv_url ? (
                <a
                  href={source.arxiv_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex h-10 items-center justify-center gap-2 rounded-md bg-[var(--accent)] px-3 text-sm font-medium text-white hover:bg-[var(--accent-strong)]"
                >
                  Open arXiv
                  <ExternalLink size={16} />
                </a>
              ) : null}
              {pdfUrl(source) ? (
                <a
                  href={pdfUrl(source) || undefined}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex h-10 items-center justify-center gap-2 rounded-md border border-[#d8dfeb] bg-white px-3 text-sm font-medium text-[#253044] hover:border-[#aeb9cc]"
                >
                  Open PDF
                  <ArrowUpRight size={16} />
                </a>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>
    </aside>
  );
}
