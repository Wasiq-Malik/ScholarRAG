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

import type { HealthResponse, QuerySource, RetrieveResponse } from "@/types/scholarrag";

const EXAMPLE_QUERIES = [
  "retrieval augmented generation for scientific question answering",
  "graph retrieval augmented generation for literature review",
  "hallucination detection in large language models",
];

const PAGE_SIZE = 10;
const RETRIEVE_TOP_K = 50;

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
        onChunk(payload.text);
      }
      if (eventName === "error") {
        throw new Error(payload.detail || "Overview streaming failed");
      }
      if (eventName === "done") {
        return;
      }
    }
  }
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
  const selectedSource = useMemo(() => {
    if (!sources.length) {
      return null;
    }
    return sources.find((source) => source.point_id === selectedId) || sources[0];
  }, [selectedId, sources]);

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

    try {
      const retrieveResponse = await fetch("/api/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: normalized,
          top_k: RETRIEVE_TOP_K,
          filters: {},
        }),
      });
      const retrieveData = await retrieveResponse.json();
      if (!retrieveResponse.ok) {
        throw new Error(retrieveData.detail || "Retrieval failed");
      }

      setResultsState({ status: "success", data: retrieveData });
      setSelectedId(retrieveData.sources?.[0]?.point_id || null);

      let streamedText = "";
      setOverviewState({ status: "loading", text: "" });
      try {
        await streamOverview(normalized, retrieveData.sources, (piece) => {
          streamedText += piece;
          setOverviewState({ status: "loading", text: streamedText });
        });
        setOverviewState({ status: "success", text: streamedText.trim() });
      } catch (error) {
        setOverviewState({
          status: "error",
          text: streamedText.trim(),
          message: error instanceof Error ? error.message : "Overview failed",
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
        message: error instanceof Error ? error.message : "Search failed",
      });
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void runSearch();
  }

  const hasResults = resultsState.status === "success";
  const isLoading = resultsState.status === "loading";

  return (
    <main className="min-h-screen">
      {hasResults || isLoading ? (
        <header className="sticky top-0 z-30 border-b border-[var(--line)] bg-white/92 backdrop-blur">
          <div className="mx-auto flex w-full max-w-[1500px] items-center gap-3 px-4 py-3 sm:px-5">
            <div className="hidden shrink-0 md:block">
              <ScholarRagLogo />
            </div>
            <div className="md:hidden">
              <div className="grid size-10 place-items-center rounded-xl bg-[#1558d6] text-white shadow-[0_16px_32px_rgba(21,88,214,0.24)]">
                <LibraryBig size={18} />
              </div>
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
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-[#d6ddeb] bg-white px-3 py-1 text-xs font-medium text-[var(--muted)]">
              <Activity size={14} />
              {health
                ? health.vector_backend === "mock"
                  ? "Mock dataset active"
                  : `${health.vector_backend.toUpperCase()} index online`
                : "Checking backend"}
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
              latencyMs={
                resultsState.status === "success"
                  ? Number(resultsState.data.retrieval.search_ms ?? resultsState.data.retrieval.latency_ms ?? NaN)
                  : null
              }
            />
            <AiOverview state={overviewState} />
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
            {resultsState.status === "error" ? (
              <div className="mt-4 flex items-start gap-3 rounded-lg border border-[#f0c9c9] bg-[#fff7f7] p-4 text-sm text-[#9b1c1c]">
                <AlertCircle className="mt-0.5 shrink-0" size={18} />
                <span>{resultsState.message}</span>
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
      <span>{loading ? "Retrieving papers" : `${resultCount} retrieved papers`}</span>
      <span>·</span>
      <span>{loading ? "Measuring latency" : formatLatencyMs(latencyMs)}</span>
      <span>·</span>
      <span>
        {health?.vector_backend
          ? health.vector_backend === "mock"
            ? "Mock preview mode"
            : `${health.vector_backend.toUpperCase()} backend`
          : "Backend status unknown"}
      </span>
      <span className="inline-flex items-center gap-1 rounded-full bg-white px-2 py-1 text-xs">
        <Sparkles size={13} />
        AI overview stream
      </span>
    </div>
  );
}

function AiOverview({ state }: { state: OverviewState }) {
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
          <p className="whitespace-pre-wrap text-[15px] leading-7 text-[#253044]">{state.text}</p>
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
          className={`rounded-lg border bg-white p-4 shadow-sm transition hover:border-[#aeb9cc] hover:shadow-md sm:p-5 ${
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
                  className="inline-flex h-8 items-center justify-center gap-1 rounded-full border border-[#d8dfeb] bg-white px-3 text-xs font-medium text-[#253044] transition hover:border-[#aeb9cc] hover:bg-[#f8faff]"
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
                  className="inline-flex h-8 items-center justify-center gap-1 rounded-full border border-[#d8dfeb] bg-white px-3 text-xs font-medium text-[#253044] transition hover:border-[#aeb9cc] hover:bg-[#f8faff]"
                  aria-label={`Open PDF for ${source.title || "paper"}`}
                >
                  PDF
                  <ArrowUpRight size={14} />
                </a>
              ) : null}
              <button
                type="button"
                onClick={() => onSelect(source.point_id)}
                className="hidden h-8 items-center justify-center rounded-full border border-[#d8dfeb] bg-white px-3 text-xs font-medium text-[#253044] transition hover:border-[#aeb9cc] hover:bg-[#f8faff] xl:inline-flex"
              >
                Inspect
              </button>
            </div>
          </div>
          <h3 className="mb-2 text-base font-medium leading-6 text-[#102a5c] sm:text-lg">
            {source.arxiv_url ? (
              <a
                href={source.arxiv_url}
                target="_blank"
                rel="noreferrer"
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
