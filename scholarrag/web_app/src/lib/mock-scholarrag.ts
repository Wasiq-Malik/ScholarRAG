import type { HealthResponse, QueryResponse, QuerySource } from "@/types/scholarrag";

type MockPaper = {
  paper_id: string;
  title: string;
  categories: string[];
  update_date: string;
  score: number;
  text_preview: string;
};

const MOCK_PAPERS: MockPaper[] = [
  {
    paper_id: "2510.15682",
    title: "SQuAI: Scientific Question-Answering with Multi-Agent Retrieval-Augmented Generation",
    categories: ["cs.IR", "cs.CL"],
    update_date: "2025-10-20",
    score: 0.94,
    text_preview:
      "SQuAI presents a multi-agent retrieval-augmented generation framework for scientific question answering over millions of arXiv papers, combining hybrid retrieval, evidence selection, and citation-grounded answer generation.",
  },
  {
    paper_id: "2602.17856",
    title: "Enhancing Scientific Literature Chatbots with Retrieval-Augmented Generation: A Performance Evaluation of Vector and Graph-Based Systems",
    categories: ["cs.IR", "cs.AI"],
    update_date: "2026-02-23",
    score: 0.92,
    text_preview:
      "This paper evaluates vector and graph retrieval backends for scientific literature chatbots and studies how retrieval quality affects grounded generation across domain-specific search scenarios.",
  },
  {
    paper_id: "2601.21733",
    title: "CE-GOCD: Central Entity-Guided Graph Optimization for Community Detection to Augment LLM Scientific Question Answering",
    categories: ["cs.CL"],
    update_date: "2026-01-30",
    score: 0.89,
    text_preview:
      "CE-GOCD augments scientific question answering by extracting graph structure across related papers, improving how language models retrieve and reason over scientific evidence.",
  },
  {
    paper_id: "2603.14257",
    title: "Automatic Inter-document Multi-hop Scientific QA Generation",
    categories: ["cs.CL"],
    update_date: "2026-03-17",
    score: 0.87,
    text_preview:
      "AIM-SciQA generates multi-document scientific question answering data and is positioned as a benchmark for retrieval-augmented scientific reasoning and evidence synthesis.",
  },
  {
    paper_id: "2602.22215",
    title: "Graph Your Way to Inspiration: Integrating Co-Author Graphs with Retrieval-Augmented Generation for Large Language Model Based Scientific Idea Generation",
    categories: ["cs.AI", "cs.CL", "cs.IR"],
    update_date: "2026-02-27",
    score: 0.84,
    text_preview:
      "The system combines RAG with scholarly graph structure to steer generation over research corpora, showing how graph-enhanced retrieval can shape scientific exploration workflows.",
  },
  {
    paper_id: "2508.04112",
    title: "Evidence-Grounded Survey Generation for Rapid Literature Review",
    categories: ["cs.CL", "cs.DL"],
    update_date: "2025-08-08",
    score: 0.82,
    text_preview:
      "This work studies fast survey writing from retrieved abstract evidence, with explicit support for citation-grounded summaries and controllable breadth over technical domains.",
  },
  {
    paper_id: "2507.11890",
    title: "Faithful Citation Planning for Long-Context Scientific Assistants",
    categories: ["cs.CL"],
    update_date: "2025-07-19",
    score: 0.8,
    text_preview:
      "The paper focuses on citation planning and evidence linking inside scientific assistants, aiming to keep generated answers faithful to retrieved abstracts and full papers.",
  },
  {
    paper_id: "2509.20044",
    title: "Hybrid Sparse-Dense Retrieval for Open-Domain Research Assistants",
    categories: ["cs.IR", "cs.AI"],
    update_date: "2025-09-28",
    score: 0.78,
    text_preview:
      "A hybrid retrieval stack for technical assistants that blends lexical and embedding-based retrieval, with an emphasis on recall for evolving scientific terminology.",
  },
  {
    paper_id: "2604.01118",
    title: "Structured Evidence Panels for Scholarly Search Interfaces",
    categories: ["cs.HC", "cs.IR"],
    update_date: "2026-04-03",
    score: 0.76,
    text_preview:
      "This UI-oriented paper explores dense scholarly result layouts with evidence panels, inline citations, and reading-optimized previews for expert users.",
  },
  {
    paper_id: "2506.30010",
    title: "Question-Focused Retrieval for Biomedical and Computer Science Literature",
    categories: ["cs.IR", "q-bio.QM"],
    update_date: "2025-06-30",
    score: 0.74,
    text_preview:
      "Question-focused retrieval improves retrieval precision for specialized scientific domains by conditioning ranking on claim style and expected evidence type.",
  },
  {
    paper_id: "2601.07555",
    title: "Long-Form Answer Streaming for Search-Centric AI Systems",
    categories: ["cs.HC", "cs.CL"],
    update_date: "2026-01-14",
    score: 0.72,
    text_preview:
      "This paper examines UI patterns for streaming search-grounded answers while ranked results render in parallel, including citation linking and fallback disclosure.",
  },
  {
    paper_id: "2505.22991",
    title: "Retrieval Over Abstracts Versus Full Papers for Technical Question Answering",
    categories: ["cs.IR", "cs.CL"],
    update_date: "2025-05-29",
    score: 0.71,
    text_preview:
      "The authors compare abstract-only retrieval with full-paper retrieval for technical QA, showing strong early wins from high-quality abstract retrieval before moving to full text.",
  },
];

function buildMockSource(paper: MockPaper, index: number): QuerySource {
  return {
    point_id: `mock-${paper.paper_id}-${index}`,
    score: paper.score,
    confidence_score: paper.score,
    paper_id: paper.paper_id,
    arxiv_url: `https://arxiv.org/abs/${paper.paper_id}`,
    chunk_id: 0,
    title: paper.title,
    categories: paper.categories,
    update_date: paper.update_date,
    text: paper.text_preview,
    text_preview: paper.text_preview,
  };
}

export function mockHealthResponse(): HealthResponse {
  return {
    status: "ok",
    vector_backend: "mock",
    faiss_index_exists: null,
    faiss_sqlite_exists: null,
    embedding_model: "google/embeddinggemma-300m",
    embedding_dimension: 768,
    llm_model: "gemma-4-31b-it",
    detail: "Mock mode",
  } as HealthResponse;
}

export function mockQueryResponse(question: string, topK: number): QueryResponse {
  const lowered = question.toLowerCase();
  const ranked = [...MOCK_PAPERS].sort((left, right) => {
    const leftBoost =
      Number(lowered.includes("retrieval") && left.title.toLowerCase().includes("retrieval")) +
      Number(lowered.includes("question") && left.title.toLowerCase().includes("question")) +
      Number(lowered.includes("scientific") && left.title.toLowerCase().includes("scientific"));
    const rightBoost =
      Number(lowered.includes("retrieval") && right.title.toLowerCase().includes("retrieval")) +
      Number(lowered.includes("question") && right.title.toLowerCase().includes("question")) +
      Number(lowered.includes("scientific") && right.title.toLowerCase().includes("scientific"));
    if (rightBoost !== leftBoost) {
      return rightBoost - leftBoost;
    }
    return right.score - left.score;
  });

  const selected = ranked.slice(0, topK).map(buildMockSource);

  return {
    answer:
      "Recent work in this mock ScholaRAG corpus points to several credible directions for scientific RAG. SQuAI frames scientific question answering as a multi-agent retrieval-augmented workflow over large arXiv-scale corpora [1]. Vector and graph retrieval tradeoffs are evaluated directly for scientific literature chatbots in a dedicated RAG setting [2]. Other papers in the set focus on graph-augmented retrieval for scientific QA [3], synthetic benchmark generation for multi-hop scientific QA [4], and interface patterns for streaming evidence-grounded answers in search workflows [5].",
    sources: selected,
    retrieval: {
      candidate_k: Math.max(50, topK),
      returned_k: selected.length,
      collection: "mock:recent-scientific-rag",
      backend: "mock",
      reranker: null,
      search_ms: 86,
    },
    models: {
      embedding: "google/embeddinggemma-300m",
      llm: "gemma-4-31b-it",
    },
  };
}
