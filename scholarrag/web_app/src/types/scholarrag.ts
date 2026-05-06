export type QuerySource = {
  point_id: string;
  score: number;
  confidence_score: number;
  paper_id: string | null;
  arxiv_url: string | null;
  chunk_id: number | null;
  title: string | null;
  categories: string[];
  update_date: string | null;
  text: string | null;
  text_preview: string;
};

export type RetrieveResponse = {
  sources: QuerySource[];
  retrieval: {
    candidate_k?: number;
    returned_k?: number;
    collection?: string;
    backend?: string;
    reranker?: string | null;
    [key: string]: unknown;
  };
  models: {
    embedding: string;
    llm: string;
  };
};

export type QueryResponse = RetrieveResponse & {
  answer: string;
};

export type HealthResponse = {
  status: string;
  vector_backend: string;
  faiss_index_exists?: boolean | null;
  faiss_sqlite_exists?: boolean | null;
  embedding_model: string;
  embedding_dimension: number;
  llm_model: string;
  detail?: string;
};
