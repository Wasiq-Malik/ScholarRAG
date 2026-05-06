# ScholaRAG Web App

Next.js frontend for searching the ScholaRAG API.

## Local setup

```bash
cd scholarrag/web_app
cp .env.example .env.local
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

The default `.env.example` points to the fixed Colab/ngrok backend:

```text
SCHOLARRAG_API_BASE_URL=https://complete-jay-strictly.ngrok-free.app
SCHOLARRAG_USE_MOCK_API=true
```

While the backend is offline or mid-indexing, keep `SCHOLARRAG_USE_MOCK_API=true`
to render a realistic mock scientific RAG response set. Set it to `false` when
you want the UI to hit the live backend.

## Current behavior

- Calls the backend through Next route handlers at `/api/health` and `/api/query`.
- Uses the existing blocking backend `/query` endpoint.
- Renders a Google-style search page, AI Overview, ranked arXiv results, pagination, and a right-side paper preview.

## Next backend dependency

For true progressive rendering, the backend should later expose:

- `POST /retrieve` for fast source retrieval.
- `POST /answer/stream` for streaming the AI Overview.

Until then, the UI shows loading states and renders sources once `/query` returns.
