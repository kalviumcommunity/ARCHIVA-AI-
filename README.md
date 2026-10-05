# Archiva AI — Engineering Knowledge Intelligence & Contextual Retrieval Platform

> **Archiva AI** allows software engineers to ask natural-language questions about their organization's engineering history and retrieves historical Architecture Decision Records (ADRs), postmortems, design specifications, and runbooks to produce grounded answers with source citations.

---

## 1. Problem Statement
In modern engineering organizations, critical knowledge is scattered across fragmented systems (Google Docs, Notion, Slack, Jira, GitHub, incident trackers). When engineers encounter tough problems—such as payment duplication, database connection exhaustion, or caching invalidation—they often cannot find historical solutions and end up reinventing the wheel or repeating previous mistakes.

## 2. Solution
Archiva AI indexes engineering artifacts into semantic vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`. When an engineer queries the platform, it:
1. Performs dense semantic vector search via cosine similarity.
2. Identifies the most relevant historical engineering documents (ADRs, Postmortems, Design Specs, Runbooks).
3. Constructs a grounded context prompt for retrieval-augmented generation (RAG).
4. Produces a synthesized, factual answer with precise source citations (or a deterministic fallback if no external LLM API key is present).

---

## 3. End-to-End Architecture

```text
Engineering Documents (ADRs, Postmortems, Specs, Runbooks)
                          ↓
                 Document Processing
                          ↓
                       Chunking
                          ↓
        Embeddings (all-MiniLM-L6-v2, 384d)
                          ↓
              Vector Retrieval (Cosine Similarity)
                          ↓
                   Relevant Context
                          ↓
                      RAG Pipeline
                          ↓
                 LLM (Gemini / Grounded Fallback)
                          ↓
              Answer + Structured Citations
```

---

## 4. Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, Pydantic v2
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2` — 384 dimensions)
- **Vector Math**: NumPy (Vectorized Cosine Similarity & Dot Product)
- **LLM Synthesis**: Google Gemini (`google-genai` / `google-generativeai`) with built-in deterministic offline fallback
- **Testing**: Pytest, FastAPI TestClient
- **Frontend**: Vanilla HTML5, Modern CSS3 (Dark Slate Design System), JavaScript (ES6+)

---

## 5. Project Structure

```
ARCHIVA-AI-/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI application & endpoints
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py              # Pydantic schemas (Search, Ask, Feedback)
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   └── pipeline.py             # Grounded RAG & LLM orchestration
│   │   └── retrieval/
│   │       ├── __init__.py
│   │       ├── embeddings.py           # Singleton embedding model loader
│   │       └── search.py               # Vector retrieval & cosine similarity
│   ├── data/
│   │   ├── sample_chunks.json          # 12+ realistic engineering knowledge chunks
│   │   └── feedback.json               # Recorded user evaluation feedback
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_archiva.py             # 12 unit and integration tests
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── test_embedding.py
│   └── test_search.py
├── frontend/
│   ├── index.html                      # Clean, responsive user interface
│   ├── style.css                       # Modern dark glassmorphic styling
│   └── app.js                          # Client-side API integration & modal logic
├── docs/
│   ├── PRD.md                          # Product Requirements Document
│   └── architecture.md                 # Deep-dive architecture & sequence flows
├── .env.example
├── pytest.ini
└── README.md
```

---

## 6. How Retrieval Works

1. **Embedding Generation**: At startup, `embeddings.py` loads `all-MiniLM-L6-v2` into memory once.
2. **Indexing**: Document chunks from `backend/data/sample_chunks.json` are encoded into 384-dimensional dense vectors and cached as a NumPy float32 matrix.
3. **Query Vectorization**: When a search or ask request arrives, the natural-language query is encoded into a 384-dimensional vector.
4. **Cosine Similarity**: Vector dot products and Euclidean norms are computed:
   $$\text{Similarity}(q, d) = \frac{q \cdot d}{\|q\|_2 \times \|d\|_2}$$
5. **Ranking**: Results are sorted descending by similarity score, returning the top-$k$ relevant chunks with document metadata.

---

## 7. How RAG & Grounded Synthesis Work

1. **Context Construction**: The top relevant chunks are formatted with their `document_id`, `title`, `document_type`, `team`, and `service`.
2. **Grounding Prompt**: A strict prompt instructs the model to answer *only* using supplied engineering evidence without inventing facts.
3. **LLM Provider**:
   - If `GEMINI_API_KEY` is present in the environment, the prompt is submitted to Google Gemini (`gemini-2.5-flash` / `gemini-1.5-flash`).
   - If `GEMINI_API_KEY` is missing or the network request fails, Archiva AI generates a **deterministic fallback synthesis** summarizing the matched ADRs, postmortems, and runbooks.
4. **Structured Citations**: The API returns structured citation objects alongside the text answer so user interfaces can render source cards with similarity metrics.

---

## 8. Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone & Setup Backend Virtual Environment
```bash
# Clone the repository
git clone https://github.com/kalviumcommunity/ARCHIVA-AI-.git
cd ARCHIVA-AI-

# Install backend dependencies
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
Copy `.env.example` to `.env` in the backend directory:
```bash
cp .env.example .env
```
Add your Gemini API key if available:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Note: Archiva AI works completely out of the box even without an API key thanks to its built-in deterministic grounding engine).*

---

## 9. Running the Application

### 1. Start the FastAPI Backend Server
From the `backend` directory:
```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Or from the repository root:
```bash
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Verify backend health in browser or curl:
```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","service":"Archiva AI","version":"1.0.0"}
```

API Documentation (Swagger UI):
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Start the Frontend
Open `frontend/index.html` directly in your browser or run a local static server:
```bash
cd frontend
python -m http.server 3000
```
Then visit [http://127.0.0.1:3000](http://127.0.0.1:3000) in your web browser.

---

## 10. API Endpoints

| Method | Path | Description | Request Body |
|---|---|---|---|
| `GET` | `/health` | Service health check | None |
| `POST` | `/search` | Dense vector semantic search | `{"query": "...", "top_k": 5}` |
| `POST` | `/ask` | Grounded RAG answer with citations | `{"query": "..."}` |
| `GET` | `/documents` | List all indexed engineering documents | None |
| `POST` | `/feedback` | Submit answer evaluation feedback | `{"query": "...", "helpful": true, "comment": "..."}` |

---

## 11. Demo Queries for Viva Presentation

Try these questions in the UI or via API:

1. **"Have we solved duplicate payment processing before?"**
   - *Expected Citations*: `PM-001` (Duplicate Payment Incident), `ADR-001` (Payment Idempotency)
2. **"How did we handle Redis failures?"**
   - *Expected Citations*: `RB-002` (Redis Cluster Failover Runbook), `ADR-002` (Redis Caching Strategy)
3. **"How did we solve API latency?"**
   - *Expected Citations*: `PM-003` (API Latency Spike Incident), `ADR-005` (Distributed Rate Limiting)
4. **"Why did we choose database sharding?"**
   - *Expected Citations*: `ADR-004` (Database Sharding Architecture)
5. **"How did we handle authentication?"**
   - *Expected Citations*: `ADR-003` (Authentication Strategy & JWT Architecture)

---

## 12. Running Tests

Run the complete 12-test suite:
```bash
python -m pytest backend/tests/test_archiva.py -v
```

---

## 13. Limitations & Future Improvements

- **Current Limitations (MVP)**:
  - In-memory vector matrix calculation (suitable for thousands of chunks; for millions, external vector stores like pgvector or Qdrant are recommended).
  - Synchronous document ingestion from JSON.
- **Future Roadmap**:
  - Continuous document ingestion pipeline connecting directly to GitHub PRs and Confluence.
  - Hybrid search combining BM25 keyword search with dense vector embeddings (Reciprocal Rank Fusion).
  - Role-based document access control (RBAC) across confidential engineering domains.
