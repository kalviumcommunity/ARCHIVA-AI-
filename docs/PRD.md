# Product Requirements Document (PRD) — Archiva AI

## 1. Executive Summary
**Archiva AI** is an intelligent engineering knowledge retrieval and contextual intelligence platform designed to eliminate organizational knowledge silos. It surfaces historical Architecture Decision Records (ADRs), incident postmortems, design specifications, and operational runbooks in response to natural-language engineering inquiries, delivering grounded answers with explicit citations.

---

## 2. Problem Statement
Software engineering organizations suffer from severe knowledge fragmentation:
- Critical architectural decisions and incident postmortems are stored across disjointed systems (Confluence, Google Docs, Notion, Slack, Jira, GitHub).
- New engineers or engineers operating in adjacent domains lack institutional context when designing systems or debugging critical outages.
- Teams frequently re-engineer previously solved problems (e.g., payment idempotency, database connection pooling, distributed rate limiting) or repeat past mistakes.

---

## 3. Goals & Success Criteria

### 3.1 Primary Goals
1. **Accurate Context Retrieval**: Given any natural-language technical query, retrieve the most relevant historical engineering documents using dense vector semantic search.
2. **Grounded Synthesis**: Generate an answer strictly based on retrieved organization artifacts, explicitly preventing hallucination or invention of organizational facts.
3. **Structured Source Attribution**: Every answer must provide machine-readable and human-verifiable citations (document title, document type, team, service, and similarity score).
4. **Zero-Failure Fallback**: The platform must operate seamlessly offline or when external LLM APIs are unavailable.

### 3.2 Success Metrics
- **Retrieval Precision@3**: > 90% relevance on canonical engineering queries.
- **Hallucination Rate**: 0% on ungrounded technical facts.
- **P95 Retrieval Latency**: < 100ms for in-memory dense vector search.

---

## 4. User Personas
1. **Senior Software Engineer**: Seeking prior architecture decisions before writing a new RFC or system design document.
2. **On-Call SRE / DevOps Engineer**: Triaging production outages and looking for historical postmortems or runbooks dealing with similar symptoms (e.g., Redis cluster split-brain, database connection pool exhaustion).
3. **New Hire / Junior Engineer**: Ramp-up phase to understand why certain technical choices were made (e.g., why JWT with RS256 was chosen, or why database sharding was implemented).

---

## 5. Functional Requirements

### 5.1 Knowledge Ingestion & Indexing
- Standardized chunk schema: `chunk_id`, `document_id`, `title`, `document_type`, `content`, `team`, `service`.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- High performance in-memory cosine similarity indexing.

### 5.2 Retrieval & Querying (API)
- `POST /search`: Accepts natural-language query and `top_k` parameter; returns ranked documents with cosine similarity scores.
- `POST /ask`: Performs vector retrieval + context construction + grounded answer synthesis + citation packaging.
- `GET /documents`: Returns all indexed documents and metadata for exploration.

### 5.3 Grounded RAG & LLM Integration
- Strict prompt template ensuring no hallucinations.
- Google Gemini API integration (`gemini-2.5-flash` / `gemini-1.5-flash`).
- Deterministic heuristic fallback engine for offline environments.

### 5.4 User Feedback Loop
- `POST /feedback`: Captures binary helpfulness (`helpful: true/false`) and optional user comments for continuous evaluation.

---

## 6. User Interface Requirements
- Clean, responsive web dashboard (dark theme tailored for developers).
- Natural language query input with fast submit and suggested query chips.
- Clear separation between grounded answer and structured source cards.
- Interactive Knowledge Base viewer modal.
- Real-time backend connectivity status indicator.
