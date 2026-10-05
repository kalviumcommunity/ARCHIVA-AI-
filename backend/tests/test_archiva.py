import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.retrieval.embeddings import generate_embedding, generate_embeddings, get_model
from app.retrieval.search import search, get_all_documents, cosine_similarity
from app.rag.pipeline import answer_question, generate_deterministic_fallback
import numpy as np

client = TestClient(app)


# 1. Health endpoint test
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "Archiva" in data.get("service", "")


# 2. Embedding generation test
def test_embedding_generation():
    text = "We introduced idempotency keys to prevent duplicate transactions."
    emb = generate_embedding(text)
    assert isinstance(emb, list)
    assert len(emb) > 0
    assert all(isinstance(val, float) for val in emb)


# 3. Embedding dimensions test (384 dimensions for all-MiniLM-L6-v2)
def test_embedding_dimensions():
    model = get_model()
    text = "Redis caching for latency reduction"
    emb = generate_embedding(text)
    assert len(emb) == 384

    # Test batch embeddings
    batch = ["first query", "second query"]
    embs = generate_embeddings(batch)
    assert len(embs) == 2
    assert len(embs[0]) == 384
    assert len(embs[1]) == 384


# 4. Retrieval returns results
def test_retrieval_returns_results():
    results = search(query="database query optimization", top_k=3)
    assert isinstance(results, list)
    assert len(results) == 3
    for r in results:
        assert "chunk_id" in r
        assert "document_id" in r
        assert "title" in r
        assert "document_type" in r
        assert "content" in r
        assert "score" in r
        assert isinstance(r["score"], float)


# 5. Duplicate payment query retrieves payment-related documents
def test_duplicate_payment_retrieval():
    query = "Have we solved duplicate payment processing before?"
    results = search(query=query, top_k=3)
    
    titles = [r["title"] for r in results]
    # The top 2 should include Duplicate Payment Incident and Payment Idempotency
    top_titles = titles[:2]
    assert any("Payment" in t for t in top_titles)
    assert any("Duplicate" in t or "Idempotency" in t for t in top_titles)


# 6. Unrelated queries ranking behavior
def test_ranking_relevance():
    payment_query = "payment idempotency keys"
    payment_results = search(query=payment_query, top_k=5)
    
    # Verify top result is payment-related with higher score than unrelated docs like notification service
    top_doc = payment_results[0]
    assert "Payment" in top_doc["title"] or "Duplicate" in top_doc["title"]
    
    # Ensure score of top payment doc is significantly higher than notification spec
    notification_doc = next((r for r in payment_results if "Notification" in r["title"]), None)
    if notification_doc:
        assert top_doc["score"] > notification_doc["score"]


# 7. POST /search endpoint test
def test_search_endpoint():
    payload = {
        "query": "How did we handle Redis failures?",
        "top_k": 3
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert len(data["results"]) == 3
    assert data["results"][0]["score"] > 0.3


# 8. POST /ask endpoint test
def test_ask_endpoint():
    payload = {
        "query": "How did we solve API latency?"
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == payload["query"]
    assert len(data["answer"]) > 20
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0


# 9. GET /documents endpoint test
def test_documents_endpoint():
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert data["total"] >= 10
    assert len(data["documents"]) == data["total"]
    
    # Verify schema of documents
    doc = data["documents"][0]
    assert "document_id" in doc
    assert "title" in doc
    assert "document_type" in doc
    assert "content" in doc


# 10. Fallback answer test
def test_fallback_answer():
    fake_chunks = [
        {
            "chunk_id": "test-01",
            "document_id": "ADR-999",
            "title": "Circuit Breakers",
            "document_type": "ADR",
            "team": "Platform",
            "service": "Gateway",
            "content": "Implemented circuit breaker pattern with resilience4j.",
            "score": 0.85
        }
    ]
    answer = generate_deterministic_fallback("Circuit breakers", fake_chunks)
    assert "ADR-999" in answer
    assert "Circuit Breakers" in answer
    assert "Supporting Sources" in answer


# 11. Source citations structure and validation
def test_source_citations_structure():
    res = answer_question("Why did we choose database sharding?")
    assert "sources" in res
    assert len(res["sources"]) > 0
    for s in res["sources"]:
        assert "document_id" in s
        assert "title" in s
        assert "document_type" in s
        assert "score" in s


# 12. POST /feedback endpoint test
def test_feedback_endpoint():
    payload = {
        "query": "Have we solved duplicate payment processing before?",
        "helpful": True,
        "comment": "Very accurate citation of ADR-001."
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
