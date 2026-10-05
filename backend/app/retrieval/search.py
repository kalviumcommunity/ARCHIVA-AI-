import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.retrieval.embeddings import generate_embedding, generate_embeddings

# Locate sample_chunks.json dynamically across repository layouts
def _find_data_path() -> Path:
    candidates = [
        Path(__file__).resolve().parents[2] / "data" / "sample_chunks.json",
        Path(__file__).resolve().parents[3] / "backend" / "data" / "sample_chunks.json",
        Path.cwd() / "data" / "sample_chunks.json",
        Path.cwd() / "backend" / "data" / "sample_chunks.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


DATA_PATH = _find_data_path()

_chunks_cache: List[Dict[str, Any]] = []
_chunk_embeddings_cache: Optional[np.ndarray] = None


def load_knowledge_base() -> List[Dict[str, Any]]:
    """Load chunks from data file and compute/cache their embeddings."""
    global _chunks_cache, _chunk_embeddings_cache
    if not _chunks_cache:
        data_file = _find_data_path()
        with open(data_file, "r", encoding="utf-8") as f:
            _chunks_cache = json.load(f)

        # Batch compute embeddings for all chunks once
        texts = [chunk.get("content", "") for chunk in _chunks_cache]
        embeddings = generate_embeddings(texts)
        _chunk_embeddings_cache = np.array(embeddings, dtype=np.float32)

    return _chunks_cache


def get_all_documents() -> List[Dict[str, Any]]:
    """Return all engineering chunks and documents in the repository."""
    if not _chunks_cache:
        load_knowledge_base()
    return _chunks_cache


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform semantic vector search using cosine similarity against stored chunks.
    
    Returns list of dicts sorted descending by similarity score with keys:
    chunk_id, document_id, title, document_type, content, team, service, score
    """
    if not _chunks_cache or _chunk_embeddings_cache is None:
        load_knowledge_base()

    query_embedding = np.array(generate_embedding(query), dtype=np.float32)

    # Compute dot product and norms across matrix
    dot_products = np.dot(_chunk_embeddings_cache, query_embedding)
    matrix_norms = np.linalg.norm(_chunk_embeddings_cache, axis=1)
    query_norm = np.linalg.norm(query_embedding)

    if query_norm == 0:
        scores = np.zeros(len(_chunks_cache))
    else:
        scores = dot_products / (matrix_norms * query_norm + 1e-10)

    results = []
    for idx, chunk in enumerate(_chunks_cache):
        results.append({
            "chunk_id": chunk.get("chunk_id", f"chk-{idx}"),
            "document_id": chunk.get("document_id", "DOC-UNKNOWN"),
            "title": chunk.get("title", "Untitled"),
            "document_type": chunk.get("document_type", "Engineering Note"),
            "content": chunk.get("content", ""),
            "team": chunk.get("team", "Engineering"),
            "service": chunk.get("service", "Core"),
            "score": round(float(scores[idx]), 4)
        })

    # Sort descending by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]