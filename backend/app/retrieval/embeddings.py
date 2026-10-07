import os
import math
import hashlib
import logging
from typing import List, Optional
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("archiva.embeddings")

# Google Gemini Embedding model specifications
# gemini-embedding-2 is Google's supported text/multimodal embedding model
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 768
FALLBACK_MODELS = ["gemini-embedding-2", "gemini-embedding-exp-03-07"]

_client = None
_mock_mode: bool = False

# Domain concept keywords for deterministic test mock embeddings
_CONCEPTS = {
    "payment": ["payment", "charge", "idempotency", "duplicate", "transaction"],
    "redis": ["redis", "caching", "cache", "sentinel", "cluster", "failover"],
    "database": ["database", "postgres", "postgresql", "sharding", "shard", "pgbouncer", "connection", "pool"],
    "latency": ["latency", "p99", "spike", "timeout", "slow", "unindexed"],
    "auth": ["auth", "jwt", "token", "rs256", "refresh"],
    "notification": ["notification", "rabbitmq", "sms", "email", "queue"],
    "retry": ["retry", "backoff", "exponential", "jitter"],
    "rate_limit": ["rate", "limiting", "bucket", "limit", "429"],
}


def set_mock_mode(enabled: bool) -> None:
    """Enable or disable mock embedding mode for offline unit tests and local verification."""
    global _mock_mode
    _mock_mode = enabled


def is_mock_mode() -> bool:
    """Check if mock embedding mode is active."""
    if _mock_mode:
        return True
    env_flag = os.getenv("ARCHIVA_EMBEDDING_FALLBACK", "").strip().lower()
    return env_flag in ("1", "true", "yes")


def get_model() -> str:
    """Return the active embedding model identifier."""
    return EMBEDDING_MODEL


def _get_gemini_client():
    """Lazily initialize and return the official Google GenAI client."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Please set the GEMINI_API_KEY environment variable "
            "to use semantic vector search."
        )

    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
        return _client
    except Exception as e:
        logger.error(f"Failed to initialize google.genai client: {e}")
        raise RuntimeError(f"Failed to initialize Gemini embedding client: {e}")


def _generate_mock_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """
    Generate a deterministic 768-dimensional unit vector for testing/offline use.
    Uses concept projection and token hashing so semantic ranking tests pass without network access.
    """
    vec = np.zeros(dim, dtype=np.float32)
    clean_text = text.lower()
    words = [w.strip(".,?!-()[]{}:;\"'") for w in clean_text.split() if len(w) > 1]

    # Concept projection across first 512 dimensions (64 dims per concept)
    for c_idx, (c_name, keywords) in enumerate(_CONCEPTS.items()):
        start_idx = c_idx * 64
        matched = sum(1 for kw in keywords if kw in clean_text)
        if matched > 0:
            vec[start_idx : start_idx + 64] += matched * 3.0

    # Token hashing across remaining dimensions
    for w in words:
        h = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16)
        idx = 512 + (h % (dim - 512))
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign

    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    else:
        vec = np.full(dim, 1.0 / math.sqrt(dim), dtype=np.float32)

    return vec.tolist()


def generate_embedding(text: str) -> List[float]:
    """
    Generate a 768-dimensional dense vector embedding for a single text string using Google Gemini (gemini-embedding-2).
    Falls back to deterministic mock embedding only when explicitly enabled for unit tests.
    """
    if not text.strip():
        raise ValueError("Cannot generate embedding for empty text.")

    if is_mock_mode():
        return _generate_mock_embedding(text)

    client = _get_gemini_client()
    from google.genai import types

    config = types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM)

    # Try supported models with fallback if 404 occurs
    last_err = None
    for model_name in FALLBACK_MODELS:
        try:
            logger.info(f"Calling Gemini embed_content with model={model_name}, dim={EMBEDDING_DIM}")
            response = client.models.embed_content(
                model=model_name,
                contents=text,
                config=config,
            )
            if response.embeddings and response.embeddings[0].values:
                emb_values = list(response.embeddings[0].values)
                logger.info(f"Successfully generated embedding vector of length {len(emb_values)} using {model_name}")
                return emb_values
        except Exception as e:
            last_err = e
            logger.warning(f"Embedding attempt with model {model_name} failed: {e}")
            if "404" not in str(e) and "NOT_FOUND" not in str(e):
                # Non-404 error (e.g. auth/quota), no need to try alternative model names
                break

    logger.error(f"All Gemini embed_content attempts failed. Last error: {last_err}", exc_info=True)
    raise RuntimeError(f"Gemini embedding API call failed: {last_err}")


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate 768-dimensional embeddings for a batch of text strings efficiently using Google Gemini (gemini-embedding-2).
    Wraps each text in a types.Content object as required by gemini-embedding-2 for batching.
    """
    if not texts:
        return []

    if is_mock_mode():
        return [_generate_mock_embedding(t) for t in texts]

    client = _get_gemini_client()
    from google.genai import types

    config = types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM)
    # In gemini-embedding-2, pass each text wrapped in types.Content to generate per-document embeddings
    contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]

    last_err = None
    for model_name in FALLBACK_MODELS:
        try:
            logger.info(f"Calling Gemini batch embed_content for {len(texts)} chunks with model={model_name}")
            response = client.models.embed_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            if response.embeddings and len(response.embeddings) == len(texts):
                embeddings_list = [list(emb.values) for emb in response.embeddings]
                logger.info(f"Successfully generated batch embeddings for {len(embeddings_list)} documents using {model_name}")
                return embeddings_list
        except Exception as e:
            last_err = e
            logger.warning(f"Batch embedding attempt with model {model_name} failed: {e}")
            if "404" not in str(e) and "NOT_FOUND" not in str(e):
                break

    logger.error(f"All Gemini batch embed_content attempts failed. Last error: {last_err}", exc_info=True)
    raise RuntimeError(f"Gemini batch embedding API call failed: {last_err}")