from typing import List, Union
from sentence_transformers import SentenceTransformer

# Singleton embedding model loaded once at startup / import
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazily load and return the shared SentenceTransformer model instance."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def generate_embedding(text: str) -> List[float]:
    """Generate a 384-dimensional dense vector embedding for a single text string."""
    model = get_model()
    embedding = model.encode(text)
    return embedding.tolist()


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of text strings efficiently."""
    if not texts:
        return []
    model = get_model()
    embeddings = model.encode(texts)
    return embeddings.tolist()