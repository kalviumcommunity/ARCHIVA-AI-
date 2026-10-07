import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.models.schemas import (
    SearchRequest,
    SearchResponse,
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from app.retrieval.search import search, get_all_documents, load_knowledge_base
from app.rag.pipeline import answer_question

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("archiva.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler: load knowledge base documents into memory cache."""
    logger.info("Initializing Archiva AI knowledge base...")
    load_knowledge_base()
    logger.info("Archiva AI knowledge base loaded successfully.")
    yield


app = FastAPI(
    title="Archiva AI API",
    description="Engineering Knowledge Intelligence & Contextual Retrieval Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local web development & demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory feedback storage
_feedback_records: List[Dict[str, Any]] = []
FEEDBACK_FILE = Path(__file__).resolve().parents[1] / "data" / "feedback.json"


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "Archiva AI", "version": "1.0.0"}


@app.post("/search", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    """Semantic vector search across engineering knowledge documents."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    try:
        results = search(query=request.query, top_k=request.top_k)
        return SearchResponse(query=request.query, results=results)
    except RuntimeError as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Answer engineering questions using grounded RAG pipeline with citations."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response_data = answer_question(query=request.query)
        return AskResponse(
            question=response_data["question"],
            answer=response_data["answer"],
            sources=response_data["sources"],
        )
    except RuntimeError as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/documents")
def get_documents() -> Dict[str, Any]:
    """List all available engineering documents and chunks in the knowledge base."""
    docs = get_all_documents()
    return {
        "total": len(docs),
        "documents": docs
    }


@app.post("/feedback", response_model=FeedbackResponse)
def record_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record user feedback on answers for evaluation."""
    record = {
        "query": request.query,
        "helpful": request.helpful,
        "comment": request.comment or "",
    }
    _feedback_records.append(record)

    # Persist feedback to file safely
    try:
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(_feedback_records, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to persist feedback to disk: {e}")

    return FeedbackResponse(
        status="success",
        message="Feedback recorded successfully. Thank you!"
    )


# Serve the Vanilla JS frontend from the same FastAPI service in production.
# This keeps the frontend and API on the same origin and avoids deployment-time URL changes.
frontend_path = Path(__file__).resolve().parents[2] / "frontend"

if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
