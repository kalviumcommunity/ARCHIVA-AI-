from typing import List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="The natural-language search query from the engineer")
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of relevant chunks to retrieve")


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    document_type: str
    content: str
    team: Optional[str] = "Engineering"
    service: Optional[str] = "Core"
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]


class AskRequest(BaseModel):
    query: str = Field(..., description="The engineering question to answer using knowledge retrieval")


class SourceItem(BaseModel):
    document_id: str
    title: str
    document_type: str
    score: float
    team: Optional[str] = None
    service: Optional[str] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]


class FeedbackRequest(BaseModel):
    query: str
    helpful: bool
    comment: Optional[str] = ""


class FeedbackResponse(BaseModel):
    status: str
    message: str
