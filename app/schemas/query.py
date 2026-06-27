"""Pydantic schemas for query-related API requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Body for POST /query."""

    question: str = Field(..., min_length=3, max_length=2000, description="User question")
    session_id: str = Field(default="default", description="Conversation session identifier")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of chunks to retrieve")


class CitationSchema(BaseModel):
    """A single source citation."""

    document_name: str
    page_number: int
    relevance_score: float


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str
    citations: List[CitationSchema]
    retrieval_latency_seconds: float
    response_latency_seconds: float
    model_used: str
    session_id: str
