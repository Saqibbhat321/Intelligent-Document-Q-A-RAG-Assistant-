"""Pydantic schemas for document-related API requests and responses."""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Returned after a successful document upload or list query."""

    id: uuid.UUID
    original_filename: str
    file_type: str
    file_size_bytes: int
    total_chunks: int
    total_pages: Optional[int] = None
    status: str
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    ingestion_latency_seconds: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class UploadResponse(BaseModel):
    """Response body for POST /upload."""

    message: str
    document: DocumentResponse
