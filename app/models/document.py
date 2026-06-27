"""SQLAlchemy model for uploaded documents."""

import uuid
from datetime import datetime
from sqlalchemy import String, Integer, DateTime, Text, Float, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class Document(Base):
    """Represents an uploaded document and its processing metadata."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)  # pdf, docx, txt
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_pages: Mapped[int] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="processing"
    )  # processing | ready | error
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(256), nullable=True)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=True)
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=True)
    ingestion_latency_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.original_filename} status={self.status}>"
