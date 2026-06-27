"""SQLAlchemy model for chat history."""

import uuid
from datetime import datetime
from sqlalchemy import String, Text, Float, DateTime, JSON, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class ChatMessage(Base):
    """Represents a single Q&A interaction stored in chat history."""

    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list] = mapped_column(JSON, nullable=True)  # list of citation dicts
    retrieval_latency_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    response_latency_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    top_k: Mapped[int] = mapped_column(nullable=True)
    model_used: Mapped[str] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    def __repr__(self) -> str:
        return f"<ChatMessage id={self.id} session={self.session_id}>"
