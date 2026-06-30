"""Answer generator — ties retrieval, prompt building, and LLM together."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.models.chat import ChatMessage
from app.services.generation.llm_client import LLMClient
from app.services.generation.prompt_builder import build_prompt
from app.services.retrieval.retriever import RetrievedChunk, RetrieverService
from app.utils.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)
settings = get_settings()

# Number of previous turns to include as conversation memory
MEMORY_TURNS = 4


@dataclass
class Citation:
    """Source attribution included with each answer."""

    document_name: str
    page_number: int
    relevance_score: float


@dataclass
class GenerationResult:
    """Full result returned to the API caller."""

    answer: str
    citations: List[Citation] = field(default_factory=list)
    retrieval_latency_seconds: float = 0.0
    response_latency_seconds: float = 0.0
    model_used: str = ""


class GeneratorService:
    """
    Orchestrates: retrieve → fetch history → build prompt → call LLM → return answer + citations.
    """

    def __init__(self, retriever: RetrieverService, llm_client: LLMClient) -> None:
        self.retriever = retriever
        self.llm = llm_client
        self._tracker = MLflowTracker()

    def answer(
        self,
        question: str,
        session_id: str,
        db: Session,
        top_k: int | None = None,
    ) -> GenerationResult:
        """
        Full RAG pipeline for one user question.
        Fetches conversation history, retrieves context, generates grounded answer.
        Persists the interaction to chat_messages and logs metrics to MLflow.
        """
        # 1. Fetch conversation history for this session
        history = self._fetch_history(session_id, db)

        # 2. Retrieve relevant chunks
        retrieved, retrieval_latency = self.retriever.retrieve(question, top_k=top_k)

        # 3. Build prompt with context + history
        system_msg, user_msg = build_prompt(question, retrieved, conversation_history=history)

        # 4. Generate
        answer_text, response_latency = self.llm.generate(system_msg, user_msg)

        # 5. Build citations (deduplicated by source + page)
        seen = set()
        citations: List[Citation] = []
        for rc in retrieved:
            key = (rc.source, rc.page_number)
            if key not in seen:
                seen.add(key)
                citations.append(
                    Citation(
                        document_name=rc.source,
                        page_number=rc.page_number,
                        relevance_score=round(rc.score, 4),
                    )
                )

        result = GenerationResult(
            answer=answer_text,
            citations=citations,
            retrieval_latency_seconds=retrieval_latency,
            response_latency_seconds=response_latency,
            model_used=self.llm.model,
        )

        # 6. Persist to DB
        self._save_to_db(question, result, db, session_id, top_k)

        # 7. Track with MLflow
        self._tracker.log_query(
            session_id=session_id,
            retrieval_latency=retrieval_latency,
            response_latency=response_latency,
            top_k=top_k or settings.retrieval_top_k,
            model=self.llm.model,
            num_sources=len(citations),
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fetch_history(self, session_id: str, db: Session) -> List[dict]:
        """Return the last MEMORY_TURNS Q&A pairs for this session (oldest first)."""
        rows = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(MEMORY_TURNS)
            .all()
        )
        # Reverse so oldest turn comes first in the prompt
        return [{"question": r.question, "answer": r.answer} for r in reversed(rows)]

    def _save_to_db(
        self,
        question: str,
        result: GenerationResult,
        db: Session,
        session_id: str,
        top_k: int | None,
    ) -> None:
        sources = [
            {
                "document_name": c.document_name,
                "page_number": c.page_number,
                "relevance_score": c.relevance_score,
            }
            for c in result.citations
        ]
        record = ChatMessage(
            session_id=session_id,
            question=question,
            answer=result.answer,
            sources=sources,
            retrieval_latency_seconds=result.retrieval_latency_seconds,
            response_latency_seconds=result.response_latency_seconds,
            top_k=top_k or settings.retrieval_top_k,
            model_used=result.model_used,
        )
        db.add(record)
        db.commit()
        logger.info(f"Chat message saved — session={session_id}")
