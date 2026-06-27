"""Semantic retrieval — query the FAISS index and return ranked chunks."""

import logging
import time
from dataclasses import dataclass
from typing import List

from app.config.settings import get_settings
from app.services.ingestion.chunker import DocumentChunk
from app.services.ingestion.embedder import EmbeddingService
from app.services.ingestion.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    """A chunk returned by similarity search, with its relevance score."""

    chunk: DocumentChunk
    score: float

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def page_number(self) -> int:
        return self.chunk.page_number

    @property
    def text(self) -> str:
        return self.chunk.text


class RetrieverService:
    """
    Encapsulates query embedding + FAISS search.
    Shares the EmbeddingService and FAISSVectorStore with the ingestion pipeline.
    """

    def __init__(
        self,
        embedder: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[List[RetrievedChunk], float]:
        """
        Embed the query and return top-k chunks plus retrieval latency in seconds.
        """
        k = top_k or settings.retrieval_top_k

        if not self.vector_store.is_loaded():
            raise RuntimeError(
                "Vector store is not loaded. Upload at least one document first."
            )

        t0 = time.perf_counter()
        query_embedding = self.embedder.embed_query(query)
        raw_results = self.vector_store.search(query_embedding, top_k=k)
        latency = time.perf_counter() - t0

        results = [RetrievedChunk(chunk=chunk, score=score) for chunk, score in raw_results]
        logger.info(
            f"Retrieved {len(results)} chunks for query in {latency:.3f}s "
            f"(top_k={k})"
        )
        return results, latency
