"""Embedding generation using sentence-transformers."""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings
from app.services.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    """Generates dense vector embeddings for document chunks."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def embed_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Return a (N, D) float32 numpy array of embeddings for the chunks."""
        if not chunks:
            return np.empty((0, self.dimension), dtype=np.float32)

        texts = [c.text for c in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks …")
        embeddings = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Return a (1, D) float32 array for a single query string."""
        embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)
