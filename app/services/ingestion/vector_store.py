"""FAISS vector store — create, save, load, and search."""

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from app.config.settings import get_settings
from app.services.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)
settings = get_settings()

# File paths within the index directory
_INDEX_FILE = "index.faiss"
_META_FILE = "metadata.json"


class FAISSVectorStore:
    """Flat L2 / cosine FAISS index with JSON-backed chunk metadata."""

    def __init__(self, index_path: str | None = None, dimension: int | None = None) -> None:
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.dimension = dimension or settings.embedding_dimension
        self._index: faiss.IndexFlatIP | None = None  # inner-product ≡ cosine on normalised vecs
        self._chunks: List[DocumentChunk] = []

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------
    def create_index(self) -> None:
        """Initialise an empty FAISS IndexFlatIP (inner product for cosine similarity)."""
        self._index = faiss.IndexFlatIP(self.dimension)
        self._chunks = []
        logger.info(f"FAISS index created (dim={self.dimension})")

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the index."""
        if self._index is None:
            self.create_index()

        if embeddings.shape[0] == 0:
            logger.warning("No embeddings to add — skipping.")
            return

        self._index.add(embeddings)  # type: ignore[union-attr]
        self._chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} vectors. Total vectors: {self._index.ntotal}")

    def save(self) -> None:
        """Persist index and metadata to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path / _INDEX_FILE))

        metadata = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source": c.source,
                "page_number": c.page_number,
                "char_start": c.char_start,
            }
            for c in self._chunks
        ]
        (self.index_path / _META_FILE).write_text(json.dumps(metadata, ensure_ascii=False))
        logger.info(f"FAISS index saved to {self.index_path}")

    def load(self) -> None:
        """Load index and metadata from disk."""
        index_file = self.index_path / _INDEX_FILE
        meta_file = self.index_path / _META_FILE

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_file}")

        self._index = faiss.read_index(str(index_file))
        raw = json.loads(meta_file.read_text())
        self._chunks = [DocumentChunk(**r) for r in raw]
        logger.info(
            f"FAISS index loaded from {self.index_path} — {self._index.ntotal} vectors"
        )

    def is_loaded(self) -> bool:
        return self._index is not None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """Return top-k (chunk, score) pairs by cosine similarity."""
        if self._index is None:
            raise RuntimeError("Index not loaded. Call load() or add_chunks() first.")
        if self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, k)

        results: List[Tuple[DocumentChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and 0 <= idx < len(self._chunks):
                results.append((self._chunks[idx], float(score)))

        return results
