"""FAISS vector store — create, save, load, and search."""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from app.config.settings import get_settings
from app.services.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)
settings = get_settings()

_INDEX_FILE = "index.faiss"
_META_FILE = "metadata.json"


class FAISSVectorStore:
    """Flat inner-product FAISS index with JSON-backed chunk metadata."""

    def __init__(self, index_path: str | None = None, dimension: int | None = None) -> None:
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.dimension = dimension or settings.embedding_dimension
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: List[DocumentChunk] = []

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------
    def create_index(self) -> None:
        """Initialise an empty FAISS IndexFlatIP."""
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

        self._index.add(embeddings)
        self._chunks.extend(chunks)
        logger.info(
            f"Added {len(chunks)} vectors. Total in index: {self._index.ntotal} | "
            f"Total chunks tracked: {len(self._chunks)}"
        )

    def save(self) -> None:
        """Persist index and metadata to disk atomically."""
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Verify consistency before saving
        if self._index is not None and self._index.ntotal != len(self._chunks):
            logger.error(
                f"Index/metadata mismatch before save: "
                f"{self._index.ntotal} vectors vs {len(self._chunks)} chunks — aborting save."
            )
            return

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
        logger.info(
            f"FAISS index saved: {self._index.ntotal} vectors, "
            f"{len(self._chunks)} chunks → {self.index_path}"
        )

    def load(self) -> None:
        """Load index and metadata from disk, validating consistency."""
        index_file = self.index_path / _INDEX_FILE
        meta_file = self.index_path / _META_FILE

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_file}")

        loaded_index = faiss.read_index(str(index_file))
        raw = json.loads(meta_file.read_text())
        chunks = [DocumentChunk(**r) for r in raw]

        # Validate consistency — if mismatched, reset to avoid silent corruption
        if loaded_index.ntotal != len(chunks):
            logger.warning(
                f"FAISS index/metadata mismatch on load: "
                f"{loaded_index.ntotal} vectors vs {len(chunks)} chunks. "
                f"Resetting index — please re-upload documents."
            )
            self.create_index()
            return

        self._index = loaded_index
        self._chunks = chunks
        logger.info(
            f"FAISS index loaded: {self._index.ntotal} vectors, "
            f"{len(self._chunks)} chunks ← {self.index_path}"
        )

    def reset(self) -> None:
        """Wipe the in-memory index and delete files from disk."""
        self.create_index()
        index_file = self.index_path / _INDEX_FILE
        meta_file = self.index_path / _META_FILE
        index_file.unlink(missing_ok=True)
        meta_file.unlink(missing_ok=True)
        logger.info("FAISS index reset and disk files removed.")

    def is_loaded(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

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

        logger.debug(f"Search returned {len(results)} results for top_k={top_k}")
        return results
