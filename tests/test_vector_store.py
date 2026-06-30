"""Unit tests for the FAISS vector store."""

import pytest
import numpy as np
from pathlib import Path

from app.services.ingestion.vector_store import FAISSVectorStore
from app.services.ingestion.chunker import DocumentChunk


DIM = 8


def make_chunks(n: int):
    return [
        DocumentChunk(chunk_id=i, text=f"Chunk {i}", source="test.txt", page_number=1, char_start=i * 10)
        for i in range(n)
    ]


def make_embeddings(n: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.random((n, dim)).astype(np.float32)
    # Normalise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def store(tmp_path):
    s = FAISSVectorStore(index_path=str(tmp_path / "faiss"), dimension=DIM)
    s.create_index()
    return s


def test_create_index(store):
    # After create_index(), the FAISS object exists but has no vectors yet.
    # is_loaded() requires ntotal > 0 — an empty index is not search-ready.
    assert store._index is not None
    assert store._index.ntotal == 0
    assert not store.is_loaded()


def test_add_chunks(store):
    chunks = make_chunks(5)
    embeddings = make_embeddings(5)
    store.add_chunks(chunks, embeddings)
    assert store._index.ntotal == 5


def test_search_returns_results(store):
    chunks = make_chunks(5)
    embeddings = make_embeddings(5)
    store.add_chunks(chunks, embeddings)

    query = make_embeddings(1)
    results = store.search(query, top_k=3)
    assert len(results) == 3
    for chunk, score in results:
        assert isinstance(chunk, DocumentChunk)
        assert isinstance(score, float)


def test_save_and_load(store, tmp_path):
    chunks = make_chunks(3)
    embeddings = make_embeddings(3)
    store.add_chunks(chunks, embeddings)
    store.save()

    # Load into a new instance
    store2 = FAISSVectorStore(index_path=str(tmp_path / "faiss"), dimension=DIM)
    store2.load()
    assert store2._index.ntotal == 3
    assert len(store2._chunks) == 3


def test_search_on_unloaded_index():
    store = FAISSVectorStore(index_path="/tmp/nonexistent_xyz", dimension=DIM)
    query = make_embeddings(1)
    with pytest.raises(RuntimeError):
        store.search(query)
