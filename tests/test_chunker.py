"""Unit tests for document chunking."""

import pytest
from app.services.ingestion.chunker import DocumentChunker, DocumentChunk
from app.services.ingestion.extractor import ExtractionResult, PageContent


@pytest.fixture
def extraction_result():
    return ExtractionResult(
        filename="test.txt",
        file_type="txt",
        total_pages=1,
        pages=[
            PageContent(
                page_number=1,
                text=(
                    "Artificial intelligence is the simulation of human intelligence. "
                    "Machine learning is a subset of AI. Deep learning uses neural networks. "
                    "Natural language processing enables computers to understand text. "
                    "Computer vision allows machines to interpret images. "
                    "Reinforcement learning trains agents through rewards. "
                    "Transfer learning adapts pre-trained models to new tasks. "
                    "Generative AI creates new content from learned patterns."
                ),
                source="test.txt",
            )
        ],
    )


def test_chunk_count(extraction_result):
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(extraction_result)
    assert len(chunks) >= 1


def test_chunk_type(extraction_result):
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(extraction_result)
    for c in chunks:
        assert isinstance(c, DocumentChunk)


def test_chunk_text_not_empty(extraction_result):
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(extraction_result)
    for c in chunks:
        assert c.text.strip() != ""


def test_chunk_source_preserved(extraction_result):
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(extraction_result)
    for c in chunks:
        assert c.source == "test.txt"


def test_chunk_page_number_preserved(extraction_result):
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(extraction_result)
    for c in chunks:
        assert c.page_number == 1


def test_chunk_ids_are_sequential(extraction_result):
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(extraction_result)
    for i, c in enumerate(chunks):
        assert c.chunk_id == i


def test_empty_document():
    empty = ExtractionResult(
        filename="empty.txt",
        file_type="txt",
        total_pages=0,
        pages=[],
    )
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(empty)
    assert chunks == []
