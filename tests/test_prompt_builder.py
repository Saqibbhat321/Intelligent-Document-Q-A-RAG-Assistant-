"""Unit tests for the prompt builder."""

import pytest
from app.services.generation.prompt_builder import build_prompt, SYSTEM_PROMPT
from app.services.retrieval.retriever import RetrievedChunk
from app.services.ingestion.chunker import DocumentChunk


def make_retrieved_chunk(text: str, source: str = "doc.txt", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=0, text=text, source=source, page_number=page, char_start=0
        ),
        score=0.9,
    )


def test_system_prompt_returned():
    system, _ = build_prompt("What is AI?", [])
    assert system == SYSTEM_PROMPT


def test_no_chunks_returns_no_context_message():
    _, user = build_prompt("What is AI?", [])
    assert "No relevant context" in user


def test_chunks_included_in_prompt():
    chunks = [make_retrieved_chunk("AI is the simulation of human intelligence.")]
    _, user = build_prompt("What is AI?", chunks)
    assert "AI is the simulation of human intelligence." in user


def test_source_citation_in_prompt():
    chunks = [make_retrieved_chunk("Some text.", source="report.pdf", page=3)]
    _, user = build_prompt("Summarise.", chunks)
    assert "report.pdf" in user
    assert "Page 3" in user


def test_multiple_chunks():
    chunks = [
        make_retrieved_chunk("Chunk one text.", source="a.txt", page=1),
        make_retrieved_chunk("Chunk two text.", source="b.pdf", page=2),
    ]
    _, user = build_prompt("What do you know?", chunks)
    assert "Source 1" in user
    assert "Source 2" in user
    assert "a.txt" in user
    assert "b.pdf" in user


def test_question_appears_in_prompt():
    _, user = build_prompt("What is deep learning?", [])
    assert "What is deep learning?" in user
