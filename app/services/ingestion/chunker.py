"""Recursive character text splitter for document chunking."""

import logging
from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.ingestion.extractor import ExtractionResult, PageContent

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single chunk of text with provenance metadata."""

    chunk_id: int
    text: str
    source: str          # original filename
    page_number: int     # page / section number within the document
    char_start: int      # character offset within the page text


class DocumentChunker:
    """Splits extracted document pages into overlapping text chunks."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(self, extraction: ExtractionResult) -> List[DocumentChunk]:
        """Return all chunks for a fully extracted document."""
        chunks: List[DocumentChunk] = []
        chunk_id = 0

        for page in extraction.pages:
            page_chunks = self._split_page(page, chunk_id)
            chunks.extend(page_chunks)
            chunk_id += len(page_chunks)

        logger.info(
            f"Chunked '{extraction.filename}' → {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def _split_page(self, page: PageContent, start_id: int) -> List[DocumentChunk]:
        """Split a single page and annotate with provenance."""
        raw_splits = self._splitter.split_text(page.text)
        result: List[DocumentChunk] = []
        char_cursor = 0

        for i, split in enumerate(raw_splits):
            # Locate where this split begins in the page text
            pos = page.text.find(split, char_cursor)
            char_start = pos if pos != -1 else char_cursor

            result.append(
                DocumentChunk(
                    chunk_id=start_id + i,
                    text=split,
                    source=page.source,
                    page_number=page.page_number,
                    char_start=char_start,
                )
            )
            if pos != -1:
                char_cursor = pos + len(split)

        return result
