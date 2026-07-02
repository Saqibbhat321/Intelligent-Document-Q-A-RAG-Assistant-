import logging
import time
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.models.document import Document
from app.services.ingestion.chunker import DocumentChunker
from app.services.ingestion.embedder import EmbeddingService
from app.services.ingestion.extractor import TextExtractor
from app.services.ingestion.vector_store import FAISSVectorStore
from app.utils.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)
settings = get_settings()


class IngestionPipeline:
    """
    Orchestrates: extract → chunk → embed → index → persist metadata.
    Maintains a single in-process FAISS store that is shared across requests.
    """

    def __init__(self) -> None:
        self.extractor = TextExtractor()
        self.chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.embedder = EmbeddingService()
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self._tracker = MLflowTracker()
        self._load_existing_index()

    # Public API
    def ingest(
        self,
        file_path: str | Path,
        original_filename: str,
        db: Session,
    ) -> Document:
    
        path = Path(file_path)
        start_total = time.perf_counter()

        # Create DB record in 'processing' state
        doc_record = Document(
            filename=path.name,
            original_filename=original_filename,
            file_type=path.suffix.lower().lstrip("."),
            file_size_bytes=path.stat().st_size,
            status="processing",
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        db.add(doc_record)
        db.commit()
        db.refresh(doc_record)

        try:
            # 1. Extract
            extraction = self.extractor.extract(path, original_filename)

            # 2. Chunk
            chunks = self.chunker.chunk_document(extraction)

            if not chunks:
                raise ValueError("No text could be extracted from the document.")

            # 3. Embed
            embeddings = self.embedder.embed_chunks(chunks)

            # 4. Index
            self.vector_store.add_chunks(chunks, embeddings)
            self.vector_store.save()

            # 5. Update DB record
            elapsed = time.perf_counter() - start_total
            doc_record.status = "ready"
            doc_record.total_chunks = len(chunks)
            doc_record.total_pages = extraction.total_pages
            doc_record.ingestion_latency_seconds = elapsed
            db.commit()
            db.refresh(doc_record)

            # 6. Track with MLflow
            self._tracker.log_ingestion(
                document_id=str(doc_record.id),
                filename=original_filename,
                embedding_model=settings.embedding_model,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                total_chunks=len(chunks),
                ingestion_latency=elapsed,
            )

            logger.info(
                f"Ingestion complete: '{original_filename}' — "
                f"{len(chunks)} chunks in {elapsed:.2f}s"
            )
            return doc_record

        except Exception as exc:
            logger.exception(f"Ingestion failed for '{original_filename}': {exc}")
            doc_record.status = "error"
            doc_record.error_message = str(exc)
            db.commit()
            raise

    def _load_existing_index(self) -> None:
        """Load persisted FAISS index on startup (if it exists)."""
        try:
            self.vector_store.load()
        except FileNotFoundError:
            logger.info("No existing FAISS index found — will create on first upload.")
