"""MLflow experiment tracking for ingestion and query runs."""

import logging
import os

from app.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _mlflow_available() -> bool:
    """Quick TCP check — returns False immediately if MLflow is unreachable."""
    import socket
    from urllib.parse import urlparse

    url = urlparse(settings.mlflow_tracking_uri)
    host = url.hostname or "localhost"
    port = url.port or 5001
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


class MLflowTracker:
    """Lightweight wrapper that logs metrics and params to MLflow."""

    def __init__(self) -> None:
        self._enabled = False
        if _mlflow_available():
            try:
                import mlflow
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                mlflow.set_experiment(settings.mlflow_experiment_name)
                self._enabled = True
                logger.info("MLflow tracking enabled.")
            except Exception as exc:
                logger.warning(f"MLflow setup failed (tracking disabled): {exc}")
        else:
            logger.warning("MLflow unreachable — tracking disabled for this session.")

    # ------------------------------------------------------------------
    # Ingestion run
    # ------------------------------------------------------------------
    def log_ingestion(
        self,
        document_id: str,
        filename: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        total_chunks: int,
        ingestion_latency: float,
    ) -> None:
        if not self._enabled:
            return
        try:
            import mlflow
            with mlflow.start_run(run_name=f"ingest-{filename}"):
                mlflow.set_tag("run_type", "ingestion")
                mlflow.set_tag("document_id", document_id)
                mlflow.set_tag("filename", filename)
                mlflow.log_param("embedding_model", embedding_model)
                mlflow.log_param("chunk_size", chunk_size)
                mlflow.log_param("chunk_overlap", chunk_overlap)
                mlflow.log_metric("total_chunks", total_chunks)
                mlflow.log_metric("ingestion_latency_seconds", ingestion_latency)
        except Exception as exc:
            logger.warning(f"MLflow ingestion logging failed: {exc}")

    # ------------------------------------------------------------------
    # Query run
    # ------------------------------------------------------------------
    def log_query(
        self,
        session_id: str,
        retrieval_latency: float,
        response_latency: float,
        top_k: int,
        model: str,
        num_sources: int,
    ) -> None:
        if not self._enabled:
            return
        try:
            import mlflow
            with mlflow.start_run(run_name=f"query-{session_id[:8]}"):
                mlflow.set_tag("run_type", "query")
                mlflow.set_tag("session_id", session_id)
                mlflow.log_param("llm_model", model)
                mlflow.log_param("top_k", top_k)
                mlflow.log_metric("retrieval_latency_seconds", retrieval_latency)
                mlflow.log_metric("response_latency_seconds", response_latency)
                mlflow.log_metric(
                    "total_latency_seconds", retrieval_latency + response_latency
                )
                mlflow.log_metric("num_sources_cited", num_sources)
        except Exception as exc:
            logger.warning(f"MLflow query logging failed: {exc}")
