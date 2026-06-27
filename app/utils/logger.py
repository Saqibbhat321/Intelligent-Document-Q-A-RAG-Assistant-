"""Structured logging configuration."""

import logging
import sys


def configure_logging(debug: bool = False) -> None:
    """Set up root logger with a consistent format."""
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Quiet noisy third-party libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
