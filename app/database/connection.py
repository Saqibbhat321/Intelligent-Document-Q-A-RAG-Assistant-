"""Database connection and session management."""

import logging
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.pool import NullPool

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""


def get_engine():
    """Create and return the SQLAlchemy engine."""
    is_sqlite = settings.database_url.startswith("sqlite")
    kwargs: dict = {"pool_pre_ping": True, "echo": settings.debug}
    if not is_sqlite:
        kwargs.update({"pool_size": 10, "max_overflow": 20})
    else:
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(settings.database_url, **kwargs)

    @event.listens_for(engine, "connect")
    def set_search_path(dbapi_conn, connection_record):
        # PostgreSQL-only — skip for SQLite (used in tests)
        if settings.database_url.startswith("postgresql"):
            cursor = dbapi_conn.cursor()
            cursor.execute("SET search_path TO public")
            cursor.close()

    return engine


engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """FastAPI dependency — yields a database session."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_tables() -> None:
    """Create all tables in the database."""
    from app.models import document, chat  # noqa: F401 — register models
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")
