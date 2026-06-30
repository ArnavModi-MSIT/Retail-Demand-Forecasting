"""
Database connection layer.

Reads connection settings from environment variables so the exact same code
works against local Docker Postgres and AWS RDS — only the .env changes.

Required environment variables (see .env.example):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""
import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

load_dotenv()  # populates os.environ from .env if present; no-op if missing


def _build_connection_url() -> str:
    host     = os.environ.get("DB_HOST", "localhost")
    port     = os.environ.get("DB_PORT", "5432")
    name     = os.environ.get("DB_NAME", "retail_forecast")
    user     = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "postgres")
    sslmode  = os.environ.get("DB_SSLMODE")  # set to "require" for RDS

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    if sslmode:
        url += f"?sslmode={sslmode}"
    return url


_engine: Engine | None = None


def get_engine() -> Engine:
    """Lazily create and cache a single SQLAlchemy engine for the process."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            _build_connection_url(),
            pool_size=5,          # micro instances have low connection limits
            max_overflow=2,
            pool_pre_ping=True,   # avoids stale-connection errors after idle
        )
    return _engine


_SessionLocal = None


def _get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal


@contextmanager
def get_session():
    """
    Context manager yielding a SQLAlchemy session with automatic
    commit/rollback/close.

    Usage:
        with get_session() as session:
            session.execute(...)
    """
    session = _get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """Quick health check — returns True if the database is reachable."""
    try:
        with get_engine().connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True
    except Exception:
        return False