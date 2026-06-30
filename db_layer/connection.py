"""
Database connection layer.

Reads connection settings from environment variables so the exact same code
works against local Docker Postgres, AWS RDS, and Neon — only the source of
config changes between environments:

    - Local dev:           .env file (via python-dotenv) → os.environ
    - Streamlit Cloud:      st.secrets (TOML, set in the dashboard)
    - AWS / general prod:   real environment variables (e.g. from EC2's
                             environment or a docker-compose .env file)

Required keys (same names across all three sources):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SSLMODE (optional)
"""
import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

load_dotenv()  # populates os.environ from .env if present; no-op if missing


def _get_setting(key: str, default: str | None = None) -> str | None:
    """
    Look up a config value, preferring real environment variables /
    .env (already loaded into os.environ above), and falling back to
    Streamlit's st.secrets if running on Streamlit Community Cloud.

    os.environ is checked first so local dev and Docker/AWS deployments
    (which set real env vars) are unaffected by this change — the
    st.secrets path only activates when nothing was found in the
    environment, which is exactly the Streamlit Cloud scenario.
    """
    value = os.environ.get(key)
    if value:
        return value

    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        # Not running inside Streamlit, or st.secrets not configured —
        # fall through to the default.
        pass

    return default


def _build_connection_url() -> str:
    host     = _get_setting("DB_HOST", "localhost")
    port     = _get_setting("DB_PORT", "5432")
    name     = _get_setting("DB_NAME", "retail_forecast")
    user     = _get_setting("DB_USER", "postgres")
    password = _get_setting("DB_PASSWORD", "postgres")
    sslmode  = _get_setting("DB_SSLMODE")  # set to "require" for RDS / Neon

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