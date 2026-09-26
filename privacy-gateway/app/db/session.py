"""Async SQLAlchemy engine and session factory."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings


def build_engine(database_url: str) -> AsyncEngine:
    """Create an engine whose logs and errors never include bound parameter values."""
    # hide_parameters keeps bound values (token hashes, ciphertext) out of SQLAlchemy
    # log lines and exception messages.
    return create_async_engine(database_url, pool_pre_ping=True, hide_parameters=True)


settings = get_settings()
engine = build_engine(settings.database_url)
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
