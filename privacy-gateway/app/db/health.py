"""Non-sensitive connectivity checks for required dependencies."""

from redis.asyncio import Redis
from sqlalchemy import text

from app.db.session import engine


async def check_database() -> None:
    """Raise if PostgreSQL cannot execute a trivial connectivity query."""
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


async def check_redis(client: Redis | None) -> None:
    """Raise if Redis is not initialized or does not respond to PING."""
    if client is None:
        raise RuntimeError("Redis client is not initialized")
    await client.ping()
