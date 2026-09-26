"""Health endpoint for the API and its required backing services."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal

from fastapi import APIRouter, Request, Response
from redis.asyncio import Redis

from app.db.health import check_database, check_redis
from app.models.health import HealthResponse, ServiceStatus

router = APIRouter(tags=["health"])


async def _safe_check(
    check: Callable[..., Awaitable[None]],
    *args: object,
) -> ServiceStatus:
    """Convert a dependency probe failure to a non-sensitive status."""
    try:
        await check(*args)
    except Exception:
        return "unavailable"
    return "available"


@router.get("/health", response_model=HealthResponse, summary="Check service health")
async def get_health(request: Request, response: Response) -> HealthResponse:
    """Report whether PostgreSQL and Redis are reachable without exposing details."""
    redis_client = getattr(request.app.state, "redis", None)
    database_status, redis_status = await _gather_status(redis_client)
    services: dict[str, ServiceStatus] = {
        "postgres": database_status,
        "redis": redis_status,
    }
    status: Literal["ok", "degraded"] = (
        "ok"
        if all(service_status == "available" for service_status in services.values())
        else "degraded"
    )
    if status == "degraded":
        response.status_code = 503
    return HealthResponse(status=status, services=services)


async def _gather_status(redis_client: Redis | None) -> tuple[ServiceStatus, ServiceStatus]:
    """Run independent dependency checks concurrently."""
    database_status, redis_status = await asyncio.gather(
        _safe_check(check_database),
        _safe_check(check_redis, redis_client),
    )
    return database_status, redis_status
