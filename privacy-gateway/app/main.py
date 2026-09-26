"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from secrets import token_hex
from time import perf_counter

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from app.api.routes import chat_completions, health
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import engine

settings = get_settings()
configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Create and close shared async service clients."""
    redis_client: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
    application.state.redis = redis_client
    try:
        yield
    finally:
        await redis_client.aclose()
        await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="A starter API for a privacy-preserving LLM gateway.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error(
    _request: Request,
    _error: RequestValidationError,
) -> JSONResponse:
    """Return an OpenAI-style error without echoing user input or validation values."""
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": "Invalid request body.",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_request",
            }
        },
    )


@app.middleware("http")
async def log_request_metadata(
    request: Request,
    call_next: RequestResponseEndpoint,
) -> Response:
    """Log request metadata only; never capture request or response bodies."""
    request_id = token_hex(8)
    started_at = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        route_path = getattr(request.scope.get("route"), "path", "unmatched")
        logger.error(
            "request.failed",
            request_id=request_id,
            method=request.method,
            path=route_path,
        )
        raise

    duration_ms = round((perf_counter() - started_at) * 1000, 2)
    route_path = getattr(request.scope.get("route"), "path", "unmatched")
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request.completed",
        request_id=request_id,
        method=request.method,
        path=route_path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


app.include_router(health.router)
app.include_router(chat_completions.router, prefix="/v1")
