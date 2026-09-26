# Privacy Gateway

A FastAPI foundation for an OpenAI-compatible, privacy-preserving LLM gateway. The current build is intentionally a backend skeleton; PII processing and provider routing are not implemented.

## Run & Operate

- `cd privacy-gateway && docker compose up --build` — run the FastAPI service, PostgreSQL, and Redis
- `cd privacy-gateway && uv sync --extra dev && uv run pytest` — install Python development dependencies and run its tests
- `pnpm --filter @workspace/api-server run dev` — run the workspace's existing Express API service
- `pnpm run typecheck` — typecheck the pnpm workspace
- Gateway environment: see `privacy-gateway/.env.example`

## Stack

- Gateway: Python, FastAPI, Pydantic v2 settings
- Gateway data services: PostgreSQL, Redis, SQLAlchemy async, Alembic
- Gateway quality: pytest, Ruff, mypy, structlog
- Workspace scaffold: pnpm, TypeScript, Express, Drizzle ORM

## Where things live

- `privacy-gateway/app/api` — health and OpenAI-compatible endpoint skeletons
- `privacy-gateway/app/core` — settings and structured logging
- `privacy-gateway/app/db` — async SQLAlchemy engine and health checks
- `privacy-gateway/app/{detection,tokenization,vault,policy,routing,reconstruction}` — reserved module boundaries for later phases
- `privacy-gateway/alembic` — migration environment
- `privacy-gateway/tests` — API contract and health tests
- `lib/api-spec/openapi.yaml` — existing workspace OpenAPI contract

## Architecture decisions

- Keep the Python gateway independent from the starter Express service to preserve the requested FastAPI and async SQLAlchemy stack.
- The chat endpoint validates a minimal OpenAI-style request and returns a clear `501` until a model provider is deliberately added.
- Request logs contain route metadata only; prompt and response bodies are not logged.

## Product

The initial milestone is service scaffolding only: health checks, a request contract, and local PostgreSQL/Redis wiring. It does not process PII or contact an LLM provider.

## User preferences

- Keep the implementation focused on the current milestone; do not add unrelated product features.
- Use synthetic test data only and do not log raw PII.

## Gotchas

- Run Docker Compose commands from `privacy-gateway/`.
- The gateway's `/health` endpoint reports `503` until both PostgreSQL and Redis are reachable.
- `/v1/chat/completions` intentionally returns `501 model_not_configured` until provider routing is implemented.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
- See `privacy-gateway/README.md` for Python service setup and the next implementation boundaries
