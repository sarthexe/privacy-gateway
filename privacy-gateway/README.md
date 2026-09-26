# Privacy Gateway

A small, privacy-first API foundation for an OpenAI-compatible LLM gateway. This
initial repository provides the service structure and integration points only;
it does not detect, tokenize, store, or reconstruct PII, and it does not call an
LLM provider.

## Run the service

From this directory, start the API, PostgreSQL, and Redis with one command:

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`. Interactive API docs are at
`http://localhost:8000/docs`.

To stop the services, press Ctrl+C. The named Docker volumes retain local
PostgreSQL and Redis data between runs. To remove those volumes as well, run
`docker compose down --volumes`.

Copy `.env.example` to `.env` to adjust local settings. Gateway settings use
`GATEWAY_`-prefixed names, so they cannot accidentally inherit settings from
another service in the workspace. The example database
password is for local development only; replace it before using this setup in
any shared or hosted environment. Never commit `.env`.

## Endpoints

- `GET /health` checks PostgreSQL and Redis and returns `200` when both respond,
  or `503` with dependency status when either is unavailable.
- `POST /v1/chat/completions` accepts a text-based OpenAI Chat Completions
  request shape. It currently returns `501 model_not_configured` because provider
  routing is intentionally out of scope for this foundation.

Example request with synthetic, non-sensitive content:

```json
{
  "model": "example-model",
  "messages": [
    {"role": "user", "content": "Summarize the public dataset."}
  ]
}
```

## Development checks

Install the app and development tools with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy app
```

For local development without Docker, first start PostgreSQL and Redis, copy
`.env.example` to `.env`, then run:

```bash
uv run uvicorn app.main:app --reload
```

## Structure

```text
app/
  api/             HTTP routes and request handling
  core/            Settings and structured logging
  db/              Async SQLAlchemy engine and database health check
  detection/       Future PII detection boundary
  tokenization/    Future tokenization boundary
  vault/           Future secure storage boundary
  policy/          Future policy evaluation boundary
  routing/         Future model-provider routing boundary
  reconstruction/  Future response reconstruction boundary
  models/          API schemas and SQLAlchemy base
alembic/           Database migration configuration
tests/             API contract and health tests
```

## Privacy and scope

- Request and response bodies are not written to application logs.
- Request logs contain only a generated request ID, method, route path, status,
  and duration.
- No real PII or credentials are included in source or test data.
- Authentication, PII detection, tokenization, encrypted vault persistence,
  policy enforcement, provider routing, and detokenization are future work.