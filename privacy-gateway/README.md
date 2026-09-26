# Privacy Gateway

A small, privacy-first API foundation for an OpenAI-compatible LLM gateway. It
detects PII with Presidio, replaces it with random tokens, stores the mappings
in an encrypted, tenant-scoped PostgreSQL vault, and reconstructs text from
those tokens. It does not call an LLM provider yet, and the vault is not yet
exposed through HTTP endpoints.

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

The vault's PostgreSQL tests are skipped unless you point them at a disposable
database. Its tables are dropped and recreated:

```bash
GATEWAY_TEST_DATABASE_URL=postgresql+asyncpg://privacy_gateway:change-this-local-password@localhost:5432/vault_test   uv run pytest
```

Apply database migrations with `uv run alembic upgrade head`.

## Token vault

Token mappings are envelope-encrypted with AES-256-GCM and stored in the
`vault_entries` table, keyed by tenant and an HMAC of the token. The design,
threat model, and limitations are in [`docs/vault-security.md`](docs/vault-security.md).

Vault keys are **prototype, development-only** settings (see `.env.example`).
There are no defaults, and the local key provider refuses to run when
`GATEWAY_ENVIRONMENT=production`. Production use requires a KMS/HSM-backed
`KeyProvider`/`TokenHasher`. Never commit key values.

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
  detection/       Presidio PII detection
  tokenization/    Span-based reversible tokenization
  vault/           Encrypted, tenant-scoped token vault
  policy/          Future policy evaluation boundary
  routing/         Future model-provider routing boundary
  reconstruction/  Token detection and detokenization
  models/          API schemas and SQLAlchemy base
alembic/           Database migrations
docs/              Security design notes
tests/             API, transformation, and vault security tests
```

## Privacy and scope

- Request and response bodies are not written to application logs.
- Request logs contain only a generated request ID, method, route path, status,
  and duration.
- No real PII or credentials are included in source or test data.
- Vault logs contain only event names, tenant IDs, counts, and key IDs; never
  values, tokens, token hashes, ciphertext, keys, or SQL parameters.
- Authentication, policy enforcement, and provider routing are future work.