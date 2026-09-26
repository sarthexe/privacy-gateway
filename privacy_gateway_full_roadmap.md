# Privacy Gateway — Full Implementation Roadmap

## 0. Project Definition

### Working name
**Privacy Gateway**

### Goal

Build an OpenAI-compatible privacy-preserving LLM gateway inspired by the data-vault model of products such as Skyflow, but specialized for LLM traffic.

The gateway sits between an application and an LLM provider:

```text
Client
  ↓
Privacy Gateway
  ↓
PII Detection
  ↓
Tokenization
  ↓
Secure Vault
  ↓
LiteLLM Router
  ↓
LLM Provider
  ↓
Tokenized Response
  ↓
Detokenization
  ↓
Client
```

The central security property is:

> Raw sensitive data should remain inside the gateway/vault boundary and should not be sent to the downstream LLM provider.

### Initial stack

- Python
- FastAPI
- Pydantic v2
- PostgreSQL
- SQLAlchemy async
- Alembic
- Redis
- Microsoft Presidio
- spaCy
- LiteLLM
- AES-GCM for prototype authenticated encryption
- pytest
- Ruff
- mypy
- structlog
- Docker Compose
- OpenTelemetry / Prometheus later
- Optional dashboard later

### Source-of-truth workflow

GitHub is the canonical repository.

```text
                 GitHub
                /      \
           Replit      Codex/Claude
              |             |
       demo/deployment   core engineering
```

Do not maintain separate independent repositories.

Use one phase at a time:

```text
Implement → Test → Review → Commit → Push
```

Do not have multiple AI coding agents simultaneously changing the same branch.

---

# Phase 0 — Repository Bootstrap

## Objective

Create a clean, production-oriented backend skeleton before implementing privacy functionality.

## Current status

**Completed.**

The repository already contains the initial FastAPI structure, settings, database integration, Redis integration, Alembic, structured logging, tests, and module boundaries.

## Target structure

```text
privacy-gateway/
├── app/
│   ├── api/
│   │   ├── chat.py
│   │   └── health.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── security.py
│   │
│   ├── detection/
│   ├── tokenization/
│   ├── vault/
│   ├── policy/
│   ├── routing/
│   ├── reconstruction/
│   ├── models/
│   └── db/
│
├── tests/
├── migrations/
├── docs/
│   └── architecture.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md
```

## Implementation

1. Configure FastAPI.
2. Configure Pydantic v2 settings.
3. Configure async SQLAlchemy.
4. Configure PostgreSQL.
5. Configure Redis.
6. Add Alembic.
7. Add structured logging.
8. Add `/health`.
9. Add `/v1/chat/completions` as an OpenAI-compatible skeleton.
10. Add pytest.
11. Add Ruff.
12. Add mypy.
13. Add Docker Compose.
14. Ensure environment variables from unrelated parent/starter workspaces cannot accidentally override project settings.

## Acceptance criteria

```text
pytest       → PASS
ruff check . → PASS
mypy .       → PASS
compose config validation → PASS
```

The chat endpoint may return:

```json
{
  "error": {
    "type": "model_not_configured"
  }
}
```

until routing is implemented.

## Tool

**Replit** is appropriate for initial bootstrap and quick runtime validation.

---

# Phase 1 — PII Detection and Reversible Tokenization

## Objective

Build a standalone privacy transformation engine.

It must work without PostgreSQL, Redis, or an LLM.

## Current status

**Completed.**

Phase 1 currently implements:

```text
Raw text
   ↓
Presidio
   ↓
Entity spans
   ↓
Span-based tokenizer
   ↓
Tokenized text + in-memory mappings
   ↓
Detokenizer
   ↓
Original text
```

17 tests passed, along with Ruff and mypy.

## Supported entities

Initial set:

```text
PERSON
EMAIL_ADDRESS
PHONE_NUMBER
CREDIT_CARD
IP_ADDRESS
DATE_TIME
LOCATION
```

Future entities may include:

```text
SSN
PASSPORT
BANK_ACCOUNT
IBAN
MEDICAL_ID
DRIVER_LICENSE
ORGANIZATION
ADDRESS
```

## Detection architecture

Create an abstraction:

```python
class PIIDetector(Protocol):
    async def detect(self, text: str) -> list[Entity]:
        ...
```

Implement:

```text
PIIDetector
    ↓
PresidioDetector
    ↓
Presidio Analyzer
    ↓
Entity spans
```

## Entity model

Use a strongly typed model containing at least:

```text
entity_type
start
end
value
score
```

Offsets should be half-open:

```text
[start, end)
```

## Token format

Use a machine-parseable format such as:

```text
<PII_PERSON:01HX...>
<PII_EMAIL_ADDRESS:01HY...>
```

Requirements:

- token IDs must not contain plaintext PII
- token IDs must be non-predictable
- different values must not collide
- token format must be easy to parse later
- token identity must be scoped to the relevant transformation/session
- raw PII must never be logged

## Tokenization

Do not perform naive sequential string replacement.

Use:

```text
original text
+ entity spans
→ sorted/validated spans
→ untouched text slices
→ tokens
```

This preserves every non-sensitive character exactly.

## Repeated values

Within one transformation:

```text
John → <PII_PERSON:abc>
John → <PII_PERSON:abc>
```

Different values:

```text
John → <PII_PERSON:abc>
Alice → <PII_PERSON:def>
```

Identity should be scoped to the transformation/session until persistent vault semantics are introduced.

## Detokenization

The detokenizer must:

1. parse valid token patterns
2. look up mappings
3. replace recognized tokens
4. leave unknown tokens unchanged
5. leave malformed tokens unchanged
6. never crash because of model-generated text

## Critical invariant

```text
detokenize(tokenize(original)) == original
```

## Tests

Must cover:

- no PII
- one PII entity
- multiple entity types
- repeated same entity
- repeated different entities
- punctuation
- Unicode
- multiline content
- adjacent entities
- beginning/end boundaries
- overlapping spans
- unknown token
- malformed token
- round-trip transformation

## Acceptance criteria

The transformation engine is independent from the API layer and can be tested as a pure domain component.

## Tool

**Codex** should be the primary implementation agent.

---

# Phase 2 — Secure Encrypted Vault

## Objective

Persist token mappings without storing plaintext sensitive values in PostgreSQL.

## Status

**Next phase.**

This is the first security-critical persistence boundary.

## Core flow

```text
Token
  ↓
Vault service
  ↓
Encrypt plaintext value
  ↓
PostgreSQL
```

At reconstruction time:

```text
Token
  ↓
Authorization
  ↓
Vault lookup
  ↓
Decrypt
  ↓
Original value
```

## Database model

Initial table:

```text
token_mapping
────────────────────────────
id
tenant_id
token_id
entity_type
ciphertext
nonce
created_at
expires_at
```

Do not store unnecessary plaintext-sensitive metadata.

Potential future fields:

```text
key_version
created_by
request_id
metadata
```

Only add metadata with a clear use case.

## Encryption

Use a vetted authenticated-encryption primitive such as AES-GCM.

Do not invent cryptography.

The encryption abstraction should look conceptually like:

```python
class EncryptionService(Protocol):
    def encrypt(self, plaintext: bytes, ...) -> Ciphertext:
        ...

    def decrypt(self, ciphertext: Ciphertext, ...) -> bytes:
        ...
```

Separate:

```text
Encryption service
Vault repository
Vault service
Configuration/key handling
```

This makes future replacement with KMS/HSM easier.

## Key management

Prototype:

```text
APP_ENCRYPTION_KEY
```

in environment/configuration.

Future:

```text
Gateway
   ↓
KMS
   ↓
Data-encryption key / key-encryption key
```

Do not claim the application-managed development key is enterprise-grade key management.

## Token lookup

Never make the system rely on guessable or sequential token IDs.

Potential security pattern:

```text
token_id
  ↓
safe lookup/index
  ↓
tenant-scoped record
```

Consider storing a derived lookup identifier separately from the encrypted value where useful.

## Tenant isolation

Even before full authentication is implemented, design the vault API around:

```text
tenant_id + token_id
```

rather than token ID alone.

## Authorization boundary

Detokenization must require explicit authorization.

Conceptually:

```text
request
 ↓
authenticated principal
 ↓
tenant
 ↓
policy
 ↓
authorized vault lookup
 ↓
decryption
```

Never expose an unrestricted:

```text
GET /detokenize/<token>
```

style primitive.

## Failure behavior

The vault should fail closed.

Examples:

```text
wrong key
→ decryption error
→ no plaintext returned

modified ciphertext
→ authentication/integrity failure
→ no plaintext returned

wrong tenant
→ not found/denied
→ no plaintext returned
```

## Tests

Add:

- store/retrieve round trip
- ciphertext differs from plaintext
- ciphertext modification detection
- wrong-key failure
- wrong-tenant access
- unknown-token behavior
- expiration
- duplicate token handling
- concurrent store/lookup behavior
- logging-leak test
- vault failure does not return plaintext

## Tool

Use:

```text
Claude Code → architecture + security design
Codex       → implementation + tests
```

Recommended sequence:

```text
Claude reviews/designs
        ↓
Codex implements
        ↓
Claude/Codex security review
        ↓
Tests
        ↓
Commit
```

---

# Phase 3 — LiteLLM Privacy Proxy

## Objective

Connect the privacy engine and encrypted vault to an OpenAI-compatible LLM gateway.

## Core request flow

```text
Client
  ↓
POST /v1/chat/completions
  ↓
Authentication
  ↓
Extract messages
  ↓
PII detection
  ↓
Tokenization
  ↓
Persist mappings
  ↓
Sanitized/tokenized messages
  ↓
LiteLLM
  ↓
LLM provider
```

Response flow:

```text
LLM provider
  ↓
Tokenized response
  ↓
Token parser
  ↓
Authorization
  ↓
Vault lookup/decryption
  ↓
Detokenization
  ↓
OpenAI-compatible response
  ↓
Client
```

## OpenAI compatibility

The gateway should expose:

```text
POST /v1/chat/completions
```

and preserve the essential OpenAI-compatible request/response structure.

For example:

```python
client.chat.completions.create(
    model="gpt-...",
    messages=[...]
)
```

should be able to use the gateway as the base URL.

## Supported message types

Process:

```text
system
user
assistant
```

without losing their role or ordering.

## Provider routing

Do not write individual provider integrations first.

Use LiteLLM:

```text
Privacy Gateway
      ↓
LiteLLM
 ┌────┼────┐
OpenAI Claude Gemini
```

Start with one provider for integration tests, then verify additional providers.

## Fake LLM provider

Build a deterministic mock provider for tests.

The mock should expose what it received so tests can assert:

```text
raw PII sent downstream = false
```

## Security test

Given:

```text
My email is john@example.com
```

the outbound provider payload must contain something like:

```text
My email is <PII_EMAIL_ADDRESS:...>
```

and never:

```text
john@example.com
```

## Acceptance criteria

An end-to-end request should work:

```text
Original request
→ tokenized outbound request
→ mock/provider
→ tokenized response
→ reconstructed response
```

while preserving the expected API semantics.

## Tool

**Codex** should implement the pipeline.

---

# Phase 4 — Policy Engine

## Objective

Allow tenants/applications to control how each entity type is handled.

## Policy actions

Initial actions:

```text
ALLOW
TOKENIZE
REDACT
MASK
HASH
BLOCK
```

## Example

```yaml
EMAIL_ADDRESS:
  action: TOKENIZE

PHONE_NUMBER:
  action: TOKENIZE

CREDIT_CARD:
  action: BLOCK

PERSON:
  action: TOKENIZE

IP_ADDRESS:
  action: REDACT

DATE_TIME:
  action: ALLOW
```

## Policy flow

```text
Detected entity
      ↓
Policy engine
      ↓
Action
 ┌────┼────┬────┬─────┐
allow tokenize redact mask block
```

## Design

Create:

```python
class PolicyEngine(Protocol):
    def evaluate(
        self,
        entity: Entity,
        context: PolicyContext,
    ) -> PolicyDecision:
        ...
```

Avoid embedding policy logic directly into the Presidio detector.

Detection answers:

> What is this?

Policy answers:

> What should we do with it?

## Context

Future policy context may include:

```text
tenant
application
user/service identity
model
provider
request type
environment
```

## Important model behavior

Policies should apply before sensitive data leaves the gateway.

For example:

```text
CREDIT_CARD → BLOCK
```

must prevent the card from being sent to LiteLLM.

## Tests

Test:

- each action
- default policy
- unknown entity
- conflicting policies
- tenant-specific rules
- provider-specific rules
- policy precedence
- fail-closed behavior for security-sensitive rules

## Tool

**Claude Code** for architecture.

**Codex** for implementation/tests.

---

# Phase 5 — Authentication and Multi-Tenancy

## Objective

Turn the gateway into a shared service where independent clients/tenants cannot access one another's data.

## Core model

```text
API Key
   ↓
Principal
   ↓
Tenant
   ↓
Policy
   ↓
Vault
```

## Authentication

Initial approach:

```text
API keys
```

Potential future:

```text
OAuth2
OIDC
SSO
```

## Tenant isolation

Every persistent vault lookup must be scoped by tenant.

Conceptually:

```sql
SELECT ...
FROM token_mapping
WHERE tenant_id = :tenant
  AND token_id = :token;
```

Never:

```sql
SELECT ...
FROM token_mapping
WHERE token_id = :token;
```

followed by an application-side tenant guess.

## Key hierarchy

Conceptually:

```text
API Key
  ↓
Authenticated Principal
  ↓
Tenant ID
  ↓
Authorized vault scope
```

## Cross-tenant tests

Create:

```text
Tenant A
  token_A

Tenant B
  token_B
```

and prove:

```text
Tenant A → token_A ✅
Tenant A → token_B ❌
Tenant B → token_A ❌
```

## API key requirements

- hashed at rest where practical
- never log complete API keys
- support revocation
- support rotation
- constant-time comparison for secret material where applicable

## Tool

**Codex**.

Claude should perform a security review afterward.

---

# Phase 6 — Audit Logging and Observability

## Objective

Make the gateway observable without leaking PII.

## Audit events

Track events such as:

```text
request_received
pii_detected
entity_tokenized
policy_blocked
llm_request_sent
llm_response_received
token_reconstructed
authorization_denied
vault_failure
```

## What not to log

Never log:

```text
raw prompts containing PII
raw LLM responses containing PII
plaintext vault values
encryption keys
complete API keys
authorization secrets
```

## Safe event example

```json
{
  "event": "entity_tokenized",
  "tenant_id": "tenant_42",
  "entity_type": "EMAIL_ADDRESS",
  "request_id": "req_9182"
}
```

Avoid:

```json
{
  "email": "john@example.com"
}
```

## Correlation

Use:

```text
request_id
trace_id
tenant_id
```

to follow a request across:

```text
API
→ detector
→ tokenizer
→ vault
→ LiteLLM
→ reconstruction
```

without exposing content.

## Metrics

Initial metrics:

```text
requests_total
requests_blocked_total
pii_entities_detected_total
pii_entities_tokenized_total
vault_operations_total
vault_failures_total
provider_requests_total
provider_errors_total
gateway_latency
tokenization_latency
vault_latency
reconstruction_latency
```

## Tooling

Recommended:

```text
structlog
OpenTelemetry
Prometheus
Grafana
```

Add tracing only after the request pipeline is stable.

---

# Phase 7 — Token and Reconstruction Hardening

## Objective

Make the token system robust against real-world LLM behavior.

This phase is important because LLMs are not reliable string-preserving systems.

## Problems to handle

Models may:

```text
change token punctuation
change token casing
wrap tokens in markdown
repeat tokens
omit tokens
invent tokens
truncate tokens
copy tokens into unrelated locations
```

Example:

Expected:

```text
<PII_EMAIL_ADDRESS:abc123>
```

Model may return:

```text
Email: <PII_EMAIL_ADDRESS:abc123>.
```

or potentially malformed output.

## Parser

Use a strict token grammar.

Conceptually:

```text
TOKEN =
    <PII_[ENTITY_TYPE]:[TOKEN_ID]>
```

The parser should distinguish:

```text
valid token
unknown token
malformed token
ordinary text
```

## Unknown tokens

Do not silently query arbitrary vault records.

For unknown token:

```text
leave unchanged
```

or use an explicit policy-driven error.

## Token ownership

A token returned by a model must still be validated against:

```text
current tenant
current request/session
current authorization context
```

## Anti-confusion rules

Prevent:

```text
Token from Tenant A
      ↓
Model response
      ↓
Tenant B reconstruction
```

## Token replay

Decide and document whether tokens are:

```text
request-scoped
session-scoped
time-limited
persistent
```

Start with the narrowest useful scope.

## Tests

Generate adversarial model outputs:

```text
valid token
unknown token
malformed token
token with punctuation
duplicate token
mixed valid/unknown tokens
token from another tenant
token from another request
```

---

# Phase 8 — Developer Experience and Dashboard

## Objective

Create a usable demo and operator interface.

This phase should come after the core security architecture works.

## Dashboard

Display synthetic/demo-safe information such as:

```text
Requests                 12,843
PII detections            4,129
Tokenized                 4,002
Blocked                     127
```

Provider distribution:

```text
OpenAI
Claude
Gemini
```

## Request inspector

Show:

```text
Original
────────────────────────────
My email is john@example.com

Sanitized
────────────────────────────
My email is <PII_EMAIL_ADDRESS:9af1>

Provider
────────────────────────────
Claude

Final
────────────────────────────
My email is john@example.com
```

For a real deployed environment, avoid casually exposing raw PII in operator dashboards.

Use synthetic data for demonstrations.

## Developer documentation

Provide examples for:

```text
OpenAI SDK
curl
Python
JavaScript
```

For example:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://gateway.example.com/v1",
    api_key="..."
)
```

The developer should not need to understand the internals to use the gateway.

## Tool

**Replit** is useful here for quick UI iteration and deployment.

---

# Phase 9 — Performance and Privacy Benchmarks

## Objective

Measure the actual overhead introduced by the privacy gateway.

Do not make claims such as "fast" without measurements.

## Baselines

Compare:

```text
Direct LLM
vs
LLM through Privacy Gateway
```

## Metrics

Measure:

```text
total latency
p50 latency
p95 latency
p99 latency
tokenization latency
vault latency
LLM latency
reconstruction latency
throughput
error rate
```

## Privacy/accuracy metrics

Measure PII detection:

```text
precision
recall
F1
```

Measure reconstruction:

```text
round-trip correctness
token recognition accuracy
unknown-token handling
```

## Test datasets

Use synthetic data initially.

Categories:

```text
names
emails
phone numbers
addresses
credit cards
dates
IPs
mixed entities
long prompts
multilingual/Unicode content
```

Never upload real sensitive personal data just to benchmark the prototype.

## Critical security benchmark

Measure:

```text
raw sensitive value in outbound provider request
```

Expected:

```text
0 occurrences
```

## Tool

**Codex** for benchmark harness.

---

# Phase 10 — Adversarial Security Testing

## Objective

Attack the system as if you were an untrusted client, model, provider, or database attacker.

## Threat actors

Consider:

```text
1. malicious API client
2. compromised API key
3. malicious tenant
4. compromised database
5. curious operator
6. malicious/compromised LLM provider
7. prompt injection
8. malicious model output
```

## Attack scenarios

### Token enumeration

Can an attacker guess or iterate tokens?

### Cross-tenant access

Can one tenant resolve another tenant's token?

### Vault compromise

If the database is dumped, are plaintext values exposed?

### Log leakage

Can PII appear in:

```text
application logs
exceptions
traces
metrics
debug logs
```

### Model injection

Can the LLM trick reconstruction into revealing unrelated data?

### Forged token

Can a model invent a token that resolves to real sensitive data?

### Replay

Can a token from an old request be reused?

### Authorization bypass

Can an unauthenticated request reach:

```text
vault
reconstruction
detokenization
```

### Key failure

What happens when encryption keys are incorrect, missing, rotated, or corrupted?

## Security principle

The model should never get an arbitrary API into the vault.

The intended flow is:

```text
LLM
  X
  │
  │ no direct vault access
  │
  X
Vault
```

The gateway controls the reconstruction decision.

## Deliverables

Create a security review document containing:

```text
threat
impact
attack path
existing control
remaining weakness
mitigation
```

## Tool

**Claude Code** should lead the threat-model/security review.

**Codex** should implement confirmed fixes and regression tests.

---

# Phase 11 — Production Architecture Improvements

## Objective

Move from a portfolio-grade prototype toward a production-oriented architecture.

Only begin this phase after the previous phases are stable.

## Improvements

Potential additions:

```text
KMS / HSM-backed key management
Redis-backed short-lived session mappings
rate limiting
request quotas
secret rotation
key rotation
API key rotation
RBAC
OIDC/SSO
dead-letter/error handling
provider failover
circuit breakers
retry policies
distributed tracing
horizontal scaling
```

## KMS abstraction

Replace:

```text
environment key
```

with:

```text
EncryptionService
       ↓
KMS implementation
```

without changing the Vault interface.

## Provider resilience

LiteLLM can become:

```text
               LiteLLM
             /    |    \
        OpenAI  Claude  Gemini
           \      |      /
            failover/retry
```

but preserve the privacy transformation before provider routing.

## Scaling model

Potential architecture:

```text
                 Load Balancer
                       ↓
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       Gateway      Gateway      Gateway
          │            │            │
          └────────────┼────────────┘
                       ↓
                  PostgreSQL
                       +
                     Redis
                       +
                    LiteLLM
```

---

# Phase 12 — SDK and Integration Layer

## Objective

Make adoption nearly frictionless.

## Initial integrations

Provide:

```text
OpenAI-compatible base URL
```

This may already eliminate the need for a custom SDK for many users.

Then optionally create:

```text
Python SDK
TypeScript SDK
```

## Example

```python
from privacy_gateway import PrivacyOpenAI

client = PrivacyOpenAI(
    api_key="..."
)

response = client.chat.completions.create(
    model="gpt-...",
    messages=[...]
)
```

Internally:

```text
SDK
 ↓
Gateway
```

## Documentation

Document:

```text
installation
API keys
provider configuration
privacy policies
supported entities
token behavior
failure behavior
security model
limitations
```

---

# Phase 13 — Final Demo and Portfolio Packaging

## Objective

Turn the engineering project into a clear technical case study.

## Demo scenario

Use:

```text
Customer support assistant
```

Example input:

```text
Customer John Smith's email is
john.smith@example.com and order number
83921 has not arrived.
```

Gateway transforms:

```text
Customer <PII_PERSON:...>'s email is
<PII_EMAIL_ADDRESS:...> and order number
<PII_ORDER_ID:...> has not arrived.
```

The LLM receives only the tokenized data.

The LLM responds:

```text
Hi <PII_PERSON:...>,

I can help investigate order <PII_ORDER_ID:...>.
```

Gateway reconstructs:

```text
Hi John Smith,

I can help investigate order 83921.
```

## Demo panel

Show four panes:

```text
USER INPUT
     ↓
SANITIZED REQUEST
     ↓
LLM PROVIDER REQUEST
     ↓
RECONSTRUCTED RESPONSE
```

The provider pane should make the core privacy guarantee visually obvious.

## README

Include:

```text
1. Problem
2. Architecture
3. Why tokenization
4. Threat model
5. Data flow
6. Vault design
7. Policy engine
8. LiteLLM integration
9. Security guarantees
10. Known limitations
11. Benchmarks
12. Running locally
13. API usage
```

## Architecture diagram

The final README should contain a diagram similar to:

```text
                       CLIENT
                         │
                         ▼
               ┌───────────────────┐
               │  Privacy Gateway  │
               │                   │
               │ Authentication    │
               │ PII Detection     │
               │ Policy Engine     │
               │ Tokenization      │
               └─────────┬─────────┘
                         │
                Tokenized Request
                         │
                         ▼
                 ┌──────────────┐
                 │   LiteLLM    │
                 └──────┬───────┘
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
           OpenAI     Claude     Gemini
              │         │         │
              └─────────┼─────────┘
                        │
                Tokenized Response
                        │
                        ▼
               ┌──────────────────┐
               │ Reconstruction   │
               │ + Authorization  │
               └────────┬─────────┘
                        │
                        ▼
                      CLIENT


                  ┌───────────────┐
                  │ Secure Vault  │
                  │               │
                  │ Encrypted PII │
                  │ Token Mapping │
                  └───────────────┘
```

---

# Recommended Agent / Tool Allocation

## Replit

Use Replit for:

```text
Phase 0  bootstrap
Phase 8  dashboard/demo
Phase 13 demo/deployment
```

Avoid giving Replit ownership of the security-critical core once the project is established.

## Codex

Use Codex as the primary implementation engineer:

```text
Phase 1  tokenization
Phase 3  LiteLLM integration
Phase 5  authentication/multi-tenancy
Phase 6  observability implementation
Phase 7  parser/reconstruction hardening
Phase 9  benchmark harness
Phase 11 production implementation
Phase 12 SDKs
```

Codex should also handle:

```text
tests
migrations
refactors
bug fixes
CI improvements
```

## Claude Code

Use Claude Code for higher-value reasoning and adversarial review:

```text
Phase 2  vault security design
Phase 4  policy architecture
Phase 6  logging/privacy review
Phase 7  reconstruction security review
Phase 10 threat model
Phase 11 production architecture review
Phase 13 final technical review
```

Recommended loop:

```text
Claude
  ↓
design / threat model
  ↓
Codex
  ↓
implementation
  ↓
tests
  ↓
Claude
  ↓
security/code review
  ↓
Codex
  ↓
fixes + regression tests
```

---

# Phase Dependency Graph

```text
Phase 0
  │
  ▼
Phase 1
  │
  ▼
Phase 2
  │
  ▼
Phase 3
  │
  ├──────────────► Phase 4
  │
  ├──────────────► Phase 5
  │
  └──────────────► Phase 6
                         │
                         ▼
                    Phase 7
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          Phase 8                 Phase 9
              │                     │
              └──────────┬──────────┘
                         ▼
                    Phase 10
                         │
                         ▼
                    Phase 11
                         │
                         ▼
                    Phase 12
                         │
                         ▼
                    Phase 13
```

---

# MVP Definition

Do not expand the project indefinitely.

The first meaningful MVP is:

```text
FastAPI
   ↓
PII Detection
   ↓
Tokenization
   ↓
Encrypted PostgreSQL Vault
   ↓
LiteLLM
   ↓
LLM Provider
   ↓
Tokenized Response
   ↓
Authorized Detokenization
   ↓
Client
```

With:

```text
✅ Authentication
✅ Tenant isolation
✅ Policy engine
✅ Audit-safe logging
✅ Tests
✅ Benchmark
✅ Security review
```

A working MVP is **not** required to have:

```text
❌ Billing
❌ Kubernetes
❌ Enterprise SSO
❌ 20+ providers
❌ Multiple SDKs
❌ Fancy frontend
❌ HSM on day one
❌ Complex organization hierarchy
```

---

# Non-Negotiable Security Properties

The project should continuously test these claims.

## Property 1 — Provider privacy

```text
Raw PII in outbound LLM request = 0
```

## Property 2 — Database privacy

```text
Plaintext sensitive values in vault database = 0
```

## Property 3 — Tenant isolation

```text
Tenant A cannot resolve Tenant B's mappings
```

## Property 4 — Authorization

```text
No authorization → no detokenization
```

## Property 5 — Logging

```text
Raw PII in logs = 0
```

## Property 6 — Round-trip correctness

```text
detokenize(tokenize(x)) == x
```

## Property 7 — Fail closed

Security errors must not silently fall back to plaintext transmission or reconstruction.

---

# Suggested Git Milestones

Use tags/releases such as:

```text
v0.1.0-bootstrap
v0.2.0-tokenization
v0.3.0-secure-vault
v0.4.0-litellm-proxy
v0.5.0-policy-engine
v0.6.0-auth-multitenancy
v0.7.0-observability
v0.8.0-reconstruction-hardening
v0.9.0-benchmarks
v1.0.0-security-reviewed-mvp
```

Each milestone should have:

```text
tests passing
lint passing
type checks passing
documented architecture
git commit/tag
```

---

# Definition of Done for Every Phase

A phase is not complete merely because code exists.

Require:

```text
1. implementation
2. unit/integration tests
3. negative/security tests where applicable
4. lint
5. type checks
6. documentation
7. no unrelated changes
8. Git commit
9. GitHub push
```

For security-sensitive phases also require:

```text
10. threat/attack review
11. regression tests for discovered issues
```

---

# Current Position

At the time this roadmap was created:

```text
Phase 0 — Bootstrap              ✅
Phase 1 — Tokenization           ✅
Phase 2 — Secure Vault           ← NEXT
Phase 3 — LiteLLM Proxy
Phase 4 — Policy Engine
Phase 5 — Auth + Multi-tenancy
Phase 6 — Observability
Phase 7 — Reconstruction Hardening
Phase 8 — Dashboard
Phase 9 — Benchmarks
Phase 10 — Security Testing
Phase 11 — Production Hardening
Phase 12 — SDK/Integrations
Phase 13 — Final Demo/Portfolio
```

## Immediate next action

Commit and push Phase 1:

```bash
git add .
git commit -m "feat: add reversible PII transformation engine"
git push
```

Then start **Phase 2 with Claude Code**, beginning with a security architecture review of the vault before implementation.
