# Token vault: security design (Phase 2)

The vault persists Phase 1 token mappings so that a token produced during
tokenization can be resolved later, after the LLM response returns. PostgreSQL
stores only ciphertext and non-sensitive metadata.

```text
raw text ─► PII detection ─► SpanTokenizer ─► TokenVault.store ─► vault_entries (ciphertext)
                                   │
                                   └─► ProtectedText (tokens only) ─► LLM layer

LLM output ─► find_tokens ─► TokenVault.resolve ─► Detokenizer ─► reconstructed text
                                (authorize → HMAC lookup → decrypt)
```

## Components

| Concern | Module | Notes |
|---|---|---|
| Domain models | `app/vault/models.py`, `errors.py` | Principal, permissions, encryption context, records, fixed-message errors |
| Key handling | `app/vault/keys.py` | `KeyProvider` / `TokenHasher` protocols; **prototype** `LocalKeyProvider`, `HmacTokenHasher` |
| Encryption | `app/vault/crypto.py` | `EnvelopeCipher` (AES-256-GCM, per-record data key) |
| Persistence | `app/vault/orm.py`, `repository.py` | `vault_entries` table, `SqlAlchemyVaultRepository` |
| Service | `app/vault/service.py` | `TokenVault`, `VaultPrivacyService`, `build_token_vault` |

## Record format (`vault_entries`)

| Column | Purpose |
|---|---|
| `tenant_id` (UUID, PK) | Tenant scope. There is no tenants table yet; Phase 5 adds a foreign key and row-level security. |
| `token_hash` (bytea(32), PK) | `HMAC-SHA256(token_hash_key, "token-hash" ‖ tenant_id ‖ token)`. The raw token is never stored. |
| `entity_type` | Needed for reconstruction. It is already visible in the token itself. |
| `key_id` | Identifies the key-encryption key (KEK) that wrapped the data key, which makes key rotation possible. |
| `wrapped_dek` | The per-record data key, wrapped by the KEK. Opaque to the repository; a KMS would return its own ciphertext blob. |
| `nonce` (12 bytes), `ciphertext` | AES-256-GCM output. The ciphertext includes the 16-byte authentication tag. |
| `format_version`, `created_at`, `expires_at` | Non-sensitive lifecycle metadata. |

The table deliberately has **no** plaintext, no value hashes or fingerprints
(these would be dictionary-attackable for low-entropy PII such as phone numbers),
no value lengths, no partial values (for example the last four digits), no
offsets, and no source text.

## Cryptography

- All primitives come from `cryptography`'s `AESGCM`. Nothing is custom.
- Each value is encrypted with a fresh random 256-bit data key and a random
  96-bit nonce. The data key is then wrapped with AES-256-GCM under the active KEK.
- The associated data (AAD) binds the format version, tenant ID, token hash and
  entity type. The key wrap additionally binds `key_id`. As a result, moving a
  ciphertext to another row, tenant or entity type fails authentication.
- Fields in the AAD are length-prefixed so that different contexts can never
  encode to the same bytes.
- Keys are separated by purpose: the token-hash key must differ from every KEK.

## Key handling: prototype vs. future

**Prototype (this phase).** Keys come from environment variables
(`GATEWAY_VAULT_MASTER_KEYS`, `GATEWAY_VAULT_ACTIVE_KEY_ID`,
`GATEWAY_VAULT_TOKEN_HASH_KEY`). They are held as `SecretStr` and stay in
process memory for the life of the process. This is **not** production-grade key
management:

- There is no hardware protection.
- Nothing forces rotation.
- Nothing audits key use.
- Python cannot reliably zero key material.

`build_local_key_material` refuses to run when `GATEWAY_ENVIRONMENT=production`.

**Future (KMS/HSM).** Implement `KeyProvider` with the KMS `GenerateDataKey` and
`Decrypt` calls, passing `EncryptionContext.as_kms_context()` as the encryption
context. Implement `TokenHasher` with a KMS/HSM MAC key, for example
`GenerateMac`. The table, repository and service stay unchanged, because
`key_id` and `wrapped_dek` are already opaque.

## Authorization boundary

- Every vault operation requires a `VaultPrincipal(tenant_id, permissions)`.
- `store` requires `STORE`, and `resolve`/`detokenize` require `DETOKENIZE`.
  These checks run before any hashing, storage access or decryption.
- Principals must be built by the authenticated API layer (Phase 5) from
  verified credentials. They are never built from request content.
- The LLM/routing layer receives only `ProtectedText`, which holds the tokenized
  text and a count. It is never given a vault, a principal or a `TokenMapping`.
- There is no HTTP endpoint for detokenization and no operation that lists
  records.

## Failure behaviour (fail closed)

| Situation | Result |
|---|---|
| Unknown, expired, malformed or other-tenant token | Indistinguishable: absent from the result. `Detokenizer` leaves the token text unchanged. |
| Modified ciphertext, nonce, wrapped key, entity type, tenant or token binding | `VaultIntegrityError` aborts the **whole** call. No partial plaintext is returned. |
| Record references an unknown `key_id` | `VaultKeyUnavailableError`, fail closed |
| Token already exists | `TokenCollisionError`. Existing records are never overwritten. The service re-tokenizes a bounded number of times. |
| Database error or outage | `VaultUnavailableError` with a generic message. The original exception is suppressed so SQL parameters don't leak. Tokenization never falls back to raw text. |
| Missing permission | `VaultAuthorizationError` before any storage access |

## Logging rules

- **Never logged:**
  - plaintext values or detokenized text
  - tokens and token hashes
  - key material and data keys
  - ciphertext
  - SQL parameters (the engine sets `hide_parameters=True`)
- **Logged:** event names (`vault.stored`, `vault.resolved`,
  `vault.integrity_failure`, `vault.authorization_denied`,
  `vault.storage_failure`, `vault.purged`, `vault.token_collision`), tenant IDs,
  counts, key IDs and error class names.
- `TokenMapping` and `EntitySpan` reprs omit `value`.

## Threat model

| Threat | Mitigation | Residual risk |
|---|---|---|
| Database dump or backup theft | Only ciphertext, plus HMACs of tokens | An attacker who also has the process environment has the keys (prototype limitation) |
| Correlating DB rows with tokens seen in LLM provider logs | Tokens are stored only as keyed HMACs | Anyone who holds the hash key can correlate them |
| Token guessing or enumeration | 128-bit random tokens, tenant-scoped lookup, uniform "not found" result, per-call token cap | No rate limiting yet (Phase 5/6) |
| Cross-tenant access | Tenant is in the primary key, the HMAC input and the AAD. A forged row that has been re-keyed to another tenant fails decryption. | Tenant identity is only as strong as the future authentication layer |
| Row tampering or ciphertext swapping by someone with DB write access | GCM authentication with context-bound AAD | Deleting rows (a denial of service) is not detected |
| Token hijack by overwriting a record | Insert-only writes; a primary-key conflict raises an error | None known |
| Leakage through logs or errors | Fixed-message errors, suppressed exception chains, `hide_parameters`, redacted reprs | Future code must keep following these rules |
| Compromised gateway process | Out of scope. The process necessarily holds keys and plaintext while it works. | Needs a KMS/HSM plus process isolation |

## Known limitations

- Prototype key handling only (see above). Keys aren't rotated automatically and
  records aren't re-wrapped automatically.
- Random 96-bit nonces are safe for roughly 2³² encryptions per key. Because each
  record has its own data key, this limit applies to KEK wraps: rotate the KEK
  well before that many records.
- The same value in two different requests produces two different tokens. This
  is intentional: deduplicating across requests would need a deterministic
  fingerprint, which the design avoids storing.
- Expiry uses the application clock. Purging is a hard delete, but backups may
  still contain the rows (they are ciphertext only).
- The vault is not yet wired into HTTP endpoints. That happens when routing
  (Phase 3) and authentication (Phase 5) exist.
- Presidio tokenization runs synchronously inside the async `tokenize` call.
