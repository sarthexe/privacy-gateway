"""Shared helpers for vault tests. All values are synthetic; no real PII."""

import asyncio
import io
import re
import secrets
from collections.abc import Collection, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import structlog

from app.models.privacy import EntitySpan, EntityType
from app.tokenization import SpanTokenizer
from app.vault.crypto import EnvelopeCipher
from app.vault.errors import TokenCollisionError
from app.vault.keys import HmacTokenHasher, LocalKeyProvider
from app.vault.models import VaultPermission, VaultPrincipal, VaultRecord
from app.vault.repository import VaultRepository
from app.vault.service import TokenVault, VaultPrivacyService

EMAIL = "casey.synthetic@example.test"
PERSON = "Quillon Vantaberg"
TEXT = f"Contact {PERSON} at {EMAIL} today."
TTL = timedelta(hours=1)


def new_key() -> bytes:
    return secrets.token_bytes(32)


class Clock:
    """Mutable UTC clock for deterministic expiry tests."""

    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


class SyntheticDetector:
    """Deterministic detector for the synthetic values used in these tests."""

    _PATTERNS = (
        (EntityType.EMAIL_ADDRESS, re.compile(r"[a-z.]+@example\.test")),
        (EntityType.PERSON, re.compile(r"Quillon Vantaberg|Ysolde Marrowick")),
    )

    def detect(self, text: str) -> list[EntitySpan]:
        return [
            EntitySpan(entity_type=kind, start=m.start(), end=m.end(), value=m.group(0))
            for kind, pattern in self._PATTERNS
            for m in pattern.finditer(text)
        ]


class InMemoryVaultRepository:
    """Repository fake with the same no-overwrite and tenant-scoping semantics."""

    def __init__(self) -> None:
        self.rows: dict[tuple[UUID, bytes], VaultRecord] = {}
        self.calls = 0
        self._lock = asyncio.Lock()

    async def insert_many(self, records: Sequence[VaultRecord]) -> None:
        self.calls += 1
        async with self._lock:
            keys = [(record.tenant_id, record.token_hash) for record in records]
            if len(set(keys)) != len(keys) or any(key in self.rows for key in keys):
                raise TokenCollisionError()
            self.rows.update(zip(keys, records, strict=True))

    async def fetch_active(
        self,
        tenant_id: UUID,
        token_hashes: Collection[bytes],
        now: datetime,
    ) -> list[VaultRecord]:
        self.calls += 1
        return [
            record
            for (row_tenant, token_hash), record in self.rows.items()
            if row_tenant == tenant_id and token_hash in token_hashes and record.expires_at > now
        ]

    async def delete_expired(self, now: datetime) -> int:
        self.calls += 1
        expired = [key for key, record in self.rows.items() if record.expires_at <= now]
        for key in expired:
            del self.rows[key]
        return len(expired)

    def tamper(self, ciphertext: bytes) -> None:
        """Overwrite the ciphertext of every stored row (simulates DB tampering)."""
        for key, record in self.rows.items():
            self.rows[key] = replace(
                record, encrypted=replace(record.encrypted, ciphertext=ciphertext)
            )


def principal(
    tenant_id: UUID | None = None,
    *permissions: VaultPermission,
) -> VaultPrincipal:
    granted = permissions or (VaultPermission.STORE, VaultPermission.DETOKENIZE)
    return VaultPrincipal(tenant_id=tenant_id or uuid4(), permissions=frozenset(granted))


def make_vault(
    repository: VaultRepository,
    *,
    master_key: bytes | None = None,
    hash_key: bytes | None = None,
    key_id: str = "dev-1",
    clock: Clock | None = None,
    extra_keys: dict[str, bytes] | None = None,
) -> TokenVault:
    keys = {key_id: master_key or new_key(), **(extra_keys or {})}
    return TokenVault(
        repository,
        EnvelopeCipher(LocalKeyProvider(keys, key_id)),
        HmacTokenHasher(hash_key or new_key()),
        ttl=TTL,
        clock=clock or Clock(),
    )


def make_service(vault: TokenVault) -> VaultPrivacyService:
    return VaultPrivacyService(SpanTokenizer(SyntheticDetector()), vault)


def production_style_logger(buffer: io.StringIO) -> structlog.typing.FilteringBoundLogger:
    """A logger rendering exactly like ``configure_logging`` does, into ``buffer``."""
    return structlog.wrap_logger(  # type: ignore[no-any-return]
        structlog.PrintLogger(file=buffer),
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )
