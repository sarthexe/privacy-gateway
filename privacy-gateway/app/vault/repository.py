"""Persistence for vault records. Handles ciphertext only; never plaintext or keys."""

from collections.abc import Collection, Sequence
from datetime import datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import delete, insert, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.privacy import EntityType
from app.vault.errors import TokenCollisionError, VaultIntegrityError, VaultUnavailableError
from app.vault.models import EncryptedValue, VaultRecord
from app.vault.orm import VaultEntryRow

_UNIQUE_VIOLATION = "23505"


class VaultRepository(Protocol):
    """Storage interface used by the vault service."""

    async def insert_many(self, records: Sequence[VaultRecord]) -> None:
        """Atomically insert all records; never overwrite an existing token."""

    async def fetch_active(
        self,
        tenant_id: UUID,
        token_hashes: Collection[bytes],
        now: datetime,
    ) -> list[VaultRecord]:
        """Return unexpired records for ``tenant_id`` whose hash is in ``token_hashes``."""

    async def delete_expired(self, now: datetime) -> int:
        """Delete records that expired at or before ``now``; return the count."""


class SqlAlchemyVaultRepository:
    """PostgreSQL-backed vault repository using async SQLAlchemy sessions.

    Database errors are translated into fixed-message vault errors with the
    original exception suppressed, so SQL parameters never reach callers or logs.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def insert_many(self, records: Sequence[VaultRecord]) -> None:
        if not records:
            return
        rows = [_to_row(record) for record in records]
        try:
            async with self._session_factory() as session, session.begin():
                await session.execute(insert(VaultEntryRow), rows)
        except IntegrityError as error:
            if getattr(error.orig, "sqlstate", None) == _UNIQUE_VIOLATION:
                raise TokenCollisionError() from None
            raise VaultUnavailableError() from None
        except (SQLAlchemyError, OSError):
            raise VaultUnavailableError() from None

    async def fetch_active(
        self,
        tenant_id: UUID,
        token_hashes: Collection[bytes],
        now: datetime,
    ) -> list[VaultRecord]:
        if not token_hashes:
            return []
        statement = select(VaultEntryRow).where(
            VaultEntryRow.tenant_id == tenant_id,
            VaultEntryRow.token_hash.in_(list(token_hashes)),
            VaultEntryRow.expires_at > now,
        )
        try:
            async with self._session_factory() as session:
                rows = (await session.scalars(statement)).all()
        except (SQLAlchemyError, OSError):
            raise VaultUnavailableError() from None
        return [_to_record(row) for row in rows]

    async def delete_expired(self, now: datetime) -> int:
        statement = delete(VaultEntryRow).where(VaultEntryRow.expires_at <= now)
        try:
            async with self._session_factory() as session, session.begin():
                result = await session.execute(statement)
        except (SQLAlchemyError, OSError):
            raise VaultUnavailableError() from None
        return int(getattr(result, "rowcount", 0) or 0)


def _to_row(record: VaultRecord) -> dict[str, object]:
    return {
        "tenant_id": record.tenant_id,
        "token_hash": record.token_hash,
        "entity_type": record.entity_type.value,
        "key_id": record.encrypted.key_id,
        "wrapped_dek": record.encrypted.wrapped_dek,
        "nonce": record.encrypted.nonce,
        "ciphertext": record.encrypted.ciphertext,
        "format_version": record.format_version,
        "created_at": record.created_at,
        "expires_at": record.expires_at,
    }


def _to_record(row: VaultEntryRow) -> VaultRecord:
    try:
        entity_type = EntityType(row.entity_type)
    except ValueError:
        raise VaultIntegrityError() from None
    return VaultRecord(
        tenant_id=row.tenant_id,
        token_hash=bytes(row.token_hash),
        entity_type=entity_type,
        encrypted=EncryptedValue(
            key_id=row.key_id,
            wrapped_dek=bytes(row.wrapped_dek),
            nonce=bytes(row.nonce),
            ciphertext=bytes(row.ciphertext),
        ),
        created_at=row.created_at,
        expires_at=row.expires_at,
        format_version=row.format_version,
    )
