"""SQLAlchemy mapping for encrypted vault records.

The table stores only ciphertext and non-sensitive metadata. Tokens themselves
are not stored; ``token_hash`` is a keyed, tenant-scoped HMAC of the token.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Index,
    LargeBinary,
    SmallInteger,
    String,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class VaultEntryRow(Base):
    """One envelope-encrypted value, addressable only within its tenant."""

    __tablename__ = "vault_entries"
    __table_args__ = (
        CheckConstraint("length(token_hash) = 32", name="ck_vault_entries_token_hash_len"),
        CheckConstraint("length(nonce) = 12", name="ck_vault_entries_nonce_len"),
        CheckConstraint("length(ciphertext) >= 16", name="ck_vault_entries_ciphertext_len"),
        CheckConstraint("expires_at > created_at", name="ck_vault_entries_expiry_order"),
        Index("ix_vault_entries_expires_at", "expires_at"),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True)
    token_hash: Mapped[bytes] = mapped_column(LargeBinary(32), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(32))
    key_id: Mapped[str] = mapped_column(String(64))
    wrapped_dek: Mapped[bytes] = mapped_column(LargeBinary())
    nonce: Mapped[bytes] = mapped_column(LargeBinary(12))
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary())
    format_version: Mapped[int] = mapped_column(SmallInteger())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
