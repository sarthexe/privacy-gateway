"""Create the encrypted vault_entries table.

Revision ID: 20260926_0001
Revises:
Create Date: 2026-09-26

Stores envelope-encrypted token values. No plaintext values and no raw tokens
are persisted: rows are keyed by (tenant_id, HMAC(token)).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260926_0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "vault_entries",
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.LargeBinary(length=32), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("key_id", sa.String(length=64), nullable=False),
        sa.Column("wrapped_dek", sa.LargeBinary(), nullable=False),
        sa.Column("nonce", sa.LargeBinary(length=12), nullable=False),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("format_version", sa.SmallInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("tenant_id", "token_hash", name="pk_vault_entries"),
        sa.CheckConstraint("length(token_hash) = 32", name="ck_vault_entries_token_hash_len"),
        sa.CheckConstraint("length(nonce) = 12", name="ck_vault_entries_nonce_len"),
        sa.CheckConstraint("length(ciphertext) >= 16", name="ck_vault_entries_ciphertext_len"),
        sa.CheckConstraint("expires_at > created_at", name="ck_vault_entries_expiry_order"),
    )
    op.create_index("ix_vault_entries_expires_at", "vault_entries", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_vault_entries_expires_at", table_name="vault_entries")
    op.drop_table("vault_entries")
