"""Domain models for the encrypted token vault.

None of these types ever hold plaintext sensitive values; plaintext exists only
transiently inside the vault service and in Phase 1 ``TokenMapping`` objects.
"""

import struct
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from app.models.privacy import EntityType
from app.vault.errors import VaultAuthorizationError

VAULT_FORMAT_VERSION = 1
_AAD_DOMAIN = b"privacy-gateway/vault"


class VaultPermission(StrEnum):
    """Operations a principal may be authorized to perform against the vault."""

    STORE = "store"
    DETOKENIZE = "detokenize"


@dataclass(frozen=True, slots=True)
class VaultPrincipal:
    """An authenticated caller scoped to exactly one tenant.

    Instances must be created by the (future) authentication layer from verified
    credentials, never from request content. The vault trusts ``tenant_id`` and
    enforces ``permissions`` before touching storage or keys.
    """

    tenant_id: UUID
    permissions: frozenset[VaultPermission]

    def require(self, permission: VaultPermission) -> None:
        if permission not in self.permissions:
            raise VaultAuthorizationError()


def encode_fields(*fields: bytes) -> bytes:
    """Unambiguously join fields so distinct contexts can never encode identically."""
    return b"".join(struct.pack(">I", len(item)) + item for item in fields)


@dataclass(frozen=True, slots=True)
class EncryptionContext:
    """Non-secret values cryptographically bound to a ciphertext as AAD.

    Moving a ciphertext to another tenant, token, or entity type changes the
    context, so authenticated decryption fails.
    """

    tenant_id: UUID
    token_hash: bytes = field(repr=False)
    entity_type: EntityType
    format_version: int = VAULT_FORMAT_VERSION

    def to_aad(self) -> bytes:
        return encode_fields(
            _AAD_DOMAIN,
            str(self.format_version).encode(),
            self.tenant_id.bytes,
            self.token_hash,
            self.entity_type.value.encode(),
        )

    def as_kms_context(self) -> dict[str, str]:
        """String map form, matching KMS "encryption context" style APIs."""
        return {
            "format_version": str(self.format_version),
            "tenant_id": str(self.tenant_id),
            "token_hash": self.token_hash.hex(),
            "entity_type": self.entity_type.value,
        }


@dataclass(frozen=True, slots=True)
class DataKey:
    """A freshly generated data-encryption key and its KEK-wrapped form."""

    key_id: str
    plaintext: bytes = field(repr=False)
    wrapped: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class EncryptedValue:
    """Envelope-encrypted value: AES-GCM ciphertext plus its wrapped data key."""

    key_id: str
    wrapped_dek: bytes = field(repr=False)
    nonce: bytes = field(repr=False)
    ciphertext: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class VaultRecord:
    """One persisted vault row. Contains only ciphertext and non-sensitive metadata."""

    tenant_id: UUID
    token_hash: bytes = field(repr=False)
    entity_type: EntityType
    encrypted: EncryptedValue
    created_at: datetime
    expires_at: datetime
    format_version: int = VAULT_FORMAT_VERSION
