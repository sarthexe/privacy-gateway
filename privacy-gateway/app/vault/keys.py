"""Key management boundary for the vault.

The vault depends only on the ``KeyProvider`` and ``TokenHasher`` protocols. Their
shapes mirror cloud KMS APIs (GenerateDataKey / Decrypt with an encryption
context, and GenerateMac), so a KMS- or HSM-backed implementation can replace the
local prototype classes below without changing the vault schema or service.

PROTOTYPE WARNING: ``LocalKeyProvider`` and ``HmacTokenHasher`` hold long-lived
keys in process memory, loaded from environment variables. That is acceptable for
local development only; it is not production-grade key management.
"""

import base64
import binascii
import hashlib
import hmac
import os
import re
from collections.abc import Mapping
from typing import Protocol
from uuid import UUID

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import Settings
from app.vault.errors import VaultConfigurationError, VaultIntegrityError, VaultKeyUnavailableError
from app.vault.models import DataKey, EncryptionContext, encode_fields

KEY_BYTES = 32
NONCE_BYTES = 12
TAG_BYTES = 16
_KEY_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_WRAP_DOMAIN = b"privacy-gateway/vault/dek-wrap"
_HASH_DOMAIN = b"privacy-gateway/vault/token-hash"


class KeyProvider(Protocol):
    """Issues and unwraps per-record data keys under a key-encryption key."""

    async def generate_data_key(self, context: EncryptionContext) -> DataKey:
        """Return a new 256-bit data key and its wrapped form, bound to ``context``."""

    async def decrypt_data_key(
        self,
        key_id: str,
        wrapped: bytes,
        context: EncryptionContext,
    ) -> bytes:
        """Unwrap a data key; raise a ``VaultError`` if it cannot be authenticated."""


class TokenHasher(Protocol):
    """Derives the at-rest lookup identifier for a token."""

    async def hash_token(self, tenant_id: UUID, token: str) -> bytes:
        """Return a keyed, tenant-scoped 32-byte digest of ``token``."""


class LocalKeyProvider:
    """PROTOTYPE: wraps data keys with AES-256-GCM using in-memory master keys."""

    def __init__(self, keys: Mapping[str, bytes], active_key_id: str) -> None:
        if active_key_id not in keys:
            raise VaultConfigurationError("Active vault key id is not in the key ring.")
        if any(len(key) != KEY_BYTES for key in keys.values()):
            raise VaultConfigurationError("Vault master keys must be 32 bytes.")
        self._keys = {key_id: AESGCM(key) for key_id, key in keys.items()}
        self._active_key_id = active_key_id

    def __repr__(self) -> str:
        return f"LocalKeyProvider(active_key_id={self._active_key_id!r})"

    async def generate_data_key(self, context: EncryptionContext) -> DataKey:
        key_id = self._active_key_id
        data_key = AESGCM.generate_key(bit_length=KEY_BYTES * 8)
        nonce = os.urandom(NONCE_BYTES)
        wrapped = nonce + self._keys[key_id].encrypt(
            nonce, data_key, self._wrap_aad(key_id, context)
        )
        return DataKey(key_id=key_id, plaintext=data_key, wrapped=wrapped)

    async def decrypt_data_key(
        self,
        key_id: str,
        wrapped: bytes,
        context: EncryptionContext,
    ) -> bytes:
        cipher = self._keys.get(key_id)
        if cipher is None:
            raise VaultKeyUnavailableError()
        if len(wrapped) != NONCE_BYTES + KEY_BYTES + TAG_BYTES:
            raise VaultIntegrityError()
        nonce, sealed = wrapped[:NONCE_BYTES], wrapped[NONCE_BYTES:]
        try:
            return cipher.decrypt(nonce, sealed, self._wrap_aad(key_id, context))
        except InvalidTag:
            raise VaultIntegrityError() from None

    @staticmethod
    def _wrap_aad(key_id: str, context: EncryptionContext) -> bytes:
        return encode_fields(_WRAP_DOMAIN, key_id.encode(), context.to_aad())


class HmacTokenHasher:
    """PROTOTYPE: HMAC-SHA256 token lookup hashes with an in-memory key."""

    def __init__(self, key: bytes) -> None:
        if len(key) < KEY_BYTES:
            raise VaultConfigurationError("Vault token hash key must be at least 32 bytes.")
        self._key = key

    def __repr__(self) -> str:
        return "HmacTokenHasher()"

    async def hash_token(self, tenant_id: UUID, token: str) -> bytes:
        message = encode_fields(_HASH_DOMAIN, tenant_id.bytes, token.encode())
        return hmac.new(self._key, message, hashlib.sha256).digest()


def _decode_key(encoded: str, setting: str) -> bytes:
    try:
        return base64.b64decode(encoded.strip(), validate=True)
    except (binascii.Error, ValueError):
        raise VaultConfigurationError(f"{setting} contains invalid base64.") from None


def parse_key_ring(raw: str) -> dict[str, bytes]:
    """Parse ``"key_id:base64key[,key_id:base64key...]"`` without echoing key material."""
    ring: dict[str, bytes] = {}
    for entry in filter(None, (part.strip() for part in raw.split(","))):
        key_id, separator, encoded = entry.partition(":")
        if not separator or not _KEY_ID_PATTERN.fullmatch(key_id):
            raise VaultConfigurationError("GATEWAY_VAULT_MASTER_KEYS has an invalid entry.")
        if key_id in ring:
            raise VaultConfigurationError("GATEWAY_VAULT_MASTER_KEYS repeats a key id.")
        key = _decode_key(encoded, "GATEWAY_VAULT_MASTER_KEYS")
        if len(key) != KEY_BYTES:
            raise VaultConfigurationError("Vault master keys must be 32 bytes.")
        ring[key_id] = key
    if not ring:
        raise VaultConfigurationError("GATEWAY_VAULT_MASTER_KEYS is empty.")
    if len(set(ring.values())) != len(ring):
        raise VaultConfigurationError("GATEWAY_VAULT_MASTER_KEYS reuses key material.")
    return ring


def build_local_key_material(settings: Settings) -> tuple[LocalKeyProvider, HmacTokenHasher]:
    """Build the prototype key provider and hasher from environment settings.

    Refuses to run in production: production deployments must supply KMS/HSM-backed
    ``KeyProvider`` and ``TokenHasher`` implementations instead.
    """
    if settings.environment.lower() in {"prod", "production"}:
        raise VaultConfigurationError(
            "Environment-variable vault keys are prototype-only; configure a KMS provider."
        )
    if settings.vault_master_keys is None:
        raise VaultConfigurationError("GATEWAY_VAULT_MASTER_KEYS is not set.")
    if not settings.vault_active_key_id:
        raise VaultConfigurationError("GATEWAY_VAULT_ACTIVE_KEY_ID is not set.")
    if settings.vault_token_hash_key is None:
        raise VaultConfigurationError("GATEWAY_VAULT_TOKEN_HASH_KEY is not set.")

    ring = parse_key_ring(settings.vault_master_keys.get_secret_value())
    hash_key = _decode_key(
        settings.vault_token_hash_key.get_secret_value(), "GATEWAY_VAULT_TOKEN_HASH_KEY"
    )
    if hash_key in ring.values():
        raise VaultConfigurationError("Token hash key must differ from every master key.")
    return LocalKeyProvider(ring, settings.vault_active_key_id), HmacTokenHasher(hash_key)
