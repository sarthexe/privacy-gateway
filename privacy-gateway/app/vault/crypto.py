"""Envelope encryption service for vault values.

Each value is sealed with AES-256-GCM under its own random data key (DEK). The
DEK is wrapped by a ``KeyProvider`` key-encryption key. The same
``EncryptionContext`` is bound as associated data at both layers, so any change to
the ciphertext, nonce, wrapped key, tenant, token, or entity type fails closed.
"""

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.vault.errors import VaultIntegrityError
from app.vault.keys import NONCE_BYTES, TAG_BYTES, KeyProvider
from app.vault.models import EncryptedValue, EncryptionContext


class EnvelopeCipher:
    """Encrypt and decrypt vault values; never logs or exposes plaintext."""

    def __init__(self, key_provider: KeyProvider) -> None:
        self._key_provider = key_provider

    async def encrypt(self, plaintext: str, context: EncryptionContext) -> EncryptedValue:
        data_key = await self._key_provider.generate_data_key(context)
        nonce = os.urandom(NONCE_BYTES)
        ciphertext = AESGCM(data_key.plaintext).encrypt(
            nonce, plaintext.encode("utf-8"), context.to_aad()
        )
        return EncryptedValue(
            key_id=data_key.key_id,
            wrapped_dek=data_key.wrapped,
            nonce=nonce,
            ciphertext=ciphertext,
        )

    async def decrypt(self, encrypted: EncryptedValue, context: EncryptionContext) -> str:
        if len(encrypted.nonce) != NONCE_BYTES or len(encrypted.ciphertext) < TAG_BYTES:
            raise VaultIntegrityError()
        data_key = await self._key_provider.decrypt_data_key(
            encrypted.key_id, encrypted.wrapped_dek, context
        )
        try:
            plaintext = AESGCM(data_key).decrypt(
                encrypted.nonce, encrypted.ciphertext, context.to_aad()
            )
            return plaintext.decode("utf-8")
        except (InvalidTag, ValueError, UnicodeDecodeError):
            raise VaultIntegrityError() from None
