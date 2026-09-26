"""Security tests for vault envelope encryption and prototype key handling."""

import base64
from dataclasses import replace
from uuid import uuid4

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.models.privacy import EntityType
from app.vault.crypto import EnvelopeCipher
from app.vault.errors import VaultConfigurationError, VaultIntegrityError, VaultKeyUnavailableError
from app.vault.keys import HmacTokenHasher, LocalKeyProvider, build_local_key_material
from app.vault.models import EncryptedValue, EncryptionContext
from tests.vault_support import EMAIL, new_key


def context(**overrides: object) -> EncryptionContext:
    values: dict[str, object] = {
        "tenant_id": uuid4(),
        "token_hash": b"\x01" * 32,
        "entity_type": EntityType.EMAIL_ADDRESS,
    }
    values.update(overrides)
    return EncryptionContext(**values)  # type: ignore[arg-type]


def cipher(key: bytes | None = None, key_id: str = "dev-1") -> EnvelopeCipher:
    return EnvelopeCipher(LocalKeyProvider({key_id: key or new_key()}, key_id))


def flip_last_byte(data: bytes) -> bytes:
    return data[:-1] + bytes([data[-1] ^ 0x01])


async def test_encrypt_decrypt_round_trip() -> None:
    ctx = context()
    service = cipher()
    encrypted = await service.encrypt(EMAIL, ctx)
    assert await service.decrypt(encrypted, ctx) == EMAIL


async def test_ciphertext_does_not_contain_plaintext() -> None:
    encrypted = await cipher().encrypt(EMAIL, context())
    for blob in (encrypted.ciphertext, encrypted.wrapped_dek, encrypted.nonce):
        assert EMAIL.encode() not in blob
    assert encrypted.ciphertext != EMAIL.encode()


async def test_same_value_encrypts_differently_each_time() -> None:
    ctx, service = context(), cipher()
    first, second = await service.encrypt(EMAIL, ctx), await service.encrypt(EMAIL, ctx)
    assert first.nonce != second.nonce
    assert first.ciphertext != second.ciphertext
    assert first.wrapped_dek != second.wrapped_dek


@pytest.mark.parametrize("field", ["ciphertext", "nonce", "wrapped_dek"])
async def test_modified_encrypted_fields_fail_integrity(field: str) -> None:
    ctx, service = context(), cipher()
    encrypted = await service.encrypt(EMAIL, ctx)
    tampered = replace(encrypted, **{field: flip_last_byte(getattr(encrypted, field))})  # type: ignore[arg-type]
    with pytest.raises(VaultIntegrityError):
        await service.decrypt(tampered, ctx)


@pytest.mark.parametrize(
    "override",
    [
        {"tenant_id": uuid4()},
        {"token_hash": b"\x02" * 32},
        {"entity_type": EntityType.PERSON},
        {"format_version": 2},
    ],
)
async def test_context_mismatch_fails_integrity(override: dict[str, object]) -> None:
    original = context()
    service = cipher()
    encrypted = await service.encrypt(EMAIL, original)
    moved = replace(original, **override)  # type: ignore[arg-type]
    with pytest.raises(VaultIntegrityError):
        await service.decrypt(encrypted, moved)


async def test_truncated_values_fail_integrity() -> None:
    ctx, service = context(), cipher()
    encrypted = await service.encrypt(EMAIL, ctx)
    for broken in (
        EncryptedValue(encrypted.key_id, encrypted.wrapped_dek, encrypted.nonce, b""),
        EncryptedValue(encrypted.key_id, encrypted.wrapped_dek, b"", encrypted.ciphertext),
        EncryptedValue(encrypted.key_id, b"short", encrypted.nonce, encrypted.ciphertext),
    ):
        with pytest.raises(VaultIntegrityError):
            await service.decrypt(broken, ctx)


async def test_wrong_master_key_fails_integrity() -> None:
    ctx = context()
    encrypted = await cipher(new_key()).encrypt(EMAIL, ctx)
    with pytest.raises(VaultIntegrityError):
        await cipher(new_key()).decrypt(encrypted, ctx)


async def test_unknown_key_id_fails_closed() -> None:
    ctx = context()
    encrypted = await cipher(key_id="retired").encrypt(EMAIL, ctx)
    with pytest.raises(VaultKeyUnavailableError):
        await cipher(key_id="current").decrypt(encrypted, ctx)


async def test_rotation_keeps_old_records_readable() -> None:
    ctx, old_key, new = context(), new_key(), new_key()
    encrypted = await cipher(old_key, "k1").encrypt(EMAIL, ctx)
    rotated = EnvelopeCipher(LocalKeyProvider({"k1": old_key, "k2": new}, "k2"))
    assert await rotated.decrypt(encrypted, ctx) == EMAIL
    assert (await rotated.encrypt(EMAIL, ctx)).key_id == "k2"


async def test_token_hash_is_keyed_and_tenant_scoped() -> None:
    token = "<PII_EMAIL_ADDRESS:" + "a" * 32 + ">"
    tenant_a, tenant_b = uuid4(), uuid4()
    hasher = HmacTokenHasher(new_key())
    digest = await hasher.hash_token(tenant_a, token)
    assert len(digest) == 32
    assert digest == await hasher.hash_token(tenant_a, token)
    assert digest != await hasher.hash_token(tenant_b, token)
    assert digest != await HmacTokenHasher(new_key()).hash_token(tenant_a, token)
    assert token.encode() not in digest


def b64(key: bytes) -> str:
    return base64.b64encode(key).decode()


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "development",
        "vault_master_keys": SecretStr(f"dev-1:{b64(new_key())}"),
        "vault_active_key_id": "dev-1",
        "vault_token_hash_key": SecretStr(b64(new_key())),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type, call-arg]


def test_local_key_material_builds_from_settings() -> None:
    provider, hasher = build_local_key_material(settings())
    assert isinstance(provider, LocalKeyProvider)
    assert isinstance(hasher, HmacTokenHasher)


@pytest.mark.parametrize(
    "overrides",
    [
        {"vault_master_keys": None},
        {"vault_active_key_id": None},
        {"vault_token_hash_key": None},
        {"vault_active_key_id": "missing"},
        {"vault_master_keys": SecretStr("dev-1:not-base64!!")},
        {"vault_master_keys": SecretStr(f"dev-1:{b64(b'short')}")},
        {"vault_master_keys": SecretStr("no-separator")},
        {"vault_master_keys": SecretStr(f"dev-1:{b64(new_key())},dev-1:{b64(new_key())}")},
        {"vault_token_hash_key": SecretStr(b64(b"short"))},
        {"environment": "production"},
    ],
)
def test_invalid_or_unsafe_key_configuration_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(VaultConfigurationError):
        build_local_key_material(settings(**overrides))


def test_hash_key_must_differ_from_master_keys() -> None:
    shared = b64(new_key())
    with pytest.raises(VaultConfigurationError):
        build_local_key_material(
            settings(
                vault_master_keys=SecretStr(f"dev-1:{shared}"),
                vault_token_hash_key=SecretStr(shared),
            )
        )


def test_configuration_errors_and_reprs_do_not_reveal_keys() -> None:
    secret = b64(new_key())
    configured = settings(vault_master_keys=SecretStr(f"dev-1:{secret}"))
    assert secret not in repr(configured)
    assert secret not in str(configured.model_dump())
    provider, hasher = build_local_key_material(configured)
    assert secret not in repr(provider) + repr(hasher)

    with pytest.raises(VaultConfigurationError) as error:
        build_local_key_material(settings(vault_master_keys=SecretStr(f"dev-1:{secret[:-4]}")))
    assert secret[:-4] not in str(error.value)
