"""Security tests for the vault service using an in-memory repository."""

import asyncio
import io
import logging
import re
from collections.abc import Sequence
from dataclasses import fields, replace

import pytest

from app.models.privacy import EntityType, TokenizationResult, TokenMapping
from app.reconstruction import find_tokens
from app.vault import service as vault_service
from app.vault.errors import (
    TokenCollisionError,
    VaultAuthorizationError,
    VaultError,
    VaultIntegrityError,
    VaultRequestTooLargeError,
    VaultUnavailableError,
)
from app.vault.models import VaultPermission, VaultRecord
from app.vault.service import ProtectedText
from tests.vault_support import (
    EMAIL,
    PERSON,
    TEXT,
    TTL,
    Clock,
    InMemoryVaultRepository,
    make_service,
    make_vault,
    new_key,
    principal,
    production_style_logger,
)

TOKEN_RE = re.compile(r"<PII_[A-Z_]+:[0-9a-f]{32}>")


async def test_store_and_resolve_round_trip() -> None:
    repository = InMemoryVaultRepository()
    service = make_service(make_vault(repository))
    caller = principal()

    protected = await service.tokenize(caller, TEXT)

    assert EMAIL not in protected.text and PERSON not in protected.text
    assert protected.token_count == 2
    assert len(repository.rows) == 2
    assert await service.detokenize(caller, protected.text) == TEXT


async def test_llm_facing_result_carries_no_mappings() -> None:
    protected = await make_service(make_vault(InMemoryVaultRepository())).tokenize(
        principal(), TEXT
    )
    assert {field.name for field in fields(ProtectedText)} == {"text", "token_count"}
    assert EMAIL not in repr(protected) and PERSON not in repr(protected)


async def test_repository_never_receives_plaintext_or_raw_tokens() -> None:
    repository = InMemoryVaultRepository()
    protected = await make_service(make_vault(repository)).tokenize(principal(), TEXT)
    tokens = find_tokens(protected.text)
    for record in repository.rows.values():
        blobs = [
            record.token_hash,
            record.encrypted.ciphertext,
            record.encrypted.wrapped_dek,
            record.encrypted.nonce,
            record.encrypted.key_id.encode(),
        ]
        for secret in (EMAIL, PERSON, *tokens):
            assert all(secret.encode() not in blob for blob in blobs)


async def test_wrong_tenant_is_indistinguishable_from_unknown_token() -> None:
    repository = InMemoryVaultRepository()
    service = make_service(make_vault(repository))
    protected = await service.tokenize(principal(), TEXT)
    other_tenant = principal()

    assert await service.detokenize(other_tenant, protected.text) == protected.text
    unknown_text = TOKEN_RE.sub("<PII_PERSON:" + "0" * 32 + ">", protected.text)
    assert await service.detokenize(other_tenant, unknown_text) == unknown_text


async def test_unknown_and_malformed_tokens_are_not_resolved() -> None:
    repository = InMemoryVaultRepository()
    vault = make_vault(repository)
    caller = principal()
    unknown = "<PII_PERSON:" + "f" * 32 + ">"
    malformed = ["<PII_PERSON:xyz>", "<PII_NOT_A_TYPE:" + "a" * 32 + ">", "plain text"]

    assert await vault.resolve(caller, [unknown]) == {}
    calls_before = repository.calls
    assert await vault.resolve(caller, malformed) == {}
    assert repository.calls == calls_before, "malformed tokens must not reach storage"


async def test_expired_tokens_are_not_resolved_and_are_purged() -> None:
    clock = Clock()
    repository = InMemoryVaultRepository()
    vault = make_vault(repository, clock=clock)
    service = make_service(vault)
    caller = principal()
    protected = await service.tokenize(caller, TEXT)

    clock.advance(TTL)
    assert await service.detokenize(caller, protected.text) == protected.text
    assert await vault.purge_expired() == 2
    assert repository.rows == {}


async def test_store_requires_store_permission() -> None:
    repository = InMemoryVaultRepository()
    service = make_service(make_vault(repository))
    with pytest.raises(VaultAuthorizationError):
        await service.tokenize(principal(None, VaultPermission.DETOKENIZE), TEXT)
    assert repository.calls == 0


async def test_detokenize_requires_explicit_permission_before_any_storage_access() -> None:
    repository = InMemoryVaultRepository()
    service = make_service(make_vault(repository))
    owner = principal()
    protected = await service.tokenize(owner, TEXT)
    calls_before = repository.calls

    store_only = principal(owner.tenant_id, VaultPermission.STORE)
    with pytest.raises(VaultAuthorizationError):
        await service.detokenize(store_only, protected.text)
    assert repository.calls == calls_before


async def test_duplicate_token_is_rejected_without_overwriting() -> None:
    repository = InMemoryVaultRepository()
    vault = make_vault(repository)
    caller = principal()
    token = "<PII_PERSON:" + "1" * 32 + ">"
    original = TokenMapping(token=token, entity_type=EntityType.PERSON, value=PERSON)
    hijack = TokenMapping(token=token, entity_type=EntityType.PERSON, value="Ysolde Marrowick")

    await vault.store(caller, {token: original})
    with pytest.raises(TokenCollisionError):
        await vault.store(caller, {token: hijack})
    assert (await vault.resolve(caller, [token]))[token].value == PERSON


async def test_collision_retries_with_fresh_tokens() -> None:
    class CollidingOnce(InMemoryVaultRepository):
        failures = 1

        async def insert_many(self, records: Sequence[VaultRecord]) -> None:
            if self.failures:
                self.failures -= 1
                raise TokenCollisionError()
            await super().insert_many(records)

    repository = CollidingOnce()
    service = make_service(make_vault(repository))
    caller = principal()
    protected = await service.tokenize(caller, TEXT)
    assert await service.detokenize(caller, protected.text) == TEXT


async def test_persistent_collisions_fail_closed() -> None:
    class AlwaysColliding(InMemoryVaultRepository):
        async def insert_many(self, records: Sequence[VaultRecord]) -> None:
            raise TokenCollisionError()

    service = make_service(make_vault(AlwaysColliding()))
    with pytest.raises(TokenCollisionError):
        await service.tokenize(principal(), TEXT)


@pytest.mark.parametrize("field", ["ciphertext", "nonce", "wrapped_dek"])
async def test_tampered_record_aborts_whole_detokenization(field: str) -> None:
    repository = InMemoryVaultRepository()
    service = make_service(make_vault(repository))
    caller = principal()
    protected = await service.tokenize(caller, TEXT)
    first_key = next(iter(repository.rows))
    record = repository.rows[first_key]
    blob = getattr(record.encrypted, field)
    tampered_blob = blob[:-1] + bytes([blob[-1] ^ 0xFF])
    repository.rows[first_key] = replace(
        record, encrypted=replace(record.encrypted, **{field: tampered_blob})
    )

    with pytest.raises(VaultIntegrityError):
        await service.detokenize(caller, protected.text)


async def test_record_swapped_between_tokens_fails_integrity() -> None:
    repository = InMemoryVaultRepository()
    service = make_service(make_vault(repository))
    caller = principal()
    text = "Quillon Vantaberg and Ysolde Marrowick"
    protected = await service.tokenize(caller, text)
    (key_a, rec_a), (key_b, rec_b) = list(repository.rows.items())
    repository.rows[key_a] = replace(rec_a, encrypted=rec_b.encrypted)
    repository.rows[key_b] = replace(rec_b, encrypted=rec_a.encrypted)

    with pytest.raises(VaultIntegrityError):
        await service.detokenize(caller, protected.text)


async def test_record_relabelled_with_other_entity_type_fails_integrity() -> None:
    repository = InMemoryVaultRepository()
    vault = make_vault(repository)
    caller = principal()
    token = "<PII_PERSON:" + "2" * 32 + ">"
    await vault.store(
        caller, {token: TokenMapping(token=token, entity_type=EntityType.PERSON, value=PERSON)}
    )
    key = next(iter(repository.rows))
    repository.rows[key] = replace(repository.rows[key], entity_type=EntityType.LOCATION)
    with pytest.raises(VaultIntegrityError):
        await vault.resolve(caller, [token])


async def test_wrong_master_key_fails_closed() -> None:
    repository = InMemoryVaultRepository()
    hash_key = new_key()
    caller = principal()
    protected = await make_service(make_vault(repository, hash_key=hash_key)).tokenize(caller, TEXT)
    rekeyed = make_service(make_vault(repository, hash_key=hash_key, master_key=new_key()))
    with pytest.raises(VaultIntegrityError):
        await rekeyed.detokenize(caller, protected.text)


async def test_storage_failure_never_falls_back_to_plaintext() -> None:
    class Broken(InMemoryVaultRepository):
        async def insert_many(self, records: Sequence[VaultRecord]) -> None:
            raise RuntimeError(f"synthetic failure {EMAIL}")

        async def fetch_active(self, *args: object) -> list[VaultRecord]:
            raise ConnectionError("synthetic outage")

    service = make_service(make_vault(Broken()))
    caller = principal()
    with pytest.raises(VaultUnavailableError) as stored:
        await service.tokenize(caller, TEXT)
    assert EMAIL not in str(stored.value)
    assert stored.value.__suppress_context__

    tokenized = "Contact <PII_PERSON:" + "3" * 32 + ">."
    with pytest.raises(VaultUnavailableError):
        await service.detokenize(caller, tokenized)


async def test_malformed_mapping_is_rejected() -> None:
    vault = make_vault(InMemoryVaultRepository())
    token = "<PII_PERSON:" + "4" * 32 + ">"
    mislabelled = TokenMapping(token=token, entity_type=EntityType.EMAIL_ADDRESS, value=EMAIL)
    with pytest.raises(VaultError):
        await vault.store(principal(), {token: mislabelled})


async def test_request_size_is_bounded() -> None:
    vault = make_vault(InMemoryVaultRepository())
    tokens = [f"<PII_PERSON:{index:032x}>" for index in range(1_001)]
    with pytest.raises(VaultRequestTooLargeError):
        await vault.resolve(principal(), tokens)


async def test_concurrent_store_and_resolve_across_tenants() -> None:
    repository = InMemoryVaultRepository()
    service = make_service(make_vault(repository))
    callers = [principal() for _ in range(20)]
    protected = await asyncio.gather(*(service.tokenize(caller, TEXT) for caller in callers))
    pairs = zip(callers, protected, strict=True)
    restored = await asyncio.gather(*(service.detokenize(c, item.text) for c, item in pairs))
    assert restored == [TEXT] * len(callers)
    assert len({item.text for item in protected}) == len(callers)


async def test_sensitive_values_never_appear_in_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(vault_service, "logger", production_style_logger(buffer))
    caplog.set_level(logging.DEBUG)
    master_key, hash_key = new_key(), new_key()
    repository = InMemoryVaultRepository()
    service = make_service(make_vault(repository, master_key=master_key, hash_key=hash_key))
    caller = principal()

    protected = await service.tokenize(caller, TEXT)
    await service.detokenize(caller, protected.text)
    with pytest.raises(VaultAuthorizationError):
        await service.detokenize(principal(caller.tenant_id, VaultPermission.STORE), protected.text)
    repository.tamper(ciphertext=b"\x00" * 32)
    with pytest.raises(VaultIntegrityError):
        await service.detokenize(caller, protected.text)

    output = buffer.getvalue() + caplog.text
    assert "vault.stored" in output and "vault.integrity_failure" in output
    forbidden = [EMAIL, PERSON, *find_tokens(protected.text)]
    forbidden += [master_key.hex(), hash_key.hex()]
    for record in repository.rows.values():
        forbidden += [record.token_hash.hex(), record.encrypted.wrapped_dek.hex()]
    for value in forbidden:
        assert value not in output


def test_mapping_reprs_do_not_expose_values() -> None:
    mapping = TokenMapping(
        token="<PII_PERSON:" + "5" * 32 + ">", entity_type=EntityType.PERSON, value=PERSON
    )
    result = TokenizationResult(text="x", mappings={mapping.token: mapping})
    assert PERSON not in repr(mapping) and PERSON not in repr(result)
