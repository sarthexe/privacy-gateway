"""Vault tests against a real PostgreSQL database.

Set ``GATEWAY_TEST_DATABASE_URL`` (an empty, disposable database) to run them, e.g.
``postgresql+asyncpg://postgres@localhost:5432/vault_test``. Tables in that
database are dropped and recreated.
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.db.base import Base
from app.db.session import build_engine
from app.reconstruction import find_tokens
from app.vault.errors import TokenCollisionError, VaultIntegrityError, VaultUnavailableError
from app.vault.keys import HmacTokenHasher
from app.vault.orm import VaultEntryRow
from app.vault.repository import SqlAlchemyVaultRepository
from tests.vault_support import (
    EMAIL,
    PERSON,
    TEXT,
    TTL,
    Clock,
    make_service,
    make_vault,
    new_key,
    principal,
)

DATABASE_URL = os.environ.get("GATEWAY_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="GATEWAY_TEST_DATABASE_URL is not set; PostgreSQL tests skipped"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SessionFactory = async_sessionmaker[AsyncSession]


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[SessionFactory]:
    assert DATABASE_URL is not None
    engine = build_engine(DATABASE_URL)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@dataclass(frozen=True)
class Keys:
    master_key: bytes
    hash_key: bytes


@pytest.fixture
def keys() -> Keys:
    return Keys(master_key=new_key(), hash_key=new_key())


def repository(factory: SessionFactory) -> SqlAlchemyVaultRepository:
    return SqlAlchemyVaultRepository(factory)


async def raw_rows(factory: SessionFactory) -> list[VaultEntryRow]:
    async with factory() as session:
        return list((await session.scalars(select(VaultEntryRow))).all())


async def test_round_trip_through_postgres(session_factory: SessionFactory) -> None:
    service = make_service(make_vault(repository(session_factory)))
    caller = principal()
    protected = await service.tokenize(caller, TEXT)
    assert await service.detokenize(caller, protected.text) == TEXT


async def test_database_rows_contain_no_plaintext_or_raw_tokens(
    session_factory: SessionFactory,
) -> None:
    service = make_service(make_vault(repository(session_factory)))
    protected = await service.tokenize(principal(), TEXT)
    tokens = find_tokens(protected.text)

    async with session_factory() as session:
        dump = (await session.execute(text("SELECT vault_entries::text FROM vault_entries"))).all()
    rows = await raw_rows(session_factory)
    assert len(rows) == 2 and len(dump) == 2
    for secret in (EMAIL, PERSON, *tokens):
        assert all(secret not in row[0] for row in dump)
        assert all(secret.encode().hex() not in row[0] for row in dump)
    for row in rows:
        assert row.ciphertext != EMAIL.encode() and row.ciphertext != PERSON.encode()
        assert len(row.token_hash) == 32 and len(row.nonce) == 12


@pytest.mark.parametrize("column", ["ciphertext", "nonce", "wrapped_dek"])
async def test_ciphertext_modified_in_database_fails_integrity(
    session_factory: SessionFactory, column: str
) -> None:
    service = make_service(make_vault(repository(session_factory)))
    caller = principal()
    protected = await service.tokenize(caller, TEXT)
    async with session_factory() as session, session.begin():
        await session.execute(
            text(
                f"UPDATE vault_entries SET {column} = "
                f"overlay({column} placing '\\xff'::bytea from 1 for 1)"
            )
        )
    with pytest.raises(VaultIntegrityError):
        await service.detokenize(caller, protected.text)


async def test_row_copied_to_another_tenant_fails_integrity(
    session_factory: SessionFactory, keys: Keys
) -> None:
    repo = repository(session_factory)
    service = make_service(make_vault(repo, master_key=keys.master_key, hash_key=keys.hash_key))
    victim, attacker = principal(), principal()
    protected = await service.tokenize(victim, "Quillon Vantaberg")
    (token,) = find_tokens(protected.text)

    # Worst case: an attacker with DB write access AND the hash key re-keys the row.
    forged_hash = await HmacTokenHasher(keys.hash_key).hash_token(attacker.tenant_id, token)
    (row,) = await raw_rows(session_factory)
    async with session_factory() as session, session.begin():
        session.add(
            VaultEntryRow(
                **{
                    column.key: getattr(row, column.key)
                    for column in VaultEntryRow.__table__.columns
                    if column.key not in {"tenant_id", "token_hash"}
                },
                tenant_id=attacker.tenant_id,
                token_hash=forged_hash,
            )
        )
    with pytest.raises(VaultIntegrityError):
        await service.detokenize(attacker, protected.text)
    assert await service.detokenize(victim, protected.text) == "Quillon Vantaberg"


async def test_wrong_tenant_and_unknown_token_look_identical(
    session_factory: SessionFactory,
) -> None:
    vault = make_vault(repository(session_factory))
    service = make_service(vault)
    protected = await service.tokenize(principal(), TEXT)
    other = principal()
    tokens = find_tokens(protected.text)
    unknown = ["<PII_PERSON:" + "0" * 32 + ">"]
    assert await vault.resolve(other, tokens) == {} == await vault.resolve(other, unknown)
    assert await service.detokenize(other, protected.text) == protected.text


async def test_wrong_key_fails_closed(session_factory: SessionFactory, keys: Keys) -> None:
    repo = repository(session_factory)
    caller = principal()
    protected = await make_service(
        make_vault(repo, master_key=keys.master_key, hash_key=keys.hash_key)
    ).tokenize(caller, TEXT)
    wrong = make_service(make_vault(repo, hash_key=keys.hash_key, master_key=new_key()))
    with pytest.raises(VaultIntegrityError):
        await wrong.detokenize(caller, protected.text)


async def test_expired_tokens_are_hidden_then_purged(session_factory: SessionFactory) -> None:
    clock = Clock()
    vault = make_vault(repository(session_factory), clock=clock)
    service = make_service(vault)
    caller = principal()
    protected = await service.tokenize(caller, TEXT)

    clock.advance(TTL - timedelta(microseconds=1))
    assert await service.detokenize(caller, protected.text) == TEXT
    clock.advance(timedelta(microseconds=1))
    assert await service.detokenize(caller, protected.text) == protected.text
    assert await vault.purge_expired() == 2
    assert await raw_rows(session_factory) == []


async def test_duplicate_token_insert_is_rejected_and_original_kept(
    session_factory: SessionFactory, keys: Keys
) -> None:
    repo = repository(session_factory)
    vault = make_vault(repo, master_key=keys.master_key, hash_key=keys.hash_key)
    caller = principal()
    protected = await make_service(vault).tokenize(caller, "Quillon Vantaberg")
    (row,) = await raw_rows(session_factory)
    (record,) = await repo.fetch_active(caller.tenant_id, [row.token_hash], Clock().now)

    forged = replace(record, encrypted=replace(record.encrypted, ciphertext=b"\x00" * 32))
    with pytest.raises(TokenCollisionError) as error:
        await repo.insert_many([forged])
    assert error.value.__suppress_context__
    assert await make_service(vault).detokenize(caller, protected.text) == "Quillon Vantaberg"


async def test_batch_insert_is_atomic(session_factory: SessionFactory, keys: Keys) -> None:
    repo = repository(session_factory)
    vault = make_vault(repo, master_key=keys.master_key, hash_key=keys.hash_key)
    caller = principal()
    await make_service(vault).tokenize(caller, "Quillon Vantaberg")
    (existing,) = await repo.fetch_active(
        caller.tenant_id, [(await raw_rows(session_factory))[0].token_hash], Clock().now
    )
    fresh = replace(existing, token_hash=b"\x07" * 32)
    with pytest.raises(TokenCollisionError):
        await repo.insert_many([fresh, existing])
    assert len(await raw_rows(session_factory)) == 1


async def test_concurrent_identical_inserts_have_exactly_one_winner(
    session_factory: SessionFactory,
) -> None:
    repo = repository(session_factory)
    caller = principal()
    await make_service(make_vault(repo)).tokenize(caller, "Quillon Vantaberg")
    (row,) = await raw_rows(session_factory)
    (record,) = await repo.fetch_active(caller.tenant_id, [row.token_hash], Clock().now)
    async with session_factory() as session, session.begin():
        await session.execute(text("DELETE FROM vault_entries"))

    results = await asyncio.gather(
        *(repo.insert_many([record]) for _ in range(10)), return_exceptions=True
    )
    assert sum(result is None for result in results) == 1
    assert all(isinstance(r, TokenCollisionError) for r in results if r is not None)


async def test_concurrent_store_and_lookup(session_factory: SessionFactory) -> None:
    service = make_service(make_vault(repository(session_factory)))
    callers = [principal() for _ in range(25)]
    protected = await asyncio.gather(*(service.tokenize(caller, TEXT) for caller in callers))
    lookups = [
        service.detokenize(caller, item.text)
        for caller, item in zip(callers, protected, strict=True)
    ]
    writes = [service.tokenize(caller, TEXT) for caller in callers]
    results = await asyncio.gather(*lookups, *writes)
    assert results[: len(callers)] == [TEXT] * len(callers)
    assert len(await raw_rows(session_factory)) == 4 * len(callers)


async def test_sql_logs_and_errors_hide_parameters(
    session_factory: SessionFactory, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="sqlalchemy.engine")
    repo = repository(session_factory)
    service = make_service(make_vault(repo))
    caller = principal()
    protected = await service.tokenize(caller, TEXT)
    await service.detokenize(caller, protected.text)

    rows = await raw_rows(session_factory)
    assert "parameters hidden" in caplog.text
    for row in rows:
        assert row.token_hash.hex() not in caplog.text
        assert row.ciphertext.hex() not in caplog.text

    (record,) = await repo.fetch_active(caller.tenant_id, [rows[0].token_hash], Clock().now)
    invalid = replace(
        record, token_hash=b"\x09" * 32, encrypted=replace(record.encrypted, nonce=b"\x01" * 11)
    )
    with pytest.raises(VaultUnavailableError) as error:
        await repo.insert_many([invalid])
    assert str(error.value) == VaultUnavailableError.default_message


async def test_database_outage_fails_closed() -> None:
    engine = build_engine("postgresql+asyncpg://nobody@127.0.0.1:1/unreachable")
    service = make_service(make_vault(SqlAlchemyVaultRepository(async_sessionmaker(engine))))
    caller = principal()
    try:
        with pytest.raises(VaultUnavailableError):
            await service.tokenize(caller, TEXT)
        with pytest.raises(VaultUnavailableError):
            await service.detokenize(caller, "x <PII_PERSON:" + "a" * 32 + ">")
    finally:
        await engine.dispose()


def test_migration_upgrade_downgrade_and_no_model_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    assert DATABASE_URL is not None

    async def reset() -> None:
        engine = build_engine(DATABASE_URL)
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
            await connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
        await engine.dispose()

    async def table_names() -> list[str]:
        engine = build_engine(DATABASE_URL)
        async with engine.connect() as connection:
            names = await connection.run_sync(lambda sync: inspect(sync).get_table_names())
        await engine.dispose()
        return names

    asyncio.run(reset())
    monkeypatch.setenv("GATEWAY_DATABASE_URL", DATABASE_URL)
    get_settings.cache_clear()
    config = Config()
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    try:
        command.upgrade(config, "head")
        assert "vault_entries" in asyncio.run(table_names())
        command.check(config)
        command.downgrade(config, "base")
        assert "vault_entries" not in asyncio.run(table_names())
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()


async def test_orm_update_of_entity_type_is_detected(session_factory: SessionFactory) -> None:
    service = make_service(make_vault(repository(session_factory)))
    caller = principal()
    protected = await service.tokenize(caller, "Quillon Vantaberg")
    async with session_factory() as session, session.begin():
        await session.execute(update(VaultEntryRow).values(entity_type="LOCATION"))
    with pytest.raises(VaultIntegrityError):
        await service.detokenize(caller, protected.text)
