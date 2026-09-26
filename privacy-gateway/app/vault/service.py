"""Vault service: the only component that sees both tokens and plaintext values.

Authorization boundary: every operation takes a ``VaultPrincipal`` and checks its
permission before hashing, storage access, or decryption. The LLM/routing layer
receives only ``ProtectedText`` from ``VaultPrivacyService.tokenize`` and must
never be given a vault, principal, or ``TokenMapping``.
"""

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, TypeVar

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.models.privacy import TokenMapping
from app.reconstruction import Detokenizer, find_tokens, token_entity_type
from app.vault.crypto import EnvelopeCipher
from app.vault.errors import (
    TokenCollisionError,
    VaultAuthorizationError,
    VaultError,
    VaultIntegrityError,
    VaultRequestTooLargeError,
    VaultUnavailableError,
)
from app.vault.keys import TokenHasher, build_local_key_material
from app.vault.models import EncryptionContext, VaultPermission, VaultPrincipal, VaultRecord
from app.vault.repository import SqlAlchemyVaultRepository, VaultRepository

if TYPE_CHECKING:
    from app.tokenization import Tokenizer

# Log fields are limited to event names, tenant ids, counts, and key ids. Never
# log tokens, token hashes, values, ciphertext, or key material.
logger = structlog.get_logger(__name__)

DEFAULT_MAX_TOKENS_PER_CALL = 1_000

T = TypeVar("T")


def _utc_now() -> datetime:
    return datetime.now(UTC)


class TokenVault:
    """Tenant-scoped, encrypted persistence for token mappings."""

    def __init__(
        self,
        repository: VaultRepository,
        cipher: EnvelopeCipher,
        hasher: TokenHasher,
        *,
        ttl: timedelta,
        clock: Callable[[], datetime] = _utc_now,
        max_tokens_per_call: int = DEFAULT_MAX_TOKENS_PER_CALL,
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError("Vault TTL must be positive")
        self._repository = repository
        self._cipher = cipher
        self._hasher = hasher
        self._ttl = ttl
        self._clock = clock
        self._max_tokens = max_tokens_per_call

    def authorize(self, principal: VaultPrincipal, permission: VaultPermission) -> None:
        """Raise ``VaultAuthorizationError`` unless ``principal`` holds ``permission``."""
        try:
            principal.require(permission)
        except VaultAuthorizationError:
            logger.warning(
                "vault.authorization_denied",
                tenant_id=str(principal.tenant_id),
                permission=permission.value,
            )
            raise

    async def store(self, principal: VaultPrincipal, mappings: Mapping[str, TokenMapping]) -> None:
        """Encrypt and atomically persist every mapping, or raise and persist none."""
        self.authorize(principal, VaultPermission.STORE)
        self._check_size(len(mappings))
        created_at = self._clock()
        expires_at = created_at + self._ttl
        records: list[VaultRecord] = []
        for token, mapping in mappings.items():
            if mapping.token != token or token_entity_type(token) != mapping.entity_type:
                raise VaultError("Token mapping is malformed.")
            token_hash = await self._hasher.hash_token(principal.tenant_id, token)
            context = EncryptionContext(principal.tenant_id, token_hash, mapping.entity_type)
            records.append(
                VaultRecord(
                    tenant_id=principal.tenant_id,
                    token_hash=token_hash,
                    entity_type=mapping.entity_type,
                    encrypted=await self._cipher.encrypt(mapping.value, context),
                    created_at=created_at,
                    expires_at=expires_at,
                )
            )
        await self._call_repository(self._repository.insert_many(records))
        logger.info("vault.stored", tenant_id=str(principal.tenant_id), count=len(records))

    async def resolve(
        self,
        principal: VaultPrincipal,
        tokens: Iterable[str],
    ) -> dict[str, TokenMapping]:
        """Return mappings for known, unexpired tokens owned by the principal's tenant.

        Unknown, expired, malformed, and other-tenant tokens are indistinguishable:
        they are simply absent from the result. Any integrity or key failure aborts
        the whole call so no partial plaintext is returned.
        """
        self.authorize(principal, VaultPermission.DETOKENIZE)
        candidates = {
            token: entity_type
            for token in dict.fromkeys(tokens)
            if (entity_type := token_entity_type(token)) is not None
        }
        self._check_size(len(candidates))
        if not candidates:
            return {}

        tenant_id = principal.tenant_id
        by_hash = {await self._hasher.hash_token(tenant_id, token): token for token in candidates}
        records = await self._call_repository(
            self._repository.fetch_active(tenant_id, by_hash.keys(), self._clock())
        )

        mappings: dict[str, TokenMapping] = {}
        for record in records:
            token = by_hash.get(record.token_hash)
            try:
                if (
                    token is None
                    or record.tenant_id != tenant_id
                    or record.entity_type != candidates[token]
                    or record.expires_at <= self._clock()
                ):
                    raise VaultIntegrityError()
                context = EncryptionContext(
                    tenant_id, record.token_hash, record.entity_type, record.format_version
                )
                value = await self._cipher.decrypt(record.encrypted, context)
            except VaultError as error:
                logger.error(
                    "vault.integrity_failure",
                    tenant_id=str(tenant_id),
                    key_id=record.encrypted.key_id,
                    reason=type(error).__name__,
                )
                raise
            mappings[token] = TokenMapping(token=token, entity_type=record.entity_type, value=value)

        logger.info(
            "vault.resolved",
            tenant_id=str(tenant_id),
            requested=len(candidates),
            resolved=len(mappings),
        )
        return mappings

    async def purge_expired(self) -> int:
        """Hard-delete expired records. Backups may still retain them."""
        deleted = await self._call_repository(self._repository.delete_expired(self._clock()))
        logger.info("vault.purged", count=deleted)
        return deleted

    def _check_size(self, count: int) -> None:
        if count > self._max_tokens:
            raise VaultRequestTooLargeError()

    @staticmethod
    async def _call_repository(operation: Awaitable[T]) -> T:
        """Fail closed: any storage failure surfaces as a vault error without details."""
        try:
            return await operation
        except VaultError:
            raise
        except Exception:
            logger.error("vault.storage_failure")
            raise VaultUnavailableError() from None


@dataclass(frozen=True, slots=True)
class ProtectedText:
    """Tokenized text that is safe to hand to the LLM layer. Holds no mappings."""

    text: str
    token_count: int


class VaultPrivacyService:
    """Compose Phase 1 tokenization/detokenization with the persistent vault."""

    def __init__(
        self,
        tokenizer: "Tokenizer",
        vault: TokenVault,
        detokenizer: Detokenizer | None = None,
        *,
        max_collision_retries: int = 2,
    ) -> None:
        self._tokenizer = tokenizer
        self._vault = vault
        self._detokenizer = detokenizer or Detokenizer()
        self._max_collision_retries = max_collision_retries

    async def tokenize(self, principal: VaultPrincipal, text: str) -> ProtectedText:
        """Tokenize ``text`` and persist its mappings; raise rather than return raw text."""
        self._vault.authorize(principal, VaultPermission.STORE)
        for attempt in range(self._max_collision_retries + 1):
            result = self._tokenizer.tokenize(text)
            try:
                await self._vault.store(principal, result.mappings)
            except TokenCollisionError:
                logger.warning("vault.token_collision", attempt=attempt + 1)
                if attempt == self._max_collision_retries:
                    raise
                continue
            return ProtectedText(text=result.text, token_count=len(result.mappings))
        raise TokenCollisionError()  # pragma: no cover - loop always returns or raises

    async def detokenize(self, principal: VaultPrincipal, text: str) -> str:
        """Restore tokens the principal may access; leave all other text unchanged."""
        mappings = await self._vault.resolve(principal, find_tokens(text))
        return self._detokenizer.detokenize(text, mappings)


def build_token_vault(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
) -> TokenVault:
    """Build a vault with PROTOTYPE environment-variable keys (see ``app.vault.keys``)."""
    key_provider, hasher = build_local_key_material(settings)
    return TokenVault(
        SqlAlchemyVaultRepository(session_factory),
        EnvelopeCipher(key_provider),
        hasher,
        ttl=timedelta(seconds=settings.vault_token_ttl_seconds),
    )
