"""Encrypted, tenant-scoped persistent token vault (Phase 2)."""

from app.vault.errors import (
    TokenCollisionError,
    VaultAuthorizationError,
    VaultConfigurationError,
    VaultError,
    VaultIntegrityError,
    VaultKeyUnavailableError,
    VaultRequestTooLargeError,
    VaultUnavailableError,
)
from app.vault.models import VaultPermission, VaultPrincipal
from app.vault.service import ProtectedText, TokenVault, VaultPrivacyService, build_token_vault

__all__ = [
    "ProtectedText",
    "TokenCollisionError",
    "TokenVault",
    "VaultAuthorizationError",
    "VaultConfigurationError",
    "VaultError",
    "VaultIntegrityError",
    "VaultKeyUnavailableError",
    "VaultPermission",
    "VaultPrincipal",
    "VaultPrivacyService",
    "VaultRequestTooLargeError",
    "VaultUnavailableError",
    "build_token_vault",
]
