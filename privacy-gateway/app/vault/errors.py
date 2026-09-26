"""Vault error hierarchy.

Messages are fixed strings: they must never include tokens, token hashes, key
material, ciphertext, or plaintext values, because callers may log or surface them.
"""


class VaultError(Exception):
    """Base class for all vault failures; callers must treat these as fail-closed."""

    default_message = "Vault operation failed."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.default_message)


class VaultConfigurationError(VaultError):
    """Vault keys or settings are missing or invalid."""

    default_message = "Vault is not configured correctly."


class VaultAuthorizationError(VaultError):
    """The principal lacks the permission required for the operation."""

    default_message = "Vault operation is not permitted."


class VaultIntegrityError(VaultError):
    """Authenticated decryption or record consistency checks failed."""

    default_message = "Vault record failed integrity validation."


class VaultKeyUnavailableError(VaultError):
    """The key referenced by a record is not available to this process."""

    default_message = "Vault key is unavailable."


class TokenCollisionError(VaultError):
    """A token identifier already exists; existing records are never overwritten."""

    default_message = "Vault token already exists."


class VaultUnavailableError(VaultError):
    """The vault backing store could not complete the operation."""

    default_message = "Vault storage is unavailable."


class VaultRequestTooLargeError(VaultError):
    """A single call referenced more tokens than the vault allows."""

    default_message = "Too many vault tokens in one request."
