"""Typed application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration; secrets should be provided by the environment."""

    app_name: str = "Privacy Gateway"
    environment: str = "development"
    log_level: str = "INFO"
    database_url: str = (
        "postgresql+asyncpg://privacy_gateway:change-this-local-password"
        "@localhost:5432/privacy_gateway"
    )
    redis_url: str = "redis://localhost:6379/0"

    # Prototype vault keys. There are deliberately no defaults: the vault refuses
    # to start without explicitly provided keys. Format: "key_id:base64key,...".
    vault_master_keys: SecretStr | None = None
    vault_active_key_id: str | None = None
    vault_token_hash_key: SecretStr | None = None
    vault_token_ttl_seconds: int = Field(default=86_400, ge=60, le=30 * 86_400)

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings so all application components share one config."""
    return Settings()
