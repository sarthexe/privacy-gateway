"""Typed application settings loaded from environment variables."""

from functools import lru_cache

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings so all application components share one config."""
    return Settings()