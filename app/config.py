from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: SecretStr
    database_host: str | None = None
    database_port: int | None = None
    voyage_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-haiku-5-5"
    # Sent only when set; unset uses the model's default (medium on Claude Haiku 5.5).
    # Leave it unset for models without effort levels, such as Claude Haiku 4.5.
    anthropic_effort: Literal["low", "medium", "high"] | None = None
    retrieval_mode: Literal["vector", "hybrid"] = "vector"

    @field_validator("anthropic_effort", mode="before")
    @classmethod
    def blank_effort_is_unset(cls, value):
        return value or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
