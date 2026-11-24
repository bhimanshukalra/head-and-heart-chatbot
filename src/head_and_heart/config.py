"""Application settings, loaded from environment variables and an optional .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings. Raises a validation error at startup if required values are missing."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Literal["groq"] = Field(default="groq", description="LLM backend.")
    groq_api_key: SecretStr = Field(..., description="Groq API key.")
    model_name: str = Field(
        default="llama-3.3-70b-versatile", description="Model used for the response agents."
    )
    classifier_model_name: str = Field(
        default="llama-3.1-8b-instant", description="Cheaper model used for classification."
    )
    request_timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings; fails fast if configuration is invalid."""
    return Settings()
