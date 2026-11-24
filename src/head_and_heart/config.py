"""Application settings, loaded from environment variables and an optional .env file."""

from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
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
        default="openai/gpt-oss-120b", description="Model used for the response agents."
    )
    classifier_model_name: str = Field(
        default="openai/gpt-oss-20b", description="Cheaper model used for classification."
    )
    request_timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    max_context_tokens: int = Field(
        default=6000, gt=0, description="Approximate token budget for history sent to the model."
    )
    classifier_confidence_threshold: float = Field(
        default=0.6, ge=0, le=1, description="Below this, a message is treated as 'mixed'."
    )
    crisis_region: str = Field(default="US", description="Region code for crisis resources.")
    max_input_chars: int = Field(default=4000, gt=0, description="Longest accepted user message.")
    persist_history: bool = Field(
        default=False, description="Opt in to storing conversations in SQLite."
    )
    log_format: Literal["text", "json"] = "text"
    api_keys: SecretStr = Field(
        default=SecretStr(""),
        description="Comma-separated 'user:key' pairs accepted by the HTTP API.",
    )
    api_auth_disabled: bool = Field(
        default=False, description="Only for local development: disables API key auth."
    )
    rate_limit_per_minute: int = Field(default=20, gt=0)
    metrics_token: SecretStr = Field(
        default=SecretStr(""),
        description="If set, /metrics requires 'Authorization: Bearer <token>'.",
    )
    cors_allow_origins: str = Field(
        default="", description="Comma-separated browser origins allowed to call the API."
    )
    model_prices: str = Field(
        default="",
        description="USD per 1M tokens as 'model:input:output,...'. Empty disables cost metrics.",
    )
    retention_days: int = Field(
        default=0, ge=0, description="Delete conversations idle for this many days. 0 keeps all."
    )
    retention_check_interval_s: int = Field(default=3600, gt=0)
    checkpoint_encryption_key: SecretStr = Field(
        default=SecretStr(""),
        description="16, 24 or 32 byte key to encrypt stored conversations (AES). Empty = off.",
    )

    @field_validator("checkpoint_encryption_key")
    @classmethod
    def _validate_key(cls, key: SecretStr) -> SecretStr:
        return _check_key(key)

    checkpoint_db_path: str = Field(default="chat_history.sqlite", description="SQLite file.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


def _check_key(key: SecretStr) -> SecretStr:
    if key.get_secret_value() and len(key.get_secret_value().encode()) not in (16, 24, 32):
        raise ValueError("CHECKPOINT_ENCRYPTION_KEY must be 16, 24 or 32 bytes long.")
    return key


@lru_cache
def get_settings() -> Settings:
    """Return cached settings; fails fast if configuration is invalid."""
    load_dotenv()  # so libraries that read os.environ (e.g. LangSmith) see .env values
    return Settings()
