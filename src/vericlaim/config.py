from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = "sqlite:///./data/vericlaim.db"
    mock_provider_enabled: bool = True
    live_retrieval_enabled: bool = False
    default_provider: str = "mock"
    log_level: str = "INFO"
    cors_allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    gemini_enabled: bool = False
    cerebras_api_key: SecretStr | None = None
    cerebras_enabled: bool = False
    cerebras_model: str = "gpt-oss-120b"
    groq_api_key: SecretStr | None = None
    groq_enabled: bool = False
    groq_model: str = "openai/gpt-oss-20b"
    openrouter_api_key: SecretStr | None = None
    openrouter_enabled: bool = False
    openrouter_model: str = "openrouter/free"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-flash-lite-latest"
    okmd_api_key: SecretStr | None = None
    okmd_enabled: bool = False
    okmd_model: str = "deepseek-v4-flash"
    thaillm_api_key: SecretStr | None = None
    thaillm_enabled: bool = False
    thaillm_model: str = "OpenThaiGPT-ThaiLLM-8B-Instruct-v7.2"
    allow_non_reproducible_openrouter: bool = False
    max_atomic_claims_per_request: int = Field(default=8, ge=1, le=16)
    max_retrieval_queries_per_request: int = Field(default=16, ge=1, le=32)
    max_evidence_candidates_per_request: int = Field(default=32, ge=1, le=64)
    max_provider_calls_per_request: int = Field(default=8, ge=0, le=16)
    request_timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def secret_value(value: SecretStr | str | None) -> str | None:
    """Return a configured secret only at an outbound provider boundary."""
    if value is None:
        return None
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    return value or None
