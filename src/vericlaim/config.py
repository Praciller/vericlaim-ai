from functools import lru_cache

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
    cerebras_api_key: str | None = None
    cerebras_enabled: bool = False
    cerebras_model: str = "gpt-oss-120b"
    groq_api_key: str | None = None
    groq_enabled: bool = False
    groq_model: str = "openai/gpt-oss-20b"
    openrouter_api_key: str | None = None
    openrouter_enabled: bool = False
    openrouter_model: str = "openrouter/free"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-flash-lite-latest"
    okmd_api_key: str | None = None
    okmd_enabled: bool = False
    okmd_model: str = "deepseek-v4-flash"
    thaillm_api_key: str | None = None
    thaillm_enabled: bool = False
    thaillm_model: str = "OpenThaiGPT-ThaiLLM-8B-Instruct-v7.2"
    allow_non_reproducible_openrouter: bool = False
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
