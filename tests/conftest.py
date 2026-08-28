import os

import pytest

# Normal pytest is quota-free even when a developer's ignored .env enables live
# providers. Explicit smoke/integration commands opt in separately.
for _provider in ("GEMINI", "GROQ", "CEREBRAS", "OPENROUTER", "OKMD", "THAILLM"):
    os.environ[f"{_provider}_ENABLED"] = "false"

from vericlaim.config import Settings  # noqa: E402
from vericlaim.db import Database  # noqa: E402
from vericlaim.providers.router import ProviderRouter  # noqa: E402
from vericlaim.workflow import VerificationWorkflow  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    return Settings(database_url="sqlite:///:memory:", mock_provider_enabled=True)


@pytest.fixture
def workflow(settings: Settings) -> VerificationWorkflow:
    return VerificationWorkflow(settings, ProviderRouter(settings))


@pytest.fixture
def database() -> Database:
    db = Database("sqlite:///:memory:")
    db.init()
    return db
