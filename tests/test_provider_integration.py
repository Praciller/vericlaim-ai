import os

import pytest

from vericlaim.config import Settings
from vericlaim.providers.base import ProviderRequest
from vericlaim.providers.router import ProviderRouter

pytestmark = pytest.mark.integration


def _live_provider(name: str):
    if os.getenv("RUN_LIVE_PROVIDER_TESTS", "").casefold() != "true":
        pytest.skip("set RUN_LIVE_PROVIDER_TESTS=true to spend provider quota")
    router = ProviderRouter(Settings())
    provider = router.providers.get(name)
    if provider is None:
        pytest.skip(f"{name} is not configured and enabled")
    return provider


@pytest.mark.parametrize("name", ["gemini", "groq", "okmd", "thaillm", "openrouter"])
def test_live_provider_returns_usable_content(name: str):
    response = _live_provider(name).generate(
        ProviderRequest("integration_smoke", "Reply with OK only.", max_tokens=128)
    )
    assert response.text.strip()
