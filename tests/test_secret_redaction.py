from __future__ import annotations

import json
import logging
import traceback

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import vericlaim.api as api_module
from vericlaim.api import app
from vericlaim.config import Settings, secret_value
from vericlaim.domain.models import ProviderErrorCategory
from vericlaim.providers.base import (
    GeminiProvider,
    ProviderException,
    ProviderRequest,
)
from vericlaim.providers.router import ProviderRouter

GEMINI_SECRET = "test-gemini-secret-123"
GROQ_SECRET = "test-groq-secret-456"


class FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def json(self) -> object:
        return self._payload


def _rendered_settings(settings: Settings) -> str:
    return "\n".join(
        (
            str(settings),
            repr(settings),
            str(settings.model_dump()),
            json.dumps(settings.model_dump(mode="json"), sort_keys=True),
            settings.model_dump_json(),
        )
    )


def test_settings_redact_secrets_in_common_diagnostics() -> None:
    settings = Settings(
        _env_file=None,
        gemini_api_key=GEMINI_SECRET,
        groq_api_key=GROQ_SECRET,
        gemini_enabled=True,
        groq_enabled=True,
    )

    rendered = _rendered_settings(settings)

    assert GEMINI_SECRET not in rendered
    assert GROQ_SECRET not in rendered
    assert rendered.count("**********") >= 2
    assert secret_value(settings.gemini_api_key) == GEMINI_SECRET


def test_router_unwraps_secret_only_at_provider_boundary() -> None:
    settings = Settings(
        _env_file=None,
        mock_provider_enabled=False,
        groq_enabled=True,
        groq_api_key=GROQ_SECRET,
    )

    provider = ProviderRouter(settings).providers["groq"]

    assert provider.api_key == GROQ_SECRET  # type: ignore[attr-defined]
    assert GROQ_SECRET not in repr(provider)
    assert GROQ_SECRET not in _rendered_settings(settings)


@pytest.mark.parametrize("status_code", [401, 429, 504])
def test_http_provider_errors_do_not_expose_vendor_payload(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    monkeypatch.setattr(
        "vericlaim.providers.base.httpx.post",
        lambda *args, **kwargs: FakeResponse(  # noqa: ARG005
            {"error": {"message": f"vendor diagnostic contains {GEMINI_SECRET}"}},
            status_code=status_code,
        ),
    )

    with pytest.raises(ProviderException) as raised:
        GeminiProvider("gemini", "test-model", GEMINI_SECRET).generate(
            ProviderRequest("smoke", "Return exactly: OK", max_tokens=16)
        )

    rendered = "".join(traceback.format_exception(raised.value))
    assert raised.value.category in {
        ProviderErrorCategory.AUTHENTICATION,
        ProviderErrorCategory.RATE_LIMIT,
        ProviderErrorCategory.SERVER_ERROR,
    }
    assert GEMINI_SECRET not in rendered


def test_provider_timeout_suppresses_raw_exception_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(*args: object, **kwargs: object) -> None:  # noqa: ARG001
        raise httpx.ReadTimeout(f"upstream diagnostic contains {GEMINI_SECRET}")

    monkeypatch.setattr("vericlaim.providers.base.httpx.post", raise_timeout)

    with pytest.raises(ProviderException) as raised:
        GeminiProvider("gemini", "test-model", GEMINI_SECRET).generate(
            ProviderRequest("smoke", "Return exactly: OK", max_tokens=16)
        )

    assert raised.value.category == ProviderErrorCategory.TIMEOUT
    assert GEMINI_SECRET not in "".join(traceback.format_exception(raised.value))


def test_configuration_validation_error_does_not_include_configured_secret() -> None:
    with pytest.raises(ValidationError) as raised:
        Settings(
            _env_file=None,
            gemini_api_key=GEMINI_SECRET,
            request_timeout_seconds="not-a-number",
        )

    assert GEMINI_SECRET not in str(raised.value)


def test_structured_logging_of_settings_is_redacted(caplog: pytest.LogCaptureFixture) -> None:
    settings = Settings(_env_file=None, gemini_api_key=GEMINI_SECRET)

    with caplog.at_level(logging.ERROR):
        logging.getLogger("vericlaim.security-test").error(
            "settings=%s", settings.model_dump(mode="json")
        )

    assert GEMINI_SECRET not in caplog.text
    assert "**********" in caplog.text


def test_api_error_response_does_not_expose_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingWorkflow:
        def verify(self, request: object, *, timeout_seconds: float) -> object:  # noqa: ARG002
            raise RuntimeError(f"configuration failure contains {GEMINI_SECRET}")

    monkeypatch.setattr(api_module, "workflow", FailingWorkflow())

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/claims/verify", json={"claim": "RAG eliminates hallucinations"}
        )

    assert response.status_code == 500
    assert GEMINI_SECRET not in response.text
    assert "verification failed safely" in response.text


def test_readiness_failure_does_not_expose_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_database_check() -> None:
        raise RuntimeError(f"database diagnostic contains {GEMINI_SECRET}")

    monkeypatch.setattr(api_module.database, "check", fail_database_check)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert GEMINI_SECRET not in response.text
    assert response.json()["detail"]["checks"]["issues"] == ["database_unavailable"]
