"""Internal token guard and fail-closed production deployment configuration tests."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import vericlaim.api as api_module
from vericlaim.config import Settings
from vericlaim.db import Database
from vericlaim.rate_limit import InMemoryRateLimiter

TOKEN = "test-only-private-api-bearer-secret-1234567890123456789"
CLAIM = "RAG eliminates hallucinations."


def test_production_requires_fixture_only_or_strong_server_token():
    with pytest.raises(ValidationError, match="API_ACCESS_TOKEN"):
        Settings(app_env="production", public_demo_only=False, api_access_token=None)
    with pytest.raises(ValidationError, match="API_ACCESS_TOKEN"):
        Settings(app_env="staging", public_demo_only=False, api_access_token="short")
    settings = Settings(app_env="production", public_demo_only=False, api_access_token=TOKEN)
    assert settings.api_access_token is not None
    assert TOKEN not in repr(settings)
    assert Settings(app_env="production", public_demo_only=True).public_demo_only
    assert Settings(app_env="development", public_demo_only=False).app_env == "development"


@pytest.mark.parametrize(
    "override",
    [
        {"live_retrieval_enabled": True},
        {"mock_provider_enabled": False},
        {"default_provider": "groq"},
        {"groq_enabled": True},
        {"gemini_enabled": True},
        {"openrouter_enabled": True},
        {"cerebras_enabled": True},
        {"okmd_enabled": True},
        {"thaillm_enabled": True},
    ],
)
def test_demo_mode_cannot_accidentally_enable_live_providers(override):
    with pytest.raises(ValidationError, match="Public demo"):
        Settings(public_demo_only=True, **override)


def test_api_endpoints_reject_missing_and_invalid_tokens(monkeypatch, tmp_path):
    db = Database(f"sqlite:///{(tmp_path / 'auth.sqlite3').as_posix()}")
    monkeypatch.setattr(
        api_module,
        "settings",
        Settings(app_env="production", public_demo_only=False, api_access_token=TOKEN),
    )
    monkeypatch.setattr(api_module, "database", db)
    monkeypatch.setattr(
        api_module,
        "verify_rate_limiter",
        InMemoryRateLimiter(max_requests=100),
    )
    with TestClient(api_module.app) as client:
        for headers in ({}, {"Authorization": "Bearer invalid"}, {"X-Api-Key": TOKEN}):
            response = client.post("/api/v1/claims/verify", json={"claim": CLAIM}, headers=headers)
            assert response.status_code == 401
            assert response.json()["detail"]["code"] == "UNAUTHORIZED"
            assert response.headers["WWW-Authenticate"] == "Bearer"
            assert TOKEN not in response.text
        authorized = {"Authorization": f"Bearer {TOKEN}"}
        created = client.post("/api/v1/claims/verify", json={"claim": CLAIM}, headers=authorized)
        assert created.status_code == 200
        run_id = created.json()["run_id"]
        for suffix in ("", "/evidence", "/evidence-graph"):
            path = f"/api/v1/runs/{run_id}{suffix}"
            for headers in ({}, {"Authorization": "Bearer wrong"}, {"X-Forwarded-For": TOKEN}):
                assert client.get(path, headers=headers).status_code == 401
            assert client.get(path, headers=authorized).status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/api/v1/capabilities").status_code == 200
        assert TOKEN not in client.get("/api/v1/capabilities").text


def test_demo_only_stays_anonymous_even_when_a_server_secret_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api_module,
        "settings",
        Settings(app_env="production", public_demo_only=True, api_access_token=TOKEN),
    )
    db = Database(f"sqlite:///{(tmp_path / 'demo.sqlite3').as_posix()}")
    monkeypatch.setattr(api_module, "database", db)
    monkeypatch.setattr(
        api_module,
        "verify_rate_limiter",
        InMemoryRateLimiter(max_requests=100),
    )
    with TestClient(api_module.app) as client:
        response = client.post("/api/v1/claims/verify", json={"claim": CLAIM})
        assert response.status_code == 200
        assert client.get(f"/api/v1/runs/{response.json()['run_id']}").status_code == 200


def test_development_no_token_preserves_existing_unrestricted_local_flow(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api_module, "settings", Settings(app_env="development", api_access_token=None)
    )
    db = Database(f"sqlite:///{(tmp_path / 'dev.sqlite3').as_posix()}")
    monkeypatch.setattr(api_module, "database", db)
    monkeypatch.setattr(
        api_module,
        "verify_rate_limiter",
        InMemoryRateLimiter(max_requests=100),
    )
    with TestClient(api_module.app) as client:
        response = client.post("/api/v1/claims/verify", json={"claim": CLAIM})
        assert response.status_code == 200
