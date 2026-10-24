"""Offline tests for the read-only hosted release preflight (no external requests)."""

from __future__ import annotations

import pytest
from scripts.preflight_hosted_release import (
    PRODUCTION_API,
    PRODUCTION_WEB,
    Reply,
    run_checks,
    validate_url,
)


def responses(mode: str) -> dict[tuple[str, str], Reply]:
    demo = mode == "demo"
    api = PRODUCTION_API
    web = PRODUCTION_WEB
    mock = {"name": "mock", "configured": True, "enabled": demo}
    gemini = {"name": "gemini", "configured": True, "enabled": not demo}
    headers = {"WWW-Authenticate": 'Basic realm="VeriClaim AI Owner", charset="UTF-8"'}
    return {
        (api, "/health"): Reply(200, {"status": "ok"}, {}),
        (api, "/ready"): Reply(200, {"status": "ready", "database": "ok"}, {}),
        (api, "/api/v1/capabilities"): Reply(200, {"demo_only": demo}, {}),
        (api, "/api/v1/providers/status"): Reply(200, [mock, gemini], {}),
        (api, "/api/v1/runs/run-preflight-unauthorized-00000000"): Reply(
            401 if not demo else 404, None, {}
        ),
        (web, "/"): Reply(200 if demo else 401, None, {} if demo else headers),
        (web, "/api/backend/api/v1/capabilities"): Reply(
            200 if demo else 401,
            {"demo_only": demo} if demo else None,
            {} if demo else headers,
        ),
    }


def test_only_approved_https_deployment_origins() -> None:
    assert validate_url(PRODUCTION_API, "api") == PRODUCTION_API
    assert validate_url(PRODUCTION_WEB + "/", "web") == PRODUCTION_WEB
    assert validate_url(
        "https://vericlaim-a9fiffiu0-pracillatlove-3370s-projects.vercel.app", "web"
    ).startswith("https://vericlaim-")
    for url in (
        "http://vericlaim-api.vercel.app",
        "https://example.com",
        "https://vericlaim-api.vercel.app.evil.com",
        "https://evil:secret@vericlaim-api.vercel.app",
        "https://vericlaim-api.vercel.app/?x=1",
        "https://vericlaim-api.vercel.app/secret",
        "https://vericlaim-api.vercel.app#fragment",
        "https://vericlaim-api.vercel.app:444",
        "https://vericlaim-other-team.vercel.app",
    ):
        with pytest.raises(ValueError):
            validate_url(url, "api")


def test_read_only_demo_gate_passes_without_inference_calls() -> None:
    fixture = responses("demo")
    visited: list[str] = []

    def fetch(base: str, path: str) -> Reply:
        assert path not in ("/api/v1/claims/verify",)
        visited.append(path)
        return fixture[(base, path)]

    checks = run_checks(PRODUCTION_API, PRODUCTION_WEB, "demo", fetch=fetch)
    assert len(checks) == 7
    assert all(item.passed for item in checks)
    assert len(visited) == 6
    assert all(item not in visited for item in ("/api/v1/claims/verify",))


def test_private_mode_requires_bearer_denial_and_owner_basic_challenge() -> None:
    fixture = responses("private")

    def fetch(base: str, path: str) -> Reply:
        return fixture[(base, path)]

    checks = run_checks(PRODUCTION_API, PRODUCTION_WEB, "private", fetch=fetch)
    assert len(checks) == 9
    assert all(item.passed for item in checks)


def test_private_gate_fails_if_web_challenge_is_not_ours() -> None:
    fixture = responses("private")
    fixture[(PRODUCTION_WEB, "/")] = Reply(401, None, {"WWW-Authenticate": "Bearer"})
    fixture[(PRODUCTION_API, "/api/v1/runs/run-preflight-unauthorized-00000000")] = Reply(
        404, None, {}
    )
    checks = run_checks(
        PRODUCTION_API,
        PRODUCTION_WEB,
        "private",
        fetch=lambda base, path: fixture[(base, path)],
    )
    failures = {item.name for item in checks if not item.passed}
    assert failures == {"owner_home_denied", "direct_api_denied"}


def test_demo_gate_catches_unexpected_live_mode() -> None:
    fixture = responses("private")
    checks = run_checks(
        PRODUCTION_API,
        PRODUCTION_WEB,
        "demo",
        fetch=lambda base, path: fixture[(base, path)],
    )
    assert any(not item.passed for item in checks)


def test_network_failure_is_fail_closed() -> None:
    checks = run_checks(
        PRODUCTION_API,
        PRODUCTION_WEB,
        "demo",
        fetch=lambda base, path: (_ for _ in ()).throw(OSError("network")),
    )
    assert not all(item.passed for item in checks)
