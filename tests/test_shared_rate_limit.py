"""Shared-rate-limit security tests: atomic counters, keyed identities and fail-closed APIs."""

import re
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

import vericlaim.api as api_module
from vericlaim.config import Settings
from vericlaim.db import Database, RateLimitBucketRow
from vericlaim.rate_limit import DatabaseRateLimiter

SECRET = "test-only-secret-value-that-is-very-long-123456789"


def _db(tmp_path):
    db = Database(f"sqlite:///{(tmp_path / 'quota.sqlite3').as_posix()}")
    db.init()
    return db


def test_shared_limiter_requires_secret_and_valid_dialect(tmp_path):
    db = _db(tmp_path)
    with pytest.raises(ValueError, match="32"):
        DatabaseRateLimiter(db.engine, max_requests=1, peer_secret="short")
    with pytest.raises(ValueError):
        DatabaseRateLimiter(db.engine, max_requests=0, peer_secret=SECRET)
    with pytest.raises(ValidationError):
        Settings(rate_limit_backend="database")
    with pytest.raises(ValidationError):
        Settings(rate_limit_backend="database", rate_limit_peer_secret="too-short")
    settings = Settings(rate_limit_backend="database", rate_limit_peer_secret=SECRET)
    assert settings.rate_limit_peer_secret is not None
    assert SECRET not in repr(settings)


def test_database_quota_is_shared_across_separate_instances(tmp_path):
    db = _db(tmp_path)
    now = [125.5]
    limiter_a = DatabaseRateLimiter(
        db.engine, max_requests=2, peer_secret=SECRET, clock=lambda: now[0]
    )
    independent = Database(db.engine.url.render_as_string(hide_password=False))
    limiter_b = DatabaseRateLimiter(
        independent.engine, max_requests=2, peer_secret=SECRET, clock=lambda: now[0]
    )
    peer = "203.0.113.15"
    assert limiter_a.allow(peer) == (True, 0)
    assert limiter_b.allow(peer) == (True, 0)
    assert limiter_a.allow(peer) == (False, 55)
    assert limiter_b.allow("198.51.100.27") == (True, 0)

    with Session(db.engine) as session:
        rows = session.scalars(select(RateLimitBucketRow)).all()
        assert len(rows) == 2
        assert all(re.fullmatch("[a-f0-9]{64}", x.peer_hash) for x in rows)
        assert all(x.peer_hash != peer for x in rows)
        assert any(x.hits == 2 for x in rows)

    now[0] = 180.0
    assert limiter_b.allow(peer) == (True, 0)
    now[0] = 305.0
    assert limiter_a.allow(peer) == (True, 0)
    with Session(db.engine) as session:
        assert all(x.window_index >= 3 for x in session.scalars(select(RateLimitBucketRow)))


def test_shared_limiter_prevents_concurrent_oversubscription(tmp_path):
    db = _db(tmp_path)
    first = DatabaseRateLimiter(db.engine, max_requests=3, peer_secret=SECRET)
    second = DatabaseRateLimiter(db.engine, max_requests=3, peer_secret=SECRET)
    limiters = [first, second] * 10

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(lambda limiter: limiter.allow("203.0.113.24"), limiters))
    assert sum(allowed for allowed, _ in answers) == 3
    assert all(wait >= 1 for allowed, wait in answers if not allowed)


def test_shared_limiter_rejects_spoofed_forwarded_peer(monkeypatch, tmp_path):
    db = _db(tmp_path)
    limiter = DatabaseRateLimiter(db.engine, max_requests=1, peer_secret=SECRET)
    monkeypatch.setattr(api_module, "verify_rate_limiter", limiter)
    monkeypatch.setattr(api_module, "database", db)
    with TestClient(api_module.app) as client:
        first = client.post(
            "/api/v1/claims/verify",
            json={"claim": "RAG improves outcomes"},
            headers={"X-Forwarded-For": "198.51.100.10"},
        )
        second = client.post(
            "/api/v1/claims/verify",
            json={"claim": "RAG improves outcomes"},
            headers={"X-Forwarded-For": "198.51.100.11"},
        )
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"]["code"] == "RATE_LIMITED"
    assert int(second.headers["Retry-After"]) > 0


def test_shared_limiter_fails_closed_without_leaking_sql_errors(monkeypatch):
    class BrokenLimiter:
        def allow(self, peer):
            raise RuntimeError("secret-db-connect-string")

    monkeypatch.setattr(api_module, "verify_rate_limiter", BrokenLimiter())
    with TestClient(api_module.app) as client:
        response = client.post(
            "/api/v1/claims/verify",
            json={"claim": "RAG improves outcomes"},
        )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "RATE_LIMIT_UNAVAILABLE"
    assert "secret-db-connect-string" not in response.text
