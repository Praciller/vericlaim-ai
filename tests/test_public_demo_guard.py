"""Public demo mode rejects arbitrary data before retrieval or persistence."""

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import vericlaim.api as api_module
from vericlaim.config import Settings
from vericlaim.db import Database, VerificationRunRow
from vericlaim.rate_limit import InMemoryRateLimiter

FIXTURE = "RAG eliminates hallucinations."


def _configure_demo(monkeypatch, tmp_path):
    db = Database(f"sqlite:///{(tmp_path / 'demo.sqlite3').as_posix()}")
    monkeypatch.setattr(api_module, "settings", Settings(public_demo_only=True))
    monkeypatch.setattr(api_module, "database", db)
    monkeypatch.setattr(api_module, "verify_rate_limiter", InMemoryRateLimiter(max_requests=200))
    return db


def test_capabilities_expose_mode_but_no_internal_secrets(monkeypatch, tmp_path):
    _configure_demo(monkeypatch, tmp_path)
    with TestClient(api_module.app) as client:
        response = client.get("/api/v1/capabilities")
        assert response.status_code == 200
        assert response.json() == {"demo_only": True, "demo_claim": FIXTURE}
        assert "api_key" not in response.text
        assert client.get("/health").status_code == 200


def test_public_demo_rejects_arbitrary_claim_before_workflow_or_persistence(monkeypatch, tmp_path):
    db = _configure_demo(monkeypatch, tmp_path)

    def forbidden_workflow(*args, **kwargs):
        raise AssertionError("workflow should not be called in demo-only mode")

    monkeypatch.setattr(api_module.workflow, "verify", forbidden_workflow)
    with TestClient(api_module.app) as client:
        for text in (
            "Unpublished project source code is X",
            "RAG eliminates hallucinations",
            "PRIVATE",
        ):
            response = client.post("/api/v1/claims/verify", json={"claim": text})
            assert response.status_code == 403
            assert response.json()["detail"] == {
                "code": "DEMO_ONLY",
                "message": "This public demo accepts only the built-in deterministic example.",
            }
    with Session(db.engine) as session:
        assert session.scalar(select(func.count()).select_from(VerificationRunRow)) == 0


def test_fixture_remains_fully_usable_in_public_demo(monkeypatch, tmp_path):
    _configure_demo(monkeypatch, tmp_path)
    with TestClient(api_module.app) as client:
        response = client.post("/api/v1/claims/verify", json={"claim": f"  {FIXTURE}  "})
        assert response.status_code == 200
        public_run = response.json()
        run_id = public_run["run_id"]
        for field in (
            "claim",
            "queries",
            "evidence",
            "assessments",
            "verdict_details",
            "agent_runs",
            "provider_usage",
            "created_at",
            "completed_at",
        ):
            assert field not in public_run
        assert public_run["supporting_evidence"] is not None
        assert public_run["contradicting_evidence"] is not None
        for path in (
            f"/api/v1/runs/{run_id}",
            f"/api/v1/runs/{run_id}/evidence",
            f"/api/v1/runs/{run_id}/evidence-graph",
        ):
            assert client.get(path).status_code == 200


def test_demo_does_not_expose_historical_non_fixture_runs(monkeypatch, tmp_path, workflow):
    db = _configure_demo(monkeypatch, tmp_path)
    with TestClient(api_module.app) as client:
        historical = workflow.verify({"claim": "A private project has a confidential timeline."})
        db.save(historical)
        run_id = historical.run_id
        for suffix in ("", "/evidence", "/evidence-graph"):
            response = client.get(f"/api/v1/runs/{run_id}{suffix}")
            assert response.status_code == 404
            assert response.json()["detail"] == "run not found"

        monkeypatch.setattr(api_module, "settings", Settings(public_demo_only=False))
        assert client.get(f"/api/v1/runs/{run_id}").status_code == 200
        assert client.get("/api/v1/capabilities").json()["demo_only"] is False
