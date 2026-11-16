"""Public demo mode rejects arbitrary data before retrieval or persistence."""

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import vericlaim.api as api_module
from vericlaim.config import Settings
from vericlaim.db import Database, VerificationRunRow
from vericlaim.rate_limit import InMemoryRateLimiter
from vericlaim.workflow import VerificationWorkflow

FIXTURE = "RAG eliminates hallucinations."


def _configure_demo(monkeypatch, tmp_path):
    db = Database(f"sqlite:///{(tmp_path / 'demo.sqlite3').as_posix()}")
    settings = Settings(public_demo_only=True, live_retrieval_enabled=False)
    monkeypatch.setattr(api_module, "settings", settings)
    monkeypatch.setattr(api_module, "workflow", VerificationWorkflow(settings))
    monkeypatch.setattr(api_module, "database", db)
    monkeypatch.setattr(api_module, "verify_rate_limiter", InMemoryRateLimiter(max_requests=200))
    return db


def test_capabilities_expose_mode_but_no_internal_secrets(monkeypatch, tmp_path):
    _configure_demo(monkeypatch, tmp_path)
    with TestClient(api_module.app) as client:
        response = client.get("/api/v1/capabilities")
        assert response.status_code == 200
        assert response.json() == {
            "demo_only": True,
            "demo_claim": FIXTURE,
            "demo_claims": list(api_module.DEMO_FIXTURE_CLAIMS),
        }
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
                "message": "This public demo accepts only the built-in deterministic examples.",
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


def test_each_public_fixture_has_expected_verdict_and_unique_evidence(monkeypatch, tmp_path):
    _configure_demo(monkeypatch, tmp_path)
    with TestClient(api_module.app) as client:
        for claim, (verdict, stance, score) in zip(
            api_module.DEMO_FIXTURE_CLAIMS,
            (
                ("REFUTED", "CONTRADICTS", 0.82),
                ("SUPPORTED", "SUPPORTS", 0.7),
                ("INSUFFICIENT_EVIDENCE", None, 0.2),
            ),
            strict=True,
        ):
            created = client.post("/api/v1/claims/verify", json={"claim": claim})
            assert created.status_code == 200
            run = created.json()
            assert run["verdict"] == verdict
            assert run["confidence"] == score
            assert run["summary"].strip()
            path = f"/api/v1/runs/{run['run_id']}"
            assert client.get(path).json() == run
            stored = client.get(f"{path}/evidence").json()["evidence"]
            graph = client.get(f"{path}/evidence-graph").json()
            evidence = [node for node in graph["nodes"] if node["kind"] == "evidence"]
            sources = [node for node in graph["nodes"] if node["kind"] == "source"]
            assert len(evidence) == len(sources) == len(stored) == int(stance is not None)
            if stance is None:
                assert run["supporting_evidence"] == run["contradicting_evidence"] == []
                continue

            item, node, source = stored[0], evidence[0], sources[0]
            assert run["supporting_evidence"] == (stored if stance == "SUPPORTS" else [])
            assert run["contradicting_evidence"] == (stored if stance == "CONTRADICTS" else [])
            assert node["stance"] == stance
            assert (
                node["label"]
                == {
                    "SUPPORTS": "Supporting evidence",
                    "CONTRADICTS": "Contradicting evidence",
                }[stance]
            )
            assert node["node_id"] == f"evidence:{item['evidence_id']}"
            for field in ("source_id", "excerpt", "direction", "evidence_level", "provenance"):
                assert node[field] == item[field]
            assert node["source_id"] == source["source_id"] == run["sources"][0]["source_id"]
            assert {
                (edge["relation"], edge["source"], edge["target"]) for edge in graph["edges"]
            } >= {
                ("has_evidence", f"atomic_claim:{item['atomic_id']}", node["node_id"]),
                ("cited_from", node["node_id"], source["node_id"]),
            }


def test_invalid_fixture_projection_is_rejected_before_persistence(monkeypatch, tmp_path):
    db = _configure_demo(monkeypatch, tmp_path)
    to_result = api_module.workflow._to_result

    def drop_public_counter_evidence(state):
        return to_result(state).model_copy(update={"contradicting_evidence": []})

    monkeypatch.setattr(api_module.workflow, "_to_result", drop_public_counter_evidence)
    with TestClient(api_module.app) as client:
        response = client.post("/api/v1/claims/verify", json={"claim": FIXTURE})
        assert response.status_code == 422
        assert response.json() == {"detail": "verification request is invalid"}

    with Session(db.engine) as session:
        assert session.scalar(select(func.count()).select_from(VerificationRunRow)) == 0
