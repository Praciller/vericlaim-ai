"""Retention protects claim privacy, including traces outside the public projection."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import vericlaim.api as api_module
from vericlaim.config import Settings
from vericlaim.db import (
    AgentRunRow,
    AtomicClaimRow,
    ClaimRow,
    Database,
    EvidenceAssessmentRow,
    EvidenceRow,
    ProviderUsageRow,
    SearchQueryRow,
    SourceRow,
    VerdictRow,
    VerificationRunRow,
)

RELATED_MODELS = (
    VerificationRunRow,
    ClaimRow,
    AtomicClaimRow,
    SearchQueryRow,
    SourceRow,
    EvidenceRow,
    EvidenceAssessmentRow,
    VerdictRow,
    AgentRunRow,
    ProviderUsageRow,
)


def test_retention_bounds_validate_configuration():
    assert Settings(run_retention_days=30).run_retention_days == 30
    with pytest.raises(ValueError):
        Settings(run_retention_days=0)
    with pytest.raises(ValueError):
        Settings(run_retention_days=366)
    with pytest.raises(ValueError, match="positive"):
        Database("sqlite:///:memory:", retention_days=0)


def test_retention_expiry_blocks_read_and_cascades_all_related_rows(workflow):
    current = [datetime(2026, 10, 23, tzinfo=UTC)]
    db = Database("sqlite:///:memory:", retention_days=30, clock=lambda: current[0])
    db.init()
    result = workflow.verify({"claim": "RAG eliminates hallucinations."})
    result.created_at = current[0]
    db.save(result)
    assert db.get(result.run_id) is not None
    assert db.get_evidence(result.run_id) is not None

    with Session(db.engine) as session:
        before = {
            model.__tablename__: session.scalar(select(func.count()).select_from(model))
            for model in RELATED_MODELS
        }
    assert before["verification_runs"] == 1
    assert before["claims"] == 1
    assert before["verdicts"] == 1
    assert before["agent_runs"] >= 1

    current[0] += timedelta(days=31)
    assert db.get(result.run_id) is None
    assert db.get_evidence(result.run_id) is None
    assert db.purge_expired() == 0
    with Session(db.engine) as session:
        for model in RELATED_MODELS:
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_startup_prunes_existing_expired_records(workflow, tmp_path):
    path = tmp_path / "nested" / "app.sqlite3"
    current = [datetime(2026, 10, 23, tzinfo=UTC)]
    url = f"sqlite:///{path.as_posix()}"
    db = Database(url, retention_days=30, clock=lambda: current[0])
    db.init()
    result = workflow.verify({"claim": "RAG eliminates hallucinations."})
    result.created_at = current[0]
    db.save(result)

    current[0] += timedelta(days=31)
    restarted = Database(url, retention_days=30, clock=lambda: current[0])
    restarted.init()
    assert restarted.get(result.run_id) is None
    with Session(restarted.engine) as session:
        for model in RELATED_MODELS:
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_save_prunes_expired_runs_but_preserves_recent_runs(workflow):
    current = [datetime(2026, 10, 23, tzinfo=UTC)]
    db = Database("sqlite:///:memory:", retention_days=30, clock=lambda: current[0])
    db.init()
    old = workflow.verify({"claim": "RAG eliminates hallucinations."})
    old.created_at = current[0]
    db.save(old)
    current[0] += timedelta(days=31)
    new = workflow.verify({"claim": "RAG improves retrieval accuracy."})
    new.created_at = current[0]
    db.save(new)
    assert db.get(old.run_id) is None
    assert db.get(new.run_id) is not None
    with Session(db.engine) as session:
        assert session.scalar(select(func.count()).select_from(VerificationRunRow)) == 1


def test_public_run_and_graph_endpoints_return_404_after_expiry(workflow, monkeypatch, tmp_path):
    current = [datetime(2026, 10, 23, tzinfo=UTC)]
    db = Database(
        f"sqlite:///{(tmp_path / 'api.sqlite3').as_posix()}",
        retention_days=30,
        clock=lambda: current[0],
    )
    monkeypatch.setattr(api_module, "database", db)
    result = workflow.verify({"claim": "RAG eliminates hallucinations."})
    with TestClient(api_module.app) as client:
        result.created_at = current[0]
        db.save(result)
        assert client.get(f"/api/v1/runs/{result.run_id}").status_code == 200
        assert client.get(f"/api/v1/runs/{result.run_id}/evidence").status_code == 200
        assert client.get(f"/api/v1/runs/{result.run_id}/evidence-graph").status_code == 200
        current[0] += timedelta(days=31)
        for suffix in ("", "/evidence", "/evidence-graph"):
            response = client.get(f"/api/v1/runs/{result.run_id}{suffix}")
            assert response.status_code == 404
            assert response.json() == {"detail": "run not found"}


def test_expiry_boundary_is_exclusive(workflow):
    now = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    current = [now]
    db = Database("sqlite:///:memory:", retention_days=30, clock=lambda: current[0])
    db.init()
    result = workflow.verify({"claim": "RAG eliminates hallucinations."})
    result.created_at = current[0]
    db.save(result)

    current[0] += timedelta(days=30)
    # Exact boundary is still available; older-than-boundary is expired.
    assert db.get(result.run_id) is not None
    current[0] += timedelta(seconds=1)
    assert db.get(result.run_id) is None


def test_migration_preserve_mode_blocks_reads_without_deleting_any_rows(workflow, tmp_path):
    """A restart or read/write cannot irreversibly purge legacy production data."""
    current = [datetime(2026, 10, 23, tzinfo=UTC)]
    url = f"sqlite:///{(tmp_path / 'preserve.sqlite3').as_posix()}"
    original = Database(url, retention_days=30, clock=lambda: current[0])
    original.init()
    old = workflow.verify({"claim": "RAG eliminates hallucinations."})
    old.created_at = current[0]
    original.save(old)

    current[0] += timedelta(days=31)
    preserved = Database(url, retention_days=30, purge_enabled=False, clock=lambda: current[0])
    preserved.init()
    assert preserved.get(old.run_id) is None
    assert preserved.get_evidence(old.run_id) is None
    assert preserved.purge_expired() == 0

    new = workflow.verify({"claim": "RAG improves retrieval accuracy."})
    new.created_at = current[0]
    preserved.save(new)
    assert preserved.get(new.run_id) is not None

    restarted = Database(url, retention_days=30, purge_enabled=False, clock=lambda: current[0])
    restarted.init()
    assert restarted.get(old.run_id) is None
    assert restarted.get(new.run_id) is not None
    with Session(restarted.engine) as session:
        for model in RELATED_MODELS:
            assert session.scalar(
                select(func.count()).select_from(model).where(model.run_id == old.run_id)
            ) >= (1 if model != ProviderUsageRow else 0)
        assert session.scalar(select(func.count()).select_from(VerificationRunRow)) == 2

    # Once migration is approved and the flag is deliberately re-enabled,
    # retention performs the usual cleanup and does not delete newer rows.
    approved_purge = Database(url, retention_days=30, clock=lambda: current[0])
    approved_purge.init()
    assert approved_purge.get(old.run_id) is None
    assert approved_purge.get(new.run_id) is not None
    with Session(approved_purge.engine) as session:
        for model in RELATED_MODELS:
            assert (
                session.scalar(
                    select(func.count()).select_from(model).where(model.run_id == old.run_id)
                )
                == 0
            )
