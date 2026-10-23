"""The retention preflight must be read-only and reject remote targets."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from scripts.retention_impact import (
    create_local_engine,
    inspect_retention,
    validate_local_url,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from vericlaim.db import Database, VerificationRunRow


def test_preview_counts_expired_runs_without_deleting_anything(workflow, tmp_path):
    now = datetime(2026, 10, 23, tzinfo=UTC)
    db = Database(
        f"sqlite:///{(tmp_path / 'isolated.sqlite').as_posix()}",
        retention_days=30,
        purge_enabled=False,
        clock=lambda: now,
    )
    db.init()
    first = workflow.verify({"claim": "RAG eliminates hallucinations."})
    first.created_at = now - timedelta(days=31)
    db.save(first)
    second = workflow.verify({"claim": "RAG eliminates hallucinations."})
    second.created_at = now - timedelta(days=30)
    db.save(second)

    with Session(db.engine) as session:
        before = session.scalar(select(func.count()).select_from(VerificationRunRow))
    report = inspect_retention(db.engine, now=now, retention_days=30)
    with Session(db.engine) as session:
        after = session.scalar(select(func.count()).select_from(VerificationRunRow))

    assert before == after == 2
    assert report["total_verification_runs"] == 2
    assert report["expired_verification_runs"] == 1
    counts = report["eligible_delete_row_counts"]
    assert counts["verification_runs"] == 1
    assert counts["claims"] == 1
    assert counts["verdicts"] == 1
    assert counts["agent_runs"] > 0
    assert sum(counts.values()) == report["eligible_delete_total_rows"]
    assert report["read_only"] is True
    assert report["local_database_only"] is True
    assert db.get(first.run_id) is None
    assert db.get(second.run_id) is not None


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://owner:SECRET@ep-a.neon.tech/example",
        "postgresql://owner:SECRET@192.0.2.1/example",
        "postgresql+psycopg:///example",
        "postgresql+psycopg://owner:SECRET@127.0.0.1/example?host=ep-a.neon.tech",
        "postgresql+psycopg://owner:SECRET@localhost/example?hostaddr=192.0.2.1",
        "postgresql+psycopg://owner:SECRET@localhost/example?service=remote",
        "postgresql+psycopg://localhost/example?options=-c%20search_path%3Dpublic",
        "mysql://owner:SECRET@localhost/example",
        "sqlite:///:memory:",
        "sqlite:///test.sqlite?uri=true",
        "sqlite://///fileserver/shared/backup.sqlite",
        "",
    ],
)
def test_rejects_remote_or_unsafe_targets(url):
    with pytest.raises(ValueError):
        validate_local_url(url)


def test_local_targets_allowed_only_when_sqlite_file_exists(tmp_path):
    local = tmp_path / "db.sqlite"
    local.touch()
    assert validate_local_url(f"sqlite:///{local.as_posix()}").get_backend_name() == "sqlite"
    assert validate_local_url("postgresql+psycopg://localhost:5432/ci").host == "localhost"
    with pytest.raises(ValueError):
        validate_local_url(f"sqlite:///{(tmp_path / 'missing.sqlite').as_posix()}")


def test_psycopg_network_target_pinned_to_numeric_loopback():
    url = validate_local_url("postgresql+psycopg://localhost:5432/ci")
    with patch("scripts.retention_impact.create_engine") as builder:
        create_local_engine(url)
    assert builder.call_args.kwargs["connect_args"] == {"hostaddr": "127.0.0.1"}

    url = validate_local_url("postgresql+psycopg://[::1]:5432/ci")
    with patch("scripts.retention_impact.create_engine") as builder:
        create_local_engine(url)
    assert builder.call_args.kwargs["connect_args"] == {"hostaddr": "::1"}


def test_invalid_days_and_naive_clock_fail_without_querying(tmp_path):
    db = Database(f"sqlite:///{(tmp_path / 'test.sqlite').as_posix()}", purge_enabled=False)
    db.init()
    with pytest.raises(ValueError, match="between 1 and 365"):
        inspect_retention(db.engine, now=datetime.now(UTC), retention_days=0)
    with pytest.raises(ValueError, match="timezone-aware"):
        inspect_retention(db.engine, now=datetime(2026, 10, 23))
