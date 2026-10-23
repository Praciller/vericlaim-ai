"""Read-only, aggregate-only retention preflight for an isolated LOCAL database.

Rejects remote database URLs; the operator must also avoid local tunnels
forwarding to production. Does not initialize schemas, purge data, reveal
row contents, or accept credentials in CLI arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError

from vericlaim.db import (
    AgentRunRow,
    AtomicClaimRow,
    ClaimRow,
    EvidenceAssessmentRow,
    EvidenceRow,
    ProviderUsageRow,
    SearchQueryRow,
    SourceRow,
    VerdictRow,
    VerificationRunRow,
)

CHILD_TABLES = (
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
LOCAL_PG_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def validate_local_url(value: str) -> URL:
    """Fail closed before opening any connection, including untrusted hostnames."""
    if not value:
        raise ValueError("Set RETENTION_PREFLIGHT_DATABASE_URL to an isolated local DB")
    try:
        url = make_url(value)
    except ArgumentError as exc:
        raise ValueError("Invalid local database URL") from exc
    # DB drivers can honor URL query parameters such as host/hostaddr/service
    # that override the apparent host. SQLite URI parameters can also redirect
    # the file target. No query arguments are required for this operator tool.
    if url.query:
        raise ValueError("Database URL query parameters are not supported")
    backend = url.get_backend_name()
    if backend == "postgresql":
        if url.host not in LOCAL_PG_HOSTS or not url.database:
            raise ValueError("Remote PostgreSQL and Unix-socket connections are forbidden")
    elif backend == "sqlite":
        if url.database in (None, "", ":memory:"):
            raise ValueError("Supply an existing local SQLite database file")
        if url.database.startswith(("//", "\\\\")):
            raise ValueError("Network/UNC SQLite paths are forbidden")
        resolved = Path(url.database).resolve()
        if str(resolved).startswith(("//", "\\\\")) or not resolved.is_file():
            raise ValueError("SQLite must be an existing local file")
    else:
        raise ValueError("Only local SQLite or loopback PostgreSQL are supported")
    return url


def create_local_engine(url: URL) -> Engine:
    """Pin libpq's actual destination IP, independent of PGHOSTADDR env vars."""
    if url.get_backend_name() == "postgresql":
        loopback = "127.0.0.1" if url.host == "localhost" else url.host
        return create_engine(url, connect_args={"hostaddr": loopback})
    return create_engine(url)


def inspect_retention(
    engine: Engine, *, now: datetime, retention_days: int = 30
) -> dict[str, object]:
    """Query only aggregates using a read-only transaction/SQLite connection."""
    if not 1 <= retention_days <= 365:
        raise ValueError("Retention days must be between 1 and 365")
    if now.tzinfo is None:
        raise ValueError("Inspection clock must be timezone-aware")
    if engine.dialect.name not in ("sqlite", "postgresql"):
        raise ValueError("Unsupported database engine")
    cutoff = now.astimezone(UTC) - timedelta(days=retention_days)
    expired = select(VerificationRunRow.run_id).where(VerificationRunRow.created_at < cutoff)

    with engine.connect() as connection:
        if engine.dialect.name == "sqlite":
            connection.exec_driver_sql("PRAGMA query_only = ON")
        else:
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        try:
            total_runs = int(
                connection.scalar(select(func.count()).select_from(VerificationRunRow)) or 0
            )
            expired_runs = int(
                connection.scalar(
                    select(func.count())
                    .select_from(VerificationRunRow)
                    .where(VerificationRunRow.created_at < cutoff)
                )
                or 0
            )
            rows = {"verification_runs": expired_runs}
            for model in CHILD_TABLES:
                rows[model.__tablename__] = int(
                    connection.scalar(
                        select(func.count()).select_from(model).where(model.run_id.in_(expired))
                    )
                    or 0
                )
        finally:
            if engine.dialect.name == "sqlite":
                connection.exec_driver_sql("PRAGMA query_only = OFF")

    return {
        "snapshot_utc": now.astimezone(UTC).isoformat(),
        "cutoff_utc": cutoff.isoformat(),
        "retention_days": retention_days,
        "total_verification_runs": total_runs,
        "expired_verification_runs": expired_runs,
        "eligible_delete_row_counts": rows,
        "eligible_delete_total_rows": sum(rows.values()),
        "read_only": True,
        "local_database_only": True,
        "warning": "Snapshot estimate, not approval or live Production delete count",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Local-only, SELECT-only retention impact")
    parser.add_argument("--days", type=int, default=30, help="Retention cutoff (1-365)")
    args = parser.parse_args()
    try:
        url = validate_local_url(os.getenv("RETENTION_PREFLIGHT_DATABASE_URL", ""))
        if not 1 <= args.days <= 365:
            raise ValueError("Retention days must be between 1 and 365")
        engine = create_local_engine(url)
        try:
            result = inspect_retention(engine, now=datetime.now(UTC), retention_days=args.days)
        finally:
            engine.dispose()
    except Exception:
        # Avoid echoing URLs/credentials and provider exceptions into terminal/CI logs.
        print("RETENTION_PREFLIGHT_FAILED; no changes were made", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
