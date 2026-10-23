"""PostgreSQL Cold Start bootstrap is atomic across independent API instances.

Run against a disposable, localhost-only PostgreSQL database named vericlaim_ci.
The GitHub CI backend job starts its own ephemeral Postgres service. This test
never connects to remote databases or production credentials.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import event, inspect
from sqlalchemy.engine import make_url

from vericlaim.db import Base, Database


@pytest.mark.skipif(
    not os.getenv("VERICLAIM_TEST_POSTGRES_URL"),
    reason="requires local disposable PostgreSQL test service",
)
def test_concurrent_postgresql_schema_bootstrap_uses_transaction_lock() -> None:
    url = os.environ["VERICLAIM_TEST_POSTGRES_URL"]
    parsed = make_url(url)
    if (
        parsed.get_backend_name() != "postgresql"
        or parsed.host not in {"127.0.0.1", "localhost", "::1"}
        or parsed.database != "vericlaim_ci"
    ):
        pytest.fail("PostgreSQL integration test requires localhost/vericlaim_ci")

    workers = 10
    barrier = Barrier(workers, timeout=20)

    def initialize(_: int) -> None:
        db = Database(url, purge_enabled=False)
        try:
            barrier.wait()
            db.init()
        finally:
            db.engine.dispose()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(initialize, range(workers)))
    assert len(results) == workers

    inspect_db = Database(url, purge_enabled=False)
    try:
        statements: list[str] = []

        @event.listens_for(inspect_db.engine, "before_cursor_execute")
        def record_bootstrap(connection, cursor, statement, parameters, context, executemany):
            if "pg_advisory_xact_lock" in statement:
                statements.append(statement)

        inspect_db.init()
        assert statements, "Postgres bootstrap must acquire transaction-level advisory lock"
        existing = set(inspect(inspect_db.engine).get_table_names())
        assert set(Base.metadata.tables).issubset(existing)
    finally:
        inspect_db.engine.dispose()
