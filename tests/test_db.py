from vericlaim.db import Database


def test_postgres_urls_use_psycopg_v3_driver() -> None:
    for scheme in ("postgres://", "postgresql://"):
        database = Database(f"{scheme}user:password@example.test/vericlaim")

        assert database.engine.url.drivername == "postgresql+psycopg"


def test_sqlite_url_remains_unchanged() -> None:
    database = Database("sqlite:///:memory:")

    assert database.engine.url.drivername == "sqlite"
