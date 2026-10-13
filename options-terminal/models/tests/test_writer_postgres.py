"""Writer + runner against real Postgres (M-3, M-14, M-17, M-19, C-13, C-13a).

Skipped unless TEST_DATABASE_URL points at a server where we may create databases, e.g.
    TEST_DATABASE_URL=postgresql://postgres@localhost:5432/postgres \
        uv run pytest tests/test_writer_postgres.py
Each module run gets a fresh database with contracts/db/migrations applied; the writer connects
as a login holding only runner_rw.
"""

import json
import os
import uuid

import psycopg
import pytest

from terminal_models.core.paths import contracts_dir
from terminal_models.core.registry import Registry
from terminal_models.core.slots import load_slots
from terminal_models.runner import cli
from terminal_models.runner.snapshots import SnapshotStore
from terminal_models.runner.writer import Writer

ADMIN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not ADMIN, reason="TEST_DATABASE_URL not set")


@pytest.fixture(scope="module")
def dsn():
    name = f"ot_writer_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(ADMIN, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    base = psycopg.conninfo.conninfo_to_dict(ADMIN)
    owner = psycopg.conninfo.make_conninfo(**(base | {"dbname": name}))
    with psycopg.connect(owner, autocommit=True) as conn:
        for path in sorted((contracts_dir() / "db" / "migrations").glob("*.sql")):
            conn.execute(path.read_text())
        conn.execute(
            "DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='runner_test')"
            " THEN CREATE ROLE runner_test LOGIN; END IF; END $$"
        )
        conn.execute("GRANT runner_rw TO runner_test")
    yield psycopg.conninfo.make_conninfo(**(base | {"dbname": name, "user": "runner_test"})), owner
    with psycopg.connect(ADMIN, autocommit=True) as admin:
        admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')


def registry(production=None):
    shadow = [{"implementation": "bs_pricer_black76", "version": "1.0.0"}]
    return Registry.model_validate(
        {
            "slots": {
                "bs_pricer": {
                    "production": production,
                    "shadow": shadow,
                    "scopes": [{"underlying": "BTC"}],
                }
            }
        }
    )


def test_sync_is_idempotent_and_notifies(dsn):
    rw, owner = dsn
    slots = load_slots()
    with psycopg.connect(owner, autocommit=True) as listener:
        listener.execute("LISTEN slots")
        w = Writer(rw)
        assert w.sync_registry(registry(), slots) is True
        assert w.sync_registry(registry(), slots) is False  # no change → no NOTIFY
        w.close()
        notes = list(listener.notifies(timeout=1, stop_after=5))
    assert [n.channel for n in notes] == ["slots"]
    with psycopg.connect(owner) as conn:
        row = conn.execute(
            "SELECT production_impl, shadow_implementations, status, "
            "definition->'evaluation'->>'evaluator' FROM slots"
        ).fetchone()
    assert row == (
        None,
        [{"implementation": "bs_pricer_black76", "version": "1.0.0"}],
        "active",
        "black76_surface_v1",
    )


def test_removed_slot_is_retired_not_deleted(dsn):
    rw, owner = dsn
    w = Writer(rw)
    w.sync_registry(registry(), load_slots())
    assert w.sync_registry(Registry.model_validate({"slots": {}}), load_slots()) is True
    w.sync_registry(registry(), load_slots())  # comes back active
    w.close()
    with psycopg.connect(owner) as conn:
        assert conn.execute("SELECT status FROM slots WHERE slot_id='bs_pricer'").fetchone() == (
            "active",
        )


def test_run_from_snapshot_writes_and_replays(dsn, tmp_path, monkeypatch, capsys):
    """CLI end to end: stored inputs → run → row + NOTIFY → replay identical."""
    from terminal_models.library.bs_pricer_black76.tests import synthetic as syn

    rw, owner = dsn
    monkeypatch.setenv("DATABASE_URL", rw)
    monkeypatch.setenv("SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("TERMINAL_GIT_SHA", "a1b2c3d")
    digest = SnapshotStore(tmp_path).save(syn.inputs())

    with psycopg.connect(owner, autocommit=True) as listener:
        listener.execute("LISTEN model_runs")
        code = cli.main(
            [
                "run",
                "bs_pricer",
                "--scope",
                "underlying=BTC",
                "--role",
                "shadow",
                "--snapshot",
                digest,
                "--as-of",
                "2026-10-13T17:00:00Z",
            ]
        )
        assert code == 0, capsys.readouterr().err
        notes = [n.payload for n in listener.notifies(timeout=1, stop_after=1)]
    assert notes == ["bs_pricer"]

    with psycopg.connect(owner) as conn:
        run_id, role, status, scope_key = conn.execute(
            "SELECT run_id, role, status, scope_key FROM model_runs"
        ).fetchone()
    assert (role, status, scope_key) == ("shadow", "success", "underlying=BTC")
    assert cli.main(["replay", str(run_id)]) == 0
    assert "replay identical" in capsys.readouterr().out


def test_failed_run_is_written_and_lock_blocks_overlap(dsn):
    rw, owner = dsn
    w = Writer(rw)
    w.sync_registry(registry(), load_slots())
    with w.lock("bs_pricer", "underlying=ETH", "shadow") as first:
        other = Writer(rw)
        with other.lock("bs_pricer", "underlying=ETH", "shadow") as second:
            assert (first, second) == (True, False)
        other.close()
    envelope = json.loads(
        json.dumps(
            {
                "envelope_version": "1",
                "run_id": "0192f3a0-6b1e-7c3a-9a51-2f8e4d1c7b29",
                "slot_id": "bs_pricer",
                "implementation_id": "bs_pricer_black76",
                "implementation_version": "1.0.0",
                "scope": {"underlying": "ETH"},
                "status": "failed",
                "as_of": "2026-10-13T17:00:00Z",
                "started_at": "2026-10-13T17:00:00Z",
                "finished_at": "2026-10-13T17:00:01Z",
                "inputs": {"sources": [], "hash": "sha256:" + "0" * 64},
                "code": {"git_sha": "a1b2c3d"},
                "warnings": [],
                "payload_kind": "composite",
                "payload_version": "1",
                "payload": None,
                "error": {"code": "venue_unavailable", "message": "deribit unreachable"},
            }
        )
    )
    w.write_run(envelope, "shadow", "underlying=ETH")
    w.close()
    with psycopg.connect(owner) as conn:
        assert conn.execute(
            "SELECT status FROM model_runs WHERE scope_key='underlying=ETH'"
        ).fetchone() == ("failed",)
