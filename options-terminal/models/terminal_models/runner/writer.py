"""The only module that touches Postgres (M-3).

- Connects with DATABASE_URL: a login holding `runner_rw` (CONTRACTS C-12, X-6).
- Advisory lock per (slot_id, scope_key, role), so runs never overlap (M-19).
- One model_runs row per run in a single transaction, `NOTIFY model_runs '<slot_id>'` on
  success (C-13). Failed runs are written too (M-17).
- Registry sync (M-14): upserts each registry slot (definition, production, shadows, active),
  retires slots that left the registry (never deletes), then `NOTIFY slots` if anything changed
  (C-13a).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from terminal_models.core.registry import Registry
from terminal_models.core.slots import SlotDefinition


class Writer:
    def __init__(self, dsn: str | None = None) -> None:
        dsn = dsn or os.environ.get("DATABASE_URL")
        if not dsn:
            raise RuntimeError("DATABASE_URL is required (a login with the runner_rw role)")
        self.conn = psycopg.connect(dsn, autocommit=True)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def lock(self, slot_id: str, scope_key: str, role: str) -> Iterator[bool]:
        key = f"{slot_id}|{scope_key}|{role}"
        row = self.conn.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (key,)
        ).fetchone()
        acquired = bool(row and row[0])
        try:
            yield acquired
        finally:
            if acquired:
                self.conn.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (key,))

    def write_run(self, envelope: dict[str, Any], role: str, scope_key: str) -> None:
        with self.conn.transaction():
            self.conn.execute(
                """INSERT INTO model_runs (run_id, slot_id, implementation_id,
                       implementation_version,
                       role, scope_key, scope, status, as_of, started_at, finished_at, input_hash,
                       git_sha, envelope)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    envelope["run_id"],
                    envelope["slot_id"],
                    envelope["implementation_id"],
                    envelope["implementation_version"],
                    role,
                    scope_key,
                    Jsonb(envelope["scope"]),
                    envelope["status"],
                    envelope["as_of"],
                    envelope["started_at"],
                    envelope["finished_at"],
                    envelope["inputs"]["hash"],
                    envelope["code"]["git_sha"],
                    Jsonb(envelope),
                ),
            )
            if envelope["status"] == "success":
                self.conn.execute("SELECT pg_notify('model_runs', %s)", (envelope["slot_id"],))

    def fetch_envelope(self, run_id: str) -> tuple[dict[str, Any], str] | None:
        row = self.conn.execute(
            "SELECT envelope, role FROM model_runs WHERE run_id = %s", (run_id,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def sync_registry(self, registry: Registry, slots: dict[str, SlotDefinition]) -> bool:
        """Returns True if the slots table changed (and NOTIFY slots was sent)."""
        changed = False
        with self.conn.transaction():
            for slot_id, entry in registry.slots.items():
                slot = slots[slot_id]
                shadows = [s.model_dump() for s in entry.shadow]
                prod = entry.production
                cur = self.conn.execute(
                    """INSERT INTO slots (slot_id, title, description, payload_kind,
                           payload_version,
                           scope_dimensions, max_age_seconds, definition, production_impl,
                           production_version, shadow_implementations, status)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active')
                       ON CONFLICT (slot_id) DO UPDATE SET
                           title = EXCLUDED.title, description = EXCLUDED.description,
                           payload_kind = EXCLUDED.payload_kind,
                           payload_version = EXCLUDED.payload_version,
                           scope_dimensions = EXCLUDED.scope_dimensions,
                           max_age_seconds = EXCLUDED.max_age_seconds,
                           definition = EXCLUDED.definition,
                           production_impl = EXCLUDED.production_impl,
                           production_version = EXCLUDED.production_version,
                           shadow_implementations = EXCLUDED.shadow_implementations,
                           status = 'active', updated_at = now()
                       WHERE (slots.title, slots.description, slots.payload_kind,
                              slots.payload_version, slots.scope_dimensions, slots.max_age_seconds,
                              slots.definition, slots.production_impl, slots.production_version,
                              slots.shadow_implementations, slots.status)
                             IS DISTINCT FROM
                             (EXCLUDED.title, EXCLUDED.description, EXCLUDED.payload_kind,
                              EXCLUDED.payload_version, EXCLUDED.scope_dimensions,
                              EXCLUDED.max_age_seconds, EXCLUDED.definition,
                              EXCLUDED.production_impl, EXCLUDED.production_version,
                              EXCLUDED.shadow_implementations, 'active')""",
                    (
                        slot_id,
                        slot.title,
                        slot.description,
                        slot.payload_kind,
                        slot.payload_version,
                        slot.scope.dimensions,
                        slot.max_age_seconds,
                        Jsonb(json.loads(slot.model_dump_json(exclude_none=True))),
                        prod.implementation if prod else None,
                        prod.version if prod else None,
                        Jsonb(shadows),
                    ),
                )
                changed |= cur.rowcount > 0
            cur = self.conn.execute(
                "UPDATE slots SET status = 'retired', updated_at = now() "
                "WHERE status = 'active' AND NOT (slot_id = ANY(%s))",
                (list(registry.slots),),
            )
            changed |= cur.rowcount > 0
            if changed:
                self.conn.execute("NOTIFY slots")
        return changed
