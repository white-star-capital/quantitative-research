-- CONTRACTS.md §6. Migrations are the only way the schema changes.

CREATE TABLE slots (
    slot_id            TEXT PRIMARY KEY,
    title              TEXT NOT NULL,
    description        TEXT NOT NULL,
    payload_kind       TEXT NOT NULL,
    payload_version    TEXT NOT NULL,
    scope_dimensions   TEXT[] NOT NULL,
    max_age_seconds    INTEGER NOT NULL CHECK (max_age_seconds > 0),
    definition         JSONB NOT NULL,          -- full slot YAML as JSON
    production_impl    TEXT,                    -- from registry.yaml
    production_version TEXT,
    status             TEXT NOT NULL CHECK (status IN ('active','retired')),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE model_runs (
    run_id                 UUID PRIMARY KEY,
    slot_id                TEXT NOT NULL REFERENCES slots(slot_id),
    implementation_id      TEXT NOT NULL,
    implementation_version TEXT NOT NULL,
    role                   TEXT NOT NULL CHECK (role IN ('production','shadow')),
    scope_key              TEXT NOT NULL,       -- canonical, e.g. 'underlying=BTC'
    scope                  JSONB NOT NULL,
    status                 TEXT NOT NULL CHECK (status IN ('success','failed')),
    as_of                  TIMESTAMPTZ NOT NULL,
    started_at             TIMESTAMPTZ NOT NULL,
    finished_at            TIMESTAMPTZ NOT NULL,
    input_hash             TEXT NOT NULL,
    git_sha                TEXT NOT NULL,
    envelope               JSONB NOT NULL       -- full envelope as served
);

CREATE INDEX model_runs_latest
    ON model_runs (slot_id, scope_key, role, status, as_of DESC);

-- C-12: runner_rw writes, api_ro reads. Roles are NOLOGIN here; login users and
-- passwords are granted these roles by ops from the secrets manager (X-6).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'runner_rw') THEN
        CREATE ROLE runner_rw NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'api_ro') THEN
        CREATE ROLE api_ro NOLOGIN;
    END IF;
END
$$;

GRANT SELECT, INSERT ON model_runs TO runner_rw;          -- append-only (C-10)
GRANT SELECT, INSERT, UPDATE ON slots TO runner_rw;       -- registry sync upserts (M-14)
GRANT SELECT ON slots, model_runs TO api_ro;
