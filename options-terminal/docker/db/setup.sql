-- Idempotent setup after the migrations (db-init, every `make start`). No login users: the
-- runner and API connect with the same connection string and `SET ROLE` (DB_ROLE) to the
-- NOLOGIN roles the migrations create, runner_rw and api_ro (CONTRACTS C-12).

-- The connecting user must be allowed to SET ROLE to both (PostgreSQL 16+ does not grant that
-- to a role's creator by default).
GRANT runner_rw TO CURRENT_USER;
GRANT api_ro TO CURRENT_USER;

-- Supabase serves the public schema over its REST API to `anon` and `authenticated`, and grants
-- them new tables by default. These tables are not for that API. (No-op without Supabase.)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
    REVOKE ALL ON slots, model_runs FROM anon;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
    REVOKE ALL ON slots, model_runs FROM authenticated;
  END IF;
END $$;

-- Row-level security: only runner_rw and api_ro see rows; everyone else (bar the owner) none.
ALTER TABLE slots ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_runs ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS runner_slots ON slots;
DROP POLICY IF EXISTS api_slots ON slots;
DROP POLICY IF EXISTS runner_runs_read ON model_runs;
DROP POLICY IF EXISTS runner_runs_insert ON model_runs;
DROP POLICY IF EXISTS api_runs ON model_runs;
-- registry sync reads, inserts and updates slots (M-14); runs are append-only (C-10)
CREATE POLICY runner_slots ON slots FOR ALL TO runner_rw USING (true) WITH CHECK (true);
CREATE POLICY api_slots ON slots FOR SELECT TO api_ro USING (true);
CREATE POLICY runner_runs_read ON model_runs FOR SELECT TO runner_rw USING (true);
CREATE POLICY runner_runs_insert ON model_runs FOR INSERT TO runner_rw WITH CHECK (true);
CREATE POLICY api_runs ON model_runs FOR SELECT TO api_ro USING (true);
