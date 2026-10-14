-- Supabase setup for the options terminal; run by `make supabase-init` as `postgres`, after
-- contracts/db/migrations. Idempotent: re-running resets the passwords and policies.
--   psql -v runner_pw=... -v api_pw=... -f supabase_setup.sql
--
-- 1. Login users the services connect as (CONTRACTS C-12): runner writes, api reads.
SELECT 'CREATE ROLE runner' WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'runner') \gexec
SELECT 'CREATE ROLE api' WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'api') \gexec
ALTER ROLE runner LOGIN PASSWORD :'runner_pw';
ALTER ROLE api LOGIN PASSWORD :'api_pw';
GRANT runner_rw TO runner;
GRANT api_ro TO api;

-- 2. Supabase serves the public schema over its REST API to `anon` and `authenticated`, and
--    grants them new tables by default. These tables are not for that API: revoke, and enable
--    row-level security with policies for our two roles only (deny everyone else).
REVOKE ALL ON slots, model_runs FROM anon, authenticated;
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
