#!/bin/sh
# One-shot (make supabase-init): apply the migrations to a fresh Supabase database, then the
# Supabase setup (logins, anon lockdown, RLS). Connects as postgres through the session pooler.
set -eu
export PGHOST="$SUPABASE_POOLER_HOST" PGPORT=5432 PGDATABASE=postgres PGSSLMODE=require
export PGUSER="postgres.$SUPABASE_PROJECT_REF" PGPASSWORD="$SUPABASE_DB_PASSWORD"
if [ "$(psql -tAc "SELECT to_regclass('public.slots') IS NOT NULL")" = "t" ]; then
  echo "schema exists: migrations skipped (they are not re-runnable)"
else
  for f in /migrations/*.sql; do
    echo "applying $f"
    psql -v ON_ERROR_STOP=1 -q -f "$f"
  done
fi
psql -v ON_ERROR_STOP=1 -q -v runner_pw="$RUNNER_DB_PASSWORD" -v api_pw="$API_DB_PASSWORD" \
  -f /setup/supabase_setup.sql
for who in runner api; do
  pw=$([ "$who" = runner ] && echo "$RUNNER_DB_PASSWORD" || echo "$API_DB_PASSWORD")
  PGUSER="$who.$SUPABASE_PROJECT_REF" PGPASSWORD="$pw" psql -tAc "SELECT 'login ok: ' || current_user"
done
