#!/bin/sh
# db-init (runs on every `make start`, before the runner and API): apply contracts/db/migrations
# if the schema is absent, then docker/db/setup.sql. Same script for the local container and
# Supabase; DATABASE_URL is the one connection string both services use.
set -eu
until psql "$DATABASE_URL" -qtAc "SELECT 1" >/dev/null 2>&1; do
  echo "waiting for the database"; sleep 2
done
if [ "$(psql "$DATABASE_URL" -tAc "SELECT to_regclass('public.slots') IS NOT NULL")" = "t" ]; then
  echo "schema present"
else
  for f in /migrations/*.sql; do
    echo "applying $f"
    psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f "$f"
  done
fi
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f /setup/setup.sql
echo "database ready"
