#!/bin/sh
# First start of an empty volume only (postgres image initdb hook): apply contracts/db/migrations
# in order, then create the login users the services connect as (C-12: runner_rw writes, api_ro
# reads; the migrations create those roles NOLOGIN). Passwords come from .env.
set -eu
for f in /migrations/*.sql; do
  echo "applying $f"
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f "$f"
done
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -v runner_pw="$RUNNER_DB_PASSWORD" -v api_pw="$API_DB_PASSWORD" <<'SQL'
CREATE ROLE runner LOGIN PASSWORD :'runner_pw' IN ROLE runner_rw;
CREATE ROLE api LOGIN PASSWORD :'api_pw' IN ROLE api_ro;
SQL
