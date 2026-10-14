#!/bin/sh
# Runner container: sync the registry into the slots table, then run every due model at the
# top of each minute (MODELS §6). One pass at a time; a slow pass delays the next, never overlaps.
set -u
cd /app/models
until python -m terminal_models.runner sync-registry; do
  echo "sync-registry failed; retrying in 5s" >&2
  sleep 5
done
while true; do
  python -m terminal_models.runner run-due || echo "run-due pass failed (exit $?)" >&2
  sleep $((60 - $(date +%S | sed 's/^0//')))
done
