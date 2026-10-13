# models/

Python model framework, data adapters, runner and model library. Spec: [`../MODELS.md`](../MODELS.md). First model: [`../BS_PRICER_IMPLEMENTATION_GUIDE.md`](../BS_PRICER_IMPLEMENTATION_GUIDE.md).

```bash
uv sync
uv run pytest                     # unit, model and contract tests
uv run lint-imports               # M-2 import rules
uv run ruff check .
uv run python -m terminal_models.runner validate-registry
uv run python -m terminal_models.tools.gen_pricing_vectors --check   # parity vectors up to date

# Postgres-backed tests (writer, registry sync, run → replay)
TEST_DATABASE_URL=postgresql://postgres@localhost:5432/postgres uv run pytest tests/test_writer_postgres.py
```

## Runner

```bash
export DATABASE_URL=postgresql://runner@…/options_terminal   # login with runner_rw
python -m terminal_models.runner sync-registry               # registry.yaml → slots, NOTIFY slots
python -m terminal_models.runner run bs_pricer --scope underlying=ETH --role shadow [--dry-run]
python -m terminal_models.runner run-due                     # scheduler entrypoint, every minute
python -m terminal_models.runner replay <run_id>             # recorded inputs → identical output
```

- Inputs are saved to the snapshot store (`SNAPSHOT_DIR`, default `models/.snapshots/`) under their input hash, together with the raw venue bodies (M-9, D-6).
- The git SHA comes from `TERMINAL_GIT_SHA`, falling back to `git rev-parse HEAD`.

## Status

| Piece | State |
|---|---|
| `core/quant` (Black-76, smile, metrics, evaluate reference) | Done |
| Deribit adapter | Built against the documented public API; **live verification pending network access** |
| `bs_pricer_black76` | Done; synthetic-chain tests. Golden tests on recorded snapshots pending Deribit access |
| Runner: run, run-due, replay, sync-registry | Done |
| `diff` (shadow diff report, M-16) | Not yet |
