# contracts/

The data contract shared by `models/`, `api/` and `frontend/`. The rules live in [`../CONTRACTS.md`](../CONTRACTS.md); this folder holds the machine-readable artifacts.

| Path | Status |
|---|---|
| `schemas/envelope/v1.json` | Done |
| `schemas/payloads/<kind>/v1.json` | Done (scalar_set, curve, surface, timeseries, table, composite) |
| `slots/*.yaml` | `vol_surface` (CONTRACTS §4 example), `bs_pricer`, `parity_monitor` |
| `db/migrations/` | `0001_init.sql` from CONTRACTS §6 |
| `fixtures/valid/`, `fixtures/invalid/` | M0. Fixtures must come from recorded real snapshots, never invented prices |
| `fixtures/pricing/` | Python→Rust evaluation parity vectors (math only, no market data) |
| `fixtures/parity/vectors.json` | Parity formula vectors for the frontend (math only, no market data) |
| `fixtures/scope_key_vectors.json` | Shared C-9 test vectors for Python and Rust |
