# contracts/

The data contract shared by `models/`, `api/` and `frontend/`. The rules live in [`../CONTRACTS.md`](../CONTRACTS.md); this folder holds the machine-readable artifacts.

| Path | Status |
|---|---|
| `schemas/envelope/v1.json` | Done |
| `schemas/payloads/<kind>/v1.json` | M0, not written yet |
| `slots/*.yaml` | `vol_surface` only (the CONTRACTS §4 example) |
| `db/migrations/` | `0001_init.sql` from CONTRACTS §6 |
| `fixtures/valid/`, `fixtures/invalid/` | M0. Fixtures must come from recorded real snapshots, never invented prices |
| `fixtures/scope_key_vectors.json` | Shared C-9 test vectors for Python and Rust |
