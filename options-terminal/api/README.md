# api/

Read-only Rust/Axum JSON API over stored model runs. Spec: [`../API.md`](../API.md).

Slots and implementations are **data**. Models are added, promoted, shadowed and retired through `models/registry.yaml` (MODELS §5). The runner syncs the `slots` table and sends `NOTIFY slots`, and this server reloads its catalog without a restart (A-27–A-31). Nothing here names a slot or an implementation.

```bash
SQLX_OFFLINE=true cargo build          # uses committed .sqlx/ metadata; no DB needed
SQLX_OFFLINE=true cargo test           # unit + handler tests on in-memory repos
SQLX_OFFLINE=true cargo clippy --all-targets -- -D warnings
cargo fmt --check

# Integration tests: fresh DB per test, contracts/db/migrations applied
DATABASE_URL=postgres://postgres@localhost:5432/postgres cargo test --test postgres -- --ignored

# After changing any sqlx::query! (needs a migrated DB)
DATABASE_URL=… cargo sqlx prepare -- --all-targets
```

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `DATABASE_URL` | required | A login that has only the `api_ro` role (A-2) |
| `API_BIND` | `0.0.0.0:8080` | |
| `API_CORS_ORIGINS` | none | Comma-separated frontend origins; `*` is rejected (A-15) |
| `AUTH_MODE` | `none` | The only v1 value (A-17) |
| `CONTRACTS_DIR` | `../contracts` | Payload schemas served at `/v1/schemas/{kind}/{version}` |
| `API_PROBLEM_BASE_URL` | `https://options-terminal/problems` | Prefix for problem `type` URLs |
| `API_RATE_PER_MINUTE` / `API_RATE_BURST` | `120` / `30` | Per IP, `/v1` only; probes are never limited |
| `DB_MAX_CONNECTIONS` | `10` | |
| `RUST_LOG` | `info` | JSON logs with `request_id`; data responses also log `run_id` |

## Still open for M2

- Fixture-seeded contract test A-25, which waits for M0 fixtures.
- Load test against the §6 latency targets (A-26).
- Trusted-proxy CIDRs for `X-Forwarded-For` (A-18). Until then the limiter keys on the socket address, which is wrong behind a proxy.
