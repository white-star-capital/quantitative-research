# API — `api/`

Read-only Rust HTTP server, built on Axum, that serves the latest model outputs as JSON. It never runs models or writes data. Read `CONTRACTS.md` first.

## 1. Scope

**In:** GET endpoints for slots, latest outputs, run history and schemas; health and readiness; OpenAPI document; caching, rate limiting, CORS, logging.
**Out (v1):** authentication, writes of any kind, websockets/streaming, computation on payloads.

- **A-1** The server MUST register only GET (and HEAD/OPTIONS as handled by middleware) routes. Any other method returns `405`.
- **A-2** The server MUST connect to Postgres as `api_ro` (CONTRACTS C-12). Read-only at the DB is the real guarantee; A-1 is the second line.

## 2. Stack

| Concern | Choice |
|---|---|
| HTTP | `axum` **0.8.x** (pin minor), `tokio` |
| Middleware | `tower-http`: `TraceLayer`, `CorsLayer`, `CompressionLayer` (gzip, br), `TimeoutLayer`, `RequestBodyLimitLayer`, request-id (`SetRequestIdLayer` / `PropagateRequestIdLayer`) |
| Rate limiting | `tower_governor` (per-IP) |
| DB | `sqlx` (Postgres) with offline query metadata committed (`cargo sqlx prepare`) so builds don't need a live DB |
| Serialization | `serde`, `serde_json` |
| OpenAPI | `utoipa` (served at `/openapi.json`) |
| Logging | `tracing`, `tracing-subscriber` with JSON formatter |
| Errors | `thiserror` + a single `AppError` → RFC 7807 `application/problem+json` |

Lockfile committed. MSRV declared in `Cargo.toml` (`rust-version`).

## 3. Endpoints

All under `/v1`. Path parameters use Axum 0.8 syntax (`/{slot_id}`), not `/:slot_id`.

| Method & path | Returns |
|---|---|
| `GET /health` | `200 {"status":"ok"}`; liveness only, no DB call |
| `GET /ready` | `200` if DB reachable and slots table loaded; `503` otherwise |
| `GET /openapi.json` | OpenAPI 3.1 document |
| `GET /v1/slots` | Active slots: id, title, description, payload kind/version, scope dimensions and allowed values, `max_age_seconds`, current production implementation/version |
| `GET /v1/slots/{slot_id}` | One slot's full definition, status (`active`/`retired`) and its implementations (production + all shadows) |
| `GET /v1/slots/{slot_id}/latest?<scope params>&role&implementation` | Latest successful envelope for that scope + `meta` (production by default) |
| `GET /v1/slots/{slot_id}/runs?<scope params>&from&to&status&role&implementation&limit&cursor&include_payload` | Run history, newest first, cursor-paginated |
| `GET /v1/runs/{run_id}` | One envelope by run ID (any role, any status) |
| `GET /v1/schemas/{kind}/{version}` | The JSON Schema for a payload kind |

Query parameter rules:
- **A-3** Scope params are the slot's declared dimensions, e.g. `?underlying=BTC`. Missing a required dimension → `400` listing the missing keys. Unknown params → `400`. Value outside the slot's allowlist → `400` listing allowed values.
- **A-4** `role` defaults to `production`. `role=shadow` is accepted on `/latest` and `/runs` so the desk can inspect shadow outputs (MODELS §5). Shadow responses MUST carry `meta.role = "shadow"`. A slot can have any number of shadows, so `/latest?role=shadow` MUST also carry `implementation=<id>`; missing or unknown → `400` listing the slot's current shadow implementations. On `/runs`, `implementation` is an optional filter for any role.
- **A-5** `/runs`: `limit` default 50, max 500. `include_payload` default `false`; when `true`, max `limit` is 50. `from`/`to` are UTC timestamps filtering on `as_of`. `cursor` is opaque (encode `as_of` + `run_id`).
- **A-6** `/latest` with no successful run for that scope → `404` with problem type `no-output-yet`, distinct from unknown slot (`404`, `unknown-slot`). `/latest` on a retired slot → `410`, `slot-retired`; its `/runs` and `/runs/{run_id}` keep working.

### 3.1 Models come and go

The set of slots and implementations changes without API deploys. Adding, swapping, shadowing or retiring a model is a registry change in `models/` (MODELS §5); the API MUST pick it up from the database alone.

- **A-27** No slot ID, implementation ID or payload kind is compiled into the server. Routes are generic over `{slot_id}`; validation (A-3, A-20) uses the in-memory slot catalog.
- **A-28** The slot catalog is loaded from the `slots` table at startup (`/ready` is `503` until it loads), reloaded on `NOTIFY slots` (CONTRACTS C-13a), and also reloaded every 60 s in case a notification is missed. A failed reload keeps the previous catalog and logs an error.
- **A-29** "Latest production" follows CONTRACTS C-11, whatever the implementation. After a promotion, the old implementation's last output is served (and goes stale on schedule) until the new one produces a successful run; after a rollback, the reverse. `meta` and the envelope always name the implementation that produced what's shown.
- **A-30** Cache keys include the implementation for shadow lookups: `(slot_id, scope_key, role, implementation | "*")`. A catalog reload that changes a slot's implementations clears that slot's cache entries.
- **A-31** Envelopes carry `payload_kind` as a string and the payload as opaque JSON (A-21), so a new slot or implementation never needs Rust changes.

## 4. Response format

`/latest` and `/runs/{run_id}` return the stored envelope unchanged plus a `meta` block:

```json
{
  "data": { "...envelope exactly as stored..." },
  "meta": {
    "api_version": "1",
    "role": "production",
    "age_seconds": 412,
    "max_age_seconds": 900,
    "stale": false,
    "served_at": "2026-10-13T17:06:52Z"
  }
}
```

- **A-7** `data` MUST be byte-for-byte semantically identical to the stored `envelope` JSONB. The API does not reshape payloads. That keeps Python as the only producer and the contract test (CONTRACTS §7) meaningful.
- **A-8** `age_seconds = served_at − as_of`; `stale = age_seconds > max_age_seconds`. Stale outputs are still served with `200`; deciding what to show is the client's job.
- **A-9** List endpoints return `{"data": [...], "meta": {..., "next_cursor": "…" | null}}`.
- **A-10** Errors are `application/problem+json`:
  ```json
  { "type": "https://<host>/problems/unknown-slot", "title": "Unknown slot", "status": 404, "detail": "Slot 'vol_surfce' does not exist", "request_id": "…" }
  ```
  No stack traces, SQL or internal paths in any error. `500`s return a generic detail plus `request_id`; the real error goes to logs.

## 5. Caching

- **A-11** `/latest` responses MUST set `ETag` = `"<run_id>"` and honor `If-None-Match` with `304`. The frontend polls with this, so most polls cost almost nothing.
- **A-12** `Cache-Control: public, max-age=N` where `N = min(30, seconds until the output turns stale)`, floored at 0. Run-by-ID responses are immutable: `max-age=31536000, immutable`.
- **A-13** The server SHOULD keep an in-process cache of latest envelopes keyed by `(slot_id, scope_key, role)`, invalidated by Postgres `LISTEN model_runs` (CONTRACTS C-13), with a TTL fallback of 10 s in case notifications are missed. If the listener connection drops, the server MUST reconnect and clear the cache.
- **A-14** Use `moka` or an `Arc<RwLock<HashMap>>` from `tokio::sync`. Never hold a `std::sync` lock across `.await`.

## 6. Non-functional requirements

| Requirement | Target |
|---|---|
| `/latest` latency, cache hit | p95 < 20 ms server-side |
| `/latest` latency, cache miss | p95 < 75 ms |
| `/runs` (limit 50, no payload) | p95 < 150 ms |
| Request timeout | 5 s (`TimeoutLayer`) → `503` problem |
| DB pool | Configurable; default 10 connections, acquire timeout 2 s |
| Request body limit | 16 KB (GET only; protects against abuse) |
| Rate limit | Per IP, default 120 req/min with burst 30, configurable; `429` with `Retry-After` |
| Compression | gzip and br for responses > 1 KB |
| Graceful shutdown | On SIGTERM, stop accepting, drain in-flight requests up to 10 s |
| Startup | Fail fast with a clear log line if config is invalid or DB is unreachable after retries |

## 7. Security (no API key in v1)

No authentication is an accepted decision: this is a research tool (README §8 Q1). The controls below still apply.


- **A-15** CORS: explicit origin allowlist from config (the frontend's origin(s)). No `*` in production. Methods: GET, HEAD, OPTIONS.
- **A-16** Response headers on all responses: `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'` (API serves JSON only).
- **A-17** **Auth seam.** Routes are mounted under a router with an auth middleware layer that is a no-op in v1 (`AUTH_MODE=none`). Adding API keys or OAuth later MUST be a middleware implementation plus config, not a route change.
- **A-18** Deploy behind a reverse proxy or load balancer that terminates TLS. The API trusts `X-Forwarded-For` only from configured proxy CIDRs; otherwise the rate limiter keys on the socket address.
- **A-19** Network restriction (VPN or IP allowlist) is not required for v1. Revisit it if the user base moves beyond research.
- **A-20** Inputs are only path/query strings. Validate `slot_id` against the slot regex and existence before any query. All SQL through `sqlx` parameter binding; no string-built SQL.

## 8. Code structure

Follows the `State` + `FromRef` pattern from the reference snippet.

```
api/
├── Cargo.toml
├── .sqlx/                     # offline query metadata
├── src/
│   ├── main.rs                # composition root only: config → state → router → serve
│   ├── lib.rs
│   ├── config.rs              # env-driven, validated at startup
│   ├── state.rs               # AppState + FromRef impls, payload schemas from contracts/
│   ├── error.rs               # AppError → problem+json
│   ├── catalog.rs             # hot-reloading slot catalog (A-27, A-28)
│   ├── cache.rs               # latest-envelope cache + LISTEN model_runs/slots task
│   ├── domain/                # Slot, StoredRun (payload as serde_json::Value), RunMeta, scope_key
│   ├── repo/
│   │   ├── mod.rs             # trait RunRepository, trait SlotRepository
│   │   ├── postgres.rs
│   │   └── memory.rs          # test-only in-memory repos
│   ├── routes/
│   │   ├── mod.rs             # router assembly, nesting under /v1, middleware stack, OpenAPI
│   │   ├── params.rs          # query splitting, identifier and timestamp checks
│   │   ├── health.rs
│   │   ├── slots.rs
│   │   ├── runs.rs
│   │   ├── schemas.rs
│   │   └── tests.rs           # handler + lifecycle tests on in-memory repos (A-23)
│   └── middleware/
│       ├── auth.rs            # no-op in v1 (A-17)
│       ├── headers.rs
│       └── problem.rs         # every error → problem+json with request_id
└── tests/                     # integration tests against real Postgres (#[ignore]; run with --ignored)
```

```rust
#[derive(Clone)]
pub struct AppState {
    pub runs: Arc<dyn RunRepository>,     // trait object → mock in unit tests
    pub slots: Arc<dyn SlotRepository>,
    pub cache: LatestCache,               // internally Arc
    pub config: Arc<Config>,
}
// FromRef impls for each field so handlers extract only what they need.
```

- **A-21** Payloads are carried as `serde_json::Value` inside a typed envelope. The API validates the envelope fields it uses (IDs, timestamps, status) and passes the payload through. Payload typing lives in `contracts/` and Python; duplicating it in Rust would create a second source of truth.
- **A-22** No `.unwrap()`/`.expect()` outside `main.rs` startup and tests. Startup failures log a clear message and exit non-zero.

## 9. Corrections to the reference Axum snippet

The pattern in the reference (`AppState` + `FromRef`, nested routers, repository traits behind `Arc<dyn …>`, `tokio::sync` locks, composition root in `main`) is the right foundation. Fix these details when using it:

1. **Route syntax.** `/:id` is Axum 0.7. Axum 0.8 uses `/{id}`; `/:id` panics at router construction in 0.8.
2. **Missing `with_state` is a compile error, not a startup panic.** `axum::serve` requires a `Router<()>`; a router still missing its state won't type-check. That's better than the snippet claims.
3. **`sqlx::query_as!` needs a database at compile time** unless offline mode is set up. Commit `.sqlx/` and set `SQLX_OFFLINE=true` in CI.
4. **`async_trait` is still needed** for repository traits used as `Arc<dyn Trait>`. Native `async fn` in traits isn't object-safe.
5. **Don't `.unwrap()` env vars and DB connect in `main`**; validate config into a typed struct and exit with a readable error (A-22).

## 10. Testing

- **A-23** Unit tests for handlers using mock repositories (no DB).
- **A-24** Integration tests spin up Postgres (testcontainers or CI service), apply `contracts/db/migrations`, seed `contracts/fixtures/valid`, and assert every endpoint: success shapes, every documented error, ETag/304, staleness flag flipping at `max_age_seconds`, pagination, shadow role, 405 on POST.
- **A-25** Contract test: for each valid fixture, `GET /v1/runs/{run_id}` returns `data` equal to the fixture (CONTRACTS §7).
- **A-26** A load test (e.g. `oha` or `k6`) in CI or pre-release verifies the §6 latency targets on a seeded DB.

## 11. Acceptance (M2)

- [ ] All §3 endpoints implemented and documented in `/openapi.json`.
- [ ] All A-24 integration tests green.
- [ ] Non-GET methods return 405; DB role is `api_ro`.
- [ ] Latency targets met under load test.
- [ ] Killing the DB → `/ready` 503, `/health` 200, cached `/latest` still served until TTL.
- [ ] Structured logs include `request_id` and, for data responses, `run_id`.
