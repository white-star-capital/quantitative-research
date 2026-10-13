# Contracts — `contracts/`

The data contract shared by `models/`, `api/` and `frontend/`. If two components disagree, this document wins. Changes here go through a PR reviewed by an owner of each consuming component.

## 1. Folder layout

```
contracts/
├── schemas/
│   ├── envelope/v1.json            # run envelope (JSON Schema 2020-12)
│   └── payloads/
│       ├── scalar_set/v1.json
│       ├── curve/v1.json
│       ├── surface/v1.json
│       ├── timeseries/v1.json
│       ├── table/v1.json
│       └── composite/v1.json
├── slots/
│   └── <slot_id>.yaml              # one file per public slot
├── db/
│   └── migrations/                 # ordered SQL migrations (single owner: contracts)
├── fixtures/
│   ├── valid/<kind>/*.json         # golden fixtures, full envelopes
│   └── invalid/<kind>/*.json       # each fixture violates exactly one rule; filename says which
└── README.md
```

- **C-1** JSON Schema (draft 2020-12) files in `contracts/schemas/` MUST be the single source of truth. Python Pydantic models, Rust types and TypeScript types MUST either be generated from these schemas or be contract-tested against the fixtures (§7).
- **C-2** Schemas are versioned by path. A breaking change creates a new version file (`v2.json`); `v1` stays until no slot uses it.

## 2. Output envelope

Every model run produces exactly one envelope. The API returns envelopes as stored, plus a `meta` block (`API.md` §4).

```json
{
  "envelope_version": "1",
  "run_id": "0192f3a0-6b1e-7c3a-9a51-2f8e4d1c7b20",
  "slot_id": "vol_surface",
  "implementation_id": "vol_surface_svi",
  "implementation_version": "2.1.0",
  "scope": { "underlying": "BTC" },
  "status": "success",
  "as_of": "2026-10-13T17:00:00Z",
  "started_at": "2026-10-13T17:00:02Z",
  "finished_at": "2026-10-13T17:00:09Z",
  "inputs": {
    "sources": [{ "name": "options_chain", "snapshot_at": "2026-10-13T17:00:00Z" }],
    "hash": "sha256:9f2c…"
  },
  "code": { "git_sha": "a1b2c3d" },
  "warnings": [],
  "payload_kind": "surface",
  "payload_version": "1",
  "payload": { }
}
```

| Field | Type | Rule |
|---|---|---|
| `envelope_version` | string | `"1"` for this spec |
| `run_id` | UUIDv7 string | Time-ordered; unique per run |
| `slot_id` | string | `^[a-z][a-z0-9_]{2,63}$`; must exist in `contracts/slots/` |
| `implementation_id` | string | Same pattern as `slot_id` |
| `implementation_version` | semver string | From the implementation manifest |
| `scope` | object of string → string | Keys must match the slot's declared scope dimensions exactly |
| `status` | `"success"` \| `"failed"` | Failed runs MUST have `payload: null` and a non-empty `error` |
| `as_of` | UTC timestamp | The market time the output describes. **Not** the run time. |
| `started_at`, `finished_at` | UTC timestamp | Run wall-clock times |
| `inputs.hash` | string | `sha256:` over canonicalized inputs; same inputs → same hash |
| `code.git_sha` | string | Commit the runner executed |
| `warnings` | array of `{code, message}` | Non-fatal sanity-check results (e.g. butterfly arbitrage flagged) |
| `payload_kind` | enum | One of §3 |
| `payload_version` | string | Version of the payload schema |
| `payload` | object \| null | Validates against `schemas/payloads/<kind>/v<version>.json` |
| `error` | object | Present only when `status = failed`: `{code, message}`; no stack traces |

## 3. Payload kinds

Payload kinds are what make the frontend generic. A renderer exists per kind, so any slot whose payload is a known kind renders without frontend work.

Common building blocks:

```json
"unit": { "quantity": "implied_vol", "scale": "decimal" }
```

`quantity` is from a controlled vocabulary (`implied_vol`, `realized_vol`, `price`, `price_usd`, `delta`, `gamma`, `vega`, `theta`, `rate`, `ratio`, `bps`, `strike`, `moneyness_log`, `moneyness_simple`, `delta_bucket`, `tenor_days`, `expiry_date`, `timestamp`, `count`, `other`). `scale` is one of `decimal`, `percent`, `bps`, `absolute`. Adding a vocabulary term is a contract PR.

Greeks MUST state their bump convention in `unit.convention` (e.g. vega `"per_1vol_point"`, theta `"per_calendar_day"`).

| Kind | Use for | Shape |
|---|---|---|
| `scalar_set` | Headline metrics: ATM IV, 25Δ risk reversal, butterfly, forward | `items: [{key, label, value, unit}]` |
| `curve` | Smile (IV vs strike/delta), term structure (IV vs tenor), any y(x) | `x: {label, unit, values[]}`, `series: [{key, label, unit, values[]}]`, every `values` array the same length as `x.values` |
| `surface` | Vol surface, any z(x, y) grid | `x: {label, unit, values[]}`, `y: {label, unit, values[]}`, `z: {label, unit, values[[...]]}` with `z.values` of shape `len(y) × len(x)`, row-major by y |
| `timeseries` | Realized vs implied history, metric history | `t: [UTC timestamps, strictly increasing]`, `series: [{key, label, unit, values[]}]` |
| `table` | Chains, greeks tables, ranked lists | `columns: [{key, label, type, unit?}]`, `rows: [[...]]`; `type` ∈ `number`, `string`, `timestamp`, `date` |
| `composite` | One run that produces several views (e.g. surface + smile + metrics) | `sections: [{key, label, payload_kind, payload_version, payload}]`; nesting depth 1 (no composite inside composite) |

Rules for all kinds:
- **C-3** Missing values are `null` (README X-3). The frontend renders `null` as a gap, never as zero.
- **C-4** Array length and shape constraints above are enforced by the schema or by the runner's validation (JSON Schema can't express every cross-field length rule; the runner MUST check the rest).
- **C-5** Keys (`key` fields) are stable identifiers; labels are display text and MAY change between versions.
- **C-6** Payload size SHOULD stay under 2 MB uncompressed. Over 5 MB is rejected by the runner. If a slot legitimately needs more, it needs a paging design, not a bigger limit.

## 4. Slot definitions

`contracts/slots/<slot_id>.yaml`:

```yaml
slot_id: vol_surface
title: Implied volatility surface
description: >
  Fitted implied vol by log-moneyness and tenor.
payload_kind: surface          # or composite; fixed for the life of the slot's major version
payload_version: "1"
scope:
  dimensions: [underlying]     # API query params; envelope.scope keys
  allowed:
    underlying: [BTC, ETH]     # optional allowlist
max_age_seconds: 900           # older than this → stale
conventions: >
  x = log-moneyness ln(K/F), y = tenor in calendar days, z = implied vol (decimal).
display:
  default_view: heatmap        # renderer hint; frontend MAY ignore
  order: 10                    # catalog ordering
```

- **C-7** A slot's `payload_kind`, `payload_version` and `scope.dimensions` are its public contract. Changing any of them is a breaking change: create a new slot (`vol_surface_v2`) or bump the API version. Never mutate them in place.
- **C-8** `max_age_seconds` is set per slot and drives staleness everywhere.

## 5. Candidate slots (confirm in README §8 Q3)

Illustrative, to size the framework. Not a commitment.

| Slot | Kind | Scope | Source |
|---|---|---|---|
| `vol_surface` | `surface` | underlying | Deribit |
| `smile` | `curve` | underlying, expiry | Deribit |
| `term_structure` | `curve` | underlying | Deribit |
| `skew_metrics` | `scalar_set` | underlying | Deribit |
| `realized_vs_implied` | `timeseries` | underlying | Hyperliquid (realized), Deribit (implied) |
| `greeks_chain` | `table` | underlying, expiry | Deribit |
| `vol_dashboard` | `composite` | underlying | Both |

## 6. Database schema

Postgres. Migrations live in `contracts/db/migrations/` and are the only way the schema changes.

```sql
CREATE TABLE slots (
    slot_id            TEXT PRIMARY KEY,
    title              TEXT NOT NULL,
    description        TEXT NOT NULL,
    payload_kind       TEXT NOT NULL,
    payload_version    TEXT NOT NULL,
    scope_dimensions   TEXT[] NOT NULL,
    max_age_seconds    INTEGER NOT NULL CHECK (max_age_seconds > 0),
    definition         JSONB NOT NULL,          -- full slot YAML as JSON
    production_impl    TEXT,                    -- from registry.yaml
    production_version TEXT,
    status             TEXT NOT NULL CHECK (status IN ('active','retired')),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE model_runs (
    run_id                 UUID PRIMARY KEY,
    slot_id                TEXT NOT NULL REFERENCES slots(slot_id),
    implementation_id      TEXT NOT NULL,
    implementation_version TEXT NOT NULL,
    role                   TEXT NOT NULL CHECK (role IN ('production','shadow')),
    scope_key              TEXT NOT NULL,       -- canonical, e.g. 'underlying=BTC'
    scope                  JSONB NOT NULL,
    status                 TEXT NOT NULL CHECK (status IN ('success','failed')),
    as_of                  TIMESTAMPTZ NOT NULL,
    started_at             TIMESTAMPTZ NOT NULL,
    finished_at            TIMESTAMPTZ NOT NULL,
    input_hash             TEXT NOT NULL,
    git_sha                TEXT NOT NULL,
    envelope               JSONB NOT NULL       -- full envelope as served
);

CREATE INDEX model_runs_latest
    ON model_runs (slot_id, scope_key, role, status, as_of DESC);
```

- **C-9** `scope_key` is canonical: keys sorted, `k=v` joined by `&`, values URL-encoded. Python and Rust MUST share a test vector set for this function.
- **C-10** Rows are append-only. No updates or deletes except by the retention job (README §8 Q6).
- **C-11** "Latest" for a slot and scope = the most recent `as_of` with `role = 'production'` and `status = 'success'`. Failed runs never become latest.
- **C-12** Two DB roles: `runner_rw` (INSERT on `model_runs`, UPSERT on `slots`) and `api_ro` (SELECT only). The API MUST connect as `api_ro`.
- **C-13** The runner SHOULD `NOTIFY model_runs, '<slot_id>'` after each successful insert so the API can invalidate its cache.

## 7. Contract tests (required in CI)

| Test | Where | Asserts |
|---|---|---|
| Schema self-check | `contracts/` | Every valid fixture passes; every invalid fixture fails for the reason in its filename |
| Python round-trip | `models/` | Every valid fixture parses into the Pydantic types and re-serializes to equivalent JSON |
| Python output | `models/` | Every implementation's test output validates against its slot's schema |
| Rust deserialize | `api/` | Every valid fixture deserializes into the API's envelope type; served JSON equals stored JSON |
| Frontend render | `frontend/` | Every valid fixture renders without error; invalid payloads render the "unsupported" state |
| `scope_key` vectors | `models/` + `api/` | Both implementations produce identical keys for the shared vector file |
