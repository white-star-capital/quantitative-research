# Frontend — `frontend/`

The terminal UI. It reads from the API and renders model outputs as charts, tables and figures.

## 0. Governing documents

Two documents in the repo are **binding** for the frontend and take precedence over this file on anything they cover:

- `front_end_design.md`: visual design, layout, components, typography, color, interaction patterns.
- `building_secure_front_ends.md`: frontend security practices.

This file defines the **functional** requirements and the integration with the API. It does not redefine design or security rules. Where this file states a security minimum (§6), it is a floor: if `building_secure_front_ends.md` is stricter, it wins. If the two documents conflict with this one, raise it before building rather than picking one silently.

> These requirements were written without reading the two governing documents. Before M3 starts, the frontend lead MUST reconcile §2 (stack), §4 (views) and §6 (security) against them and note any conflicts in this file.

## 1. Principles

1. **Render by payload kind, not by slot.** One renderer per kind (`CONTRACTS.md` §3). A new slot with a known kind appears in the terminal with zero frontend code.
2. **Never present stale data as fresh.** Freshness is visible on every output.
3. **The API is untrusted input.** Every string from the API is rendered as text.
4. **Units come from the data.** Formatting is driven by `unit`, never by assumptions about a slot.

## 2. Stack

If `front_end_design.md` specifies the framework, component library or charting library, use those. If it does not, the recommendation is:

- TypeScript in strict mode.
- API types generated from `/openapi.json` and payload types generated from `contracts/schemas/` (e.g. `json-schema-to-typescript`). No hand-written duplicates of contract types.
- A charting library that supports heatmaps, 3D surfaces, multi-series line charts and time axes, and that can be themed to `front_end_design.md`. Plotly.js, ECharts and Observable Plot all qualify; choose one for consistency.
- Lockfile committed.

## 3. API integration

- **F-1** The API base URL comes from build-time config. No hard-coded hosts.
- **F-2** The catalog comes from `GET /v1/slots`. Never hard-code the slot list; ordering uses `display.order`.
- **F-3** Polling: each mounted slot view polls `/latest` every `min(30 s, max_age_seconds / 4)` with `If-None-Match`, so unchanged data returns `304`. Polling pauses when the browser tab is hidden and resumes (with an immediate fetch) when visible.
- **F-4** Network and `5xx` errors: keep showing the last good data, show a non-blocking error indicator, retry with exponential backoff (cap 60 s). `429`: respect `Retry-After`.
- **F-5** Validate API responses at runtime against the generated types (e.g. a schema validator on the envelope). Anything that fails validation goes to the "unsupported" state (§5), not into a chart.

## 4. Views

### 4.1 Catalog

- **F-6** Lists active slots with title, description, current implementation and version, and freshness for the default scope.
- **F-7** Filter/search by title. Click opens the slot view.

### 4.2 Slot view (`/slots/{slot_id}?underlying=BTC&…`)

- **F-8** Scope selectors are generated from the slot's `scope.dimensions` and `allowed` values. Selection is reflected in the URL so views are deep-linkable and shareable.
- **F-9** The header MUST show: slot title, `as_of` (in UTC, with the user's local time on hover), age ("6m ago"), implementation and version, and a freshness badge (fresh / stale). Stale is driven by `meta.stale` from the API, not computed in the browser.
- **F-10** Warnings from the envelope (`warnings[]`, e.g. butterfly arbitrage flagged) MUST be visible, for example as an icon with the messages listed on hover or expand.
- **F-11** Body renders the payload with the renderer for `payload_kind` (§4.3).
- **F-12** A footer or info panel shows `run_id`, `code.git_sha`, `inputs` sources and `snapshot_at`. This is the audit trail users and the desk need when a number looks wrong.
- **F-13** History: a control to load recent runs (`/runs`) and step back through them, or (for `scalar_set`) plot a field over time. SHOULD for v1.
- **F-14** Export: download the current payload as JSON, and as CSV for `curve`, `timeseries`, `table` and `scalar_set`. SHOULD for v1.

### 4.3 Renderers

| Kind | Required rendering | Notes |
|---|---|---|
| `scalar_set` | Metric tiles: label, formatted value, unit | `null` → "—" |
| `curve` | Line chart, one line per series, x per `x.unit` | Smile: x may be strike, log-moneyness or delta bucket, labelled from the payload |
| `surface` | Heatmap (default) with color scale legend; 3D surface as an optional toggle | `display.default_view` is a hint; `null` cells are blank, not interpolated |
| `timeseries` | Time-axis line chart, multi-series, zoom/pan | x axis in UTC by default |
| `table` | Sortable table, numeric columns right-aligned and formatted per unit | Large tables virtualized (> 500 rows) |
| `composite` | Sections in order, each with its own renderer and section label | Layout per `front_end_design.md` |

- **F-15** Formatting is unit-driven. `implied_vol` with `scale: decimal` displays as percent with configurable decimals (0.5523 → 55.23%). Greeks show their `convention` in the axis or column header (e.g. "Vega (per 1 vol pt)").
- **F-16** `null` values render as gaps in lines, blank cells in heatmaps and "—" in tables. Never zero.
- **F-17** Charts show hover tooltips with exact values and units, and axis labels from the payload.
- **F-18** Every renderer is a pure function of `(payload, unit metadata, theme)`. No renderer knows which slot it is drawing.

## 5. States

Every slot view MUST handle each state explicitly; the design for each comes from `front_end_design.md`.

| State | Trigger | Behavior |
|---|---|---|
| Loading | First fetch in flight | Skeleton, no spinner over stale data |
| Fresh | `meta.stale = false` | Normal render |
| Stale | `meta.stale = true` | Normal render + prominent stale badge with age |
| No output yet | `404 no-output-yet` | Empty state explaining the slot hasn't produced data for this scope |
| Unknown slot | `404 unknown-slot` | Not-found page linking back to catalog |
| Invalid scope | `400` | Show the API's message; reset to default scope |
| Unsupported | Unknown `payload_kind`/`payload_version`, or response fails validation (F-5) | Non-crashing panel: "This output can't be displayed in this version of the terminal," with JSON download |
| Error | Network / `5xx` | Keep last good data, show error indicator (F-4) |

- **F-19** A rendering exception in one panel MUST NOT take down the page (error boundary per panel).

## 6. Security minimums

`building_secure_front_ends.md` governs. At minimum:

- **F-20** No secrets, tokens or credentials in the bundle or in source. (There are none in v1; keep it that way when auth arrives by using a proper auth flow, not an embedded key.)
- **F-21** Render all API-sourced strings (labels, descriptions, warning messages, error details) as text. No `innerHTML`/`dangerouslySetInnerHTML` or equivalent with API data, including chart tooltip templates that accept HTML.
- **F-22** Strict Content Security Policy served with the app: no `unsafe-inline` scripts, no `unsafe-eval`; `connect-src` limited to the API origin.
- **F-23** Dependencies pinned and scanned in CI (`npm audit` or equivalent); no runtime loading of scripts from third-party CDNs unless `building_secure_front_ends.md` allows it with SRI.
- **F-24** Deep-link params (`slot_id`, scope values) are validated against the catalog before use and never interpolated into HTML or URLs unencoded.

## 7. Non-functional

- **F-25** Initial load (catalog) interactive in < 2 s on a typical office connection; slot view first render < 1 s after data arrives.
- **F-26** A 100 × 50 surface heatmap and a 5,000-point timeseries render without visible jank.
- **F-27** Accessibility: keyboard navigation for catalog and selectors, chart color scales distinguishable for common color-vision deficiencies, text contrast per `front_end_design.md`.
- **F-28** Supported browsers: current and previous major version of Chrome, Edge, Firefox, Safari.

## 8. Testing

- **F-29** Unit tests for unit formatting (every `quantity` × `scale`), null handling and scope-key/URL round-trip.
- **F-30** Renderer tests: every valid fixture in `contracts/fixtures/valid/` renders without error; every invalid fixture lands in the Unsupported state (CONTRACTS §7).
- **F-31** A mock API (e.g. MSW) serving `contracts/fixtures/` lets M3 proceed before the real API exists, and is used for component and E2E tests.
- **F-32** Visual regression snapshots for each renderer and each state in §5.
- **F-33** E2E (Playwright or equivalent) at M4 against the real stack: catalog → slot view → change scope → stale badge appears when a seeded run ages past `max_age_seconds`.

## 9. Acceptance (M3)

- [ ] Governing documents reconciled with this file; conflicts recorded (§0).
- [ ] Catalog and slot view working against the mock API.
- [ ] One renderer per payload kind; all fixtures render (F-30).
- [ ] All §5 states implemented and snapshot-tested.
- [ ] Security minimums (§6) and `building_secure_front_ends.md` checklist signed off.
- [ ] Design review against `front_end_design.md` signed off.