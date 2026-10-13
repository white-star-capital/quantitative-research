# frontend/

Terminal UI for the options terminal. Spec: [`../FRONTEND.md`](../FRONTEND.md). Design: [`../front_end_design.md`](../front_end_design.md), which takes precedence over every frontend design choice.

```bash
npm ci                 # never npm install, except when deliberately changing a dependency
npm run dev            # VITE_API_BASE_URL=http://localhost:8080 npm run dev
npm test               # node --test (built-in runner, type stripping)
npm run build          # typecheck + vite build
npm run check:deps     # dependency policy gate
npm audit --omit=dev
```

## Dependencies

This project follows [`../building_secure_front_ends.md`](../building_secure_front_ends.md). The approved set is the baseline, with no exceptions:

| Package | Category |
|---|---|
| `react`, `react-dom` | runtime |
| `typescript`, `vite`, `@vitejs/plugin-react`, `@types/react`, `@types/react-dom` | dev |

All versions are exact (`.npmrc` `save-exact=true`), and `package-lock.json` is committed. `npm run check:deps` fails on anything else. Approved exceptions go in this table and in `EXCEPTIONS` in `scripts/check-deps.mjs`.

Not dependencies:
- IBM Plex Mono woff2 files in `public/fonts/` (SIL OFL, `public/fonts/OFL.txt`) were extracted once from the `@ibm/plex-mono@2.5.0` npm tarball.
- `src/node-test.d.ts` is a hand-written type shim for `node:test`/`node:assert`/`node:fs`, in place of `@types/node`.

## Layout

| Path | Purpose |
|---|---|
| `src/app/` | Desk shell (`App.tsx`), hash router with deep-link validation |
| `src/api/` | Types mirroring `contracts/schemas`, runtime envelope + payload checks (F-5), `fetch` client, 60 s polling hook |
| `src/format/` | Unit-driven formatting (F-15/F-16) and the pricer conventions (guide §7.8) |
| `src/renderers/` | One generic renderer per payload kind (tiles, line chart, heatmap, sortable table, composite) on hand-drawn SVG primitives in `chart/` |
| `src/views/` | Catalog, slot view, §5 states, per-panel error boundary, bespoke-view registry |
| `src/views/bs_pricer/` | `bs_pricer` bespoke view: cross-asset monitor, asset view with Chain / Vol / Pricer & OTC / Data |

## Model lifecycle in the UI

- **Shadow period:** during a slot's initial shadow period (no production model), views read the first shadow implementation and show a red SHADOW banner.
- **Bespoke views:** these are additive and keyed by the slot's `display.default_view`. GENERIC/BESPOKE in the title bar switches to the generic renderers at any time (FRONTEND §1).
- **Run pinning:** the pricer evaluates against the run on screen (`run_id`), so its numbers always match the chain being viewed.

## Privacy

The counterparty label, cost basis and blotter exist only in component state and the CSV the user exports; they are never sent (P-8). Per-viewer preferences (VRP gate, chain preset, band thresholds) use `localStorage`, wrapped in try/catch, and are conveniences only.

## Charts and the design system

Amber and cyan are the only series colours (`front_end_design.md` §2). A third or fourth series reuses them with dashes. The dataviz validator flags their lightness as outside its generic band. The design doc says this is deliberate for the terminal look, and its precedence rule wins; CVD separation passes (ΔE 22.3).

## Status

- **M3 renderers and the `bs_pricer` views are done.** They were verified end to end (runner → Postgres → API → UI) on a synthetic run.
- **Still open:** catalog freshness per slot (F-6), history (F-13), and DOM/visual/E2E test tooling. That tooling is an exception request under the dependency policy; until then the E2E flow runs from a scratch Playwright install outside the repo.

The CSP and caching headers (F-22, design §9.1/§9.3) are served by the deploy proxy, which is set up in M4 once the deployment target is chosen.
