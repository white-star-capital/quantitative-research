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
| `src/api/` | Types mirroring `contracts/schemas`, runtime validator (F-5), `fetch` client, 60 s polling hook |
| `src/format/` | Unit-driven formatting (F-15/F-16) |
| `src/views/` | Catalog, slot view, §5 states, per-panel error boundary |
| `src/renderers/` | One renderer per payload kind (M3: all currently `Pending`), SVG chart primitives |

## Status

This is scaffolding. The shell, routing, polling, validation, formatting and states are done. Still to do in M3:
- the six payload renderers;
- the crosshair and tooltip;
- catalog freshness per slot (F-6);
- history (F-13) and export (F-14);
- payload-level validation, so that invalid payload fixtures land in Unsupported.

No sample data exists anywhere. Until the API serves real runs, the UI shows its empty and error states.

The CSP and caching headers (F-22, design §9.1/§9.3) are served by the deploy proxy, which is set up in M4 once the deployment target is chosen.
