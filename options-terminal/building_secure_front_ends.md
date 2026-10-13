# Frontend Build Standards: Minimal Dependency Policy

**Status:** Mandatory
**Applies to:** All new frontend projects and greenfield rewrites
**Scope:** Client-side React/TypeScript applications

This document uses MUST / MUST NOT / SHOULD / MAY per standard engineering convention.
MUST requirements are enforced in CI and code review and are not subject to individual
developer discretion. Exceptions require sign-off per Section 6.

---

## 1. Rationale

Every dependency in `package.json` — direct or transitive — is code that runs with full
access to the build machine and, once bundled, to the end user's browser. npm's install-script
mechanism means a single compromised package can execute arbitrary code on a developer's
machine or CI runner at install time, not just at runtime. The standard mitigation is not
"choose reputable packages" — reputable packages get compromised too, on a regular basis.
The mitigation is minimizing the number of packages in the tree, full stop, and treating
every addition as a deliberate, reviewed decision rather than a convenience.

This policy trades ecosystem convenience for a dependency tree small enough that a human can
actually reason about everything in it.

---

## 2. Approved baseline dependency set

Every project MUST start from exactly this set, at these categories, and MUST NOT exceed it
without an approved exception (Section 6):

| Package | Category | Purpose |
|---|---|---|
| `react` | runtime | UI runtime |
| `react-dom` | runtime | DOM renderer |
| `typescript` | dev | type checking / transpilation |
| `vite` | dev | build tool / dev server |
| `@vitejs/plugin-react` | dev | JSX/Fast Refresh support in Vite |
| `@types/react` | dev | type definitions |
| `@types/react-dom` | dev | type definitions |

No other package is approved by default. This includes, without limitation:

- **State management libraries** (Redux, Zustand, Jotai, MobX, Recoil) — use React's built-in
  `useState`/`useReducer`/`useContext`. If a project's state complexity genuinely exceeds
  what Context can handle cleanly, that is an exception request, not a default.
- **UI component kits** (MUI, Chakra, Ant Design, shadcn/ui) — build components in-house with
  plain CSS or CSS Modules.
- **Icon packs** (lucide-react, react-icons, FontAwesome) — use inline SVG. An icon is a
  handful of SVG paths; it does not need a package with hundreds of exports you'll use three of.
- **Charting libraries** (recharts, chart.js, d3, victory) — hand-roll SVG for the specific
  chart types the product needs. See prior dashboard build for a working pattern
  (`src/components/Chart.tsx`: axis, gridlines, bar/line rendering in ~80 lines, zero deps).
- **HTTP clients** (axios, ky, swr, react-query) — use native `fetch`. Wrap it in a small
  typed hook per data source if you need loading/error state, as with
  `useSupabaseTable.ts`.
- **CSS-in-JS runtimes** (styled-components, emotion) — use plain CSS files or CSS Modules,
  which Vite supports natively with zero additional dependencies.
- **Utility libraries** (lodash, ramda, date-fns, dayjs) — modern JS/TS and the `Intl` and
  `Date` built-ins cover the overwhelming majority of what these libraries are reached for.
  Write the one function you need.
- **Form libraries** (react-hook-form, formik) — controlled inputs with `useState` are
  sufficient for the form complexity most internal tools require.

**Build tool constraint:** Vite is the only approved build tool. Create React App MUST NOT be
used (deprecated, unmaintained, ~1,300+ transitive dependencies). Next.js MUST NOT be used
for projects that do not require server-side rendering or a server runtime — it pulls in a
build/runtime surface disproportionate to a client-side dashboard or internal tool, and its
dependency tree and attack surface are not justified by "might want SSR later."

---

## 3. Version pinning

- Every `package.json` MUST set `save-exact=true` in `.npmrc`. No `^` or `~` ranges anywhere,
  direct or introduced later. Exact versions only.
- `package-lock.json` MUST be committed. It is not a build artifact — it is the source of
  truth for exactly what code runs.
- Local and CI installs MUST use `npm ci`, never `npm install`, except at the moment of
  deliberately adding or updating a dependency. `npm ci` refuses to deviate from the
  lockfile; `npm install` will silently resolve a slightly different tree if anything
  upstream has moved.

---

## 4. Required CI checks

Every repository under this policy MUST run the following in CI, blocking merge on failure:

1. **`npm ci`** — not `npm install`. A PR that requires `npm install` to succeed because the
   lockfile is stale MUST fail.
2. **Dependency count check** — CI MUST fail if `dependencies` (not `devDependencies`) in
   `package.json` exceeds the approved baseline (`react`, `react-dom`) unless the PR includes
   an approved exception reference (Section 6) in the PR description.
3. **`npm audit --omit=dev`** — MUST run on every PR that touches `package.json` or
   `package-lock.json`. Any high/critical finding blocks merge until resolved or explicitly
   waived by the reviewer in Section 6's process.
4. **Lockfile diff review** — any PR that changes `package-lock.json` MUST be reviewed by a
   second engineer specifically for the diff to that file, not just the application code.
   A one-line `package.json` change that produces a 500-line lockfile diff is a signal to
   look closely at what just got pulled in transitively.

---

## 5. Required project scaffolding

Every new project MUST include, from first commit:

- `.npmrc` with `save-exact=true`
- `.gitignore` excluding `node_modules`, `dist`, `.env`, `*.local`
- `tsconfig.json` with `"strict": true` — no project ships with strict mode off
- A `README.md` section documenting the approved dependency list and this policy by reference

---

## 6. Exception process

A dependency outside the baseline set MAY be added only when all of the following are met:

1. **Written justification** in the PR description: what problem it solves, and why it cannot
   reasonably be solved with the tools already in the approved set or with a small amount of
   in-house code.
2. **Dependency audit**, included in the PR description:
   - `npm info <package> dependencies` — direct dependency count of the candidate package
   - Approximate transitive tree size it would add (`npm ls <package>` after a trial install,
     or equivalent)
   - Last publish date and maintainer activity (a package last published years ago by a
     single maintainer is a materially different risk than an actively maintained one, even
     if both are "popular")
3. **Sign-off from a tech lead or above**, recorded as an approving review on the PR, not a
   verbal or Slack approval.
4. Once approved, the package MUST be pinned exact and MUST be added to that project's
   README as a documented, justified exception — the baseline table in Section 2 is the
   default, not an amendment; each project tracks its own deltas from it.

Exceptions are per-project, not blanket. Approval for one project does not pre-approve the
same package elsewhere — re-justify against that project's actual constraints.

---

## 7. What this policy is not

This policy does not forbid npm, forbid third-party code categorically, or mandate reinventing
genuinely hard problems (cryptography, date/timezone edge cases in some domains, etc. are
legitimate exception candidates). It forbids *default, unreviewed* dependency sprawl — the
normal state of a JS frontend where a `create-react-app` or an unmanaged `npm install
whatever-looked-useful` produces a tree of 800+ packages nobody on the team has looked at.
The goal is a dependency tree small enough that "what does this app actually depend on" has
an answer a person can hold in their head.