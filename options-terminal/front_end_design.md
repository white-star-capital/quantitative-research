# DAF III dashboard — design system & front-end gotchas (for the design team)

## Context
The user asked for this project's colour scheme and front-end gotchas, so the design team can write instructions that follow the same design structure. Everything below is taken from the code: `src/styles/tokens.css`, `app.css` and `yield.css`, the components, and `docker/nginx.conf`. The only implementation step after approval is optional: commit this as `DESIGN.md` at the repo root on `v1`, so it lives next to the code.

---

## 1. Look and feel in one line
A **terminal-style trading desk**: black background, amber labels, one monospace font, 1px grey rules, a fixed 22px row height, and colour reserved for meaning (up, down, warning, identity). It has no decoration: **no rounded corners, no shadows, no gradients, no blur**. The codebase enforces this globally (`* { border-radius: 0 }`).

## 2. Colour tokens (exact values; never derive a second palette)
| Token | Hex | Role |
|---|---|---|
| `--bg` | `#000000` | Page background |
| `--bg-raise` | `#0c0c0c` | Panels, KPI tiles, sub-bars |
| `--rule` | `#3a3a3a` | Every 1px border and divider |
| `--rule-strong` | `#6b5a32` | Emphasis rule; the command-acknowledgement flash |
| `--bar-track` | `#1a1a1a` | Empty track behind bars; chart gridlines |
| `--paper` | `#f2efe6` | Primary text and values (warm off-white, not pure white) |
| `--dim` | `#8d887c` | Secondary text, captions, units, axis ticks |
| `--amber` | `#fb8b1e` | Brand accent: titles, row codes, active border, primary line in charts |
| `--amber-dim` | `#a8722a` | Small uppercase field labels (`.label`) |
| `--amber-bright` | `#ffb01e` | Focus outline |
| `--up` | `#00d964` | Positive / in range / earning / profit |
| `--down` | `#ff3b30` | Negative / failed / stale / loss / out of range |
| `--yellow` | `#ffe14a` | Middle state (e.g. 50–100% of target) |
| `--cyan` | `#37d0e6` | Second data series (bars, "today" model line, LP principal, spot marker) |
| `--header-bar` | `#7a1020` | Section title bars (deep red, white text) |
| `--green-key` / `--red-key` / `--yellow-key` / `--blue-key` | `#009e49` / `#e10600` / `#f5c400` / `#2f6fed` | Function-key buttons only (F-keys: green = navigation, red = cancel, yellow = data tabs) |

**Rules**
- **Meaning colours:**
  - Green and red mean good and bad only. Never use them for a series identity.
  - Amber and cyan are the two series colours.
  - Yellow is the "in-between" status.
- **Text colour:** text uses `--paper` or `--dim`. A coloured value appears only when the colour carries meaning (P&L sign, status word).
- **Status is never colour-alone:** always pair it with a word (OK / FAILED / STALE / ITM / NO SPOT) or a sign (+ / −).
- **Landing page:** the Yield Monitor's top bar is solid amber with black text. That is the only place amber is used as a background.
- **Contrast:** the CVD check on amber/cyan over black passes (ΔE 22). They are brighter than generic dataviz guidance suggests; that is deliberate, for the terminal look.

## 3. Typography
- **One font:** IBM Plex Mono, weights 400 and 500 only, self-hosted (`/fonts/*.woff2`). Fallback: Consolas, Roboto Mono, ui-monospace.
- **Base:** 12px/16px. The landing page uses 13px. KPI values are 22px/28px, the only large text.
- **Labels:**
  - 10–11px, UPPERCASE, letter-spacing 0.07–0.14em, amber or amber-dim.
  - Titles use wider tracking (0.12–0.14em).
- **Numbers:**
  - Always `font-variant-numeric: tabular-nums`, right-aligned in tables, `white-space: nowrap`.
  - Minus is the real minus sign "−", not a hyphen.
  - Money: `$1,234.56`; compact on axes: `$12.5K`, `$1.2M`.
  - Signed changes show "+".
- **Dates:** `12 OCT, 15:17 UTC`. All data times are UTC; the landing clock is New York.

## 4. Layout grammar
- **Rows:**
  - `--row: 22px` is the minimum height of every row, bar, button and table line.
  - Padding is tight: 2–4px vertical, 8–10px horizontal.
  - Gaps between blocks are 2px, so the rules read as a grid.
- **Desk shell** (top to bottom):
  1. identity bar (title + AS OF time);
  2. function keys;
  3. command line;
  4. message line;
  5. scrolling work area;
  6. suggested-functions bar.
  - Only the work area scrolls; the page never does (`html.desk-page { overflow: hidden }`). The shell uses `100dvh` so mobile browser bars don't cut it.
- **Panel/card anatomy:**
  1. title line (code `1)` + name in amber caps, status word on the right with a coloured square);
  2. meta line (dim);
  3. picture (bar or chart);
  4. 3-column key/value strip (`.three`);
  5. notes (`NOTE` prefix);
  6. `99) MORE <GO>` toggle;
  7. detail `<dl>`.
  - The active card gets an amber border.
- **Section titles:** a deep-red `--header-bar` strip with white UPPERCASE text. The right side holds the source/status (e.g. "LAST 7 DAYS · FROM POSITION SNAPSHOTS").
- **Status line under a title (`.chains`):** source by source, e.g. `MAINNET OK 2 OPEN · BLOCK … · time`, or `FAILED <reason> · SHOWING N FROM LAST GOOD SNAPSHOT`.
- **Landing page:**
  - amber command bar;
  - sub-bar;
  - 6 KPI tiles;
  - wallet table (73%) beside allocation (27%);
  - footer with source and load time.

## 5. Charts (SVG, drawn by hand, no chart library)
- **Width:** measure the container (`useWidth`) and draw at real pixel width. Never scale with `viewBox`, or text sizes drift.
- **Marks:**
  - 2px lines;
  - bars at most 40px wide, with a 4px rounded top on a square baseline (the one exception to "no radius", and only on data marks);
  - dots of 8px or more with a 2px black ring.
- **Grid and axes:** gridlines are `--bar-track`, the zero line is `--dim`, axis text is 11px `--dim`. A single y-axis only, never a dual axis.
- **Series:** amber = primary or "at expiry", cyan = secondary or "today (model)". When they share a plot, the second series is also dashed.
- **Profit/loss areas:** green or red at about 14% opacity, clipped at zero.
- **Estimates:** hatched fill, plus an "≈" prefix on the value.
- **Every chart has:**
  - a legend when there are two or more series;
  - direct labels on key points (SPOT, BE, K1, LAST, VWAP);
  - a hover/keyboard crosshair with a tooltip;
  - a screen-reader `<table class="sr-only">` with the same data;
  - a dim footer sentence explaining how the numbers are made.
- **Label collisions:** labels near the right edge flip to the left of their line, and labels are clamped away from card edges.

## 6. States & data honesty (non-negotiable)
- **NO LIVE DATA YET:** the file doesn't exist yet. Never show $0 or placeholder numbers.
- **"NO SNAPSHOTS" / "—" / "not priced":** unknown is never drawn as zero.
- **STALE:** data older than its window (60 min) is labelled, in red.
- **Last good data on failure:** it stays on screen labelled `SHOWING LAST GOOD … FROM <time>`, with the error text.
- **Approximate values:** prefixed "≈", with a tooltip line explaining why.
- **A broken config file:** a red panel listing every bad field, never partial numbers.
- **No sample or fake data in production views.**

## 7. Interaction
- **Keyboard first:**
  - F-keys switch tabs;
  - typing anywhere goes to the command line;
  - Enter runs the command; a number selects a row; `99` expands it; ESC cancels.
- **Mouse:** clicking does the same things.
- **Focus:** a 1px `--amber-bright` outline, inset.
- **Feedback:** the message line flashes `--rule-strong` for 150ms on each command. The flash is off under `prefers-reduced-motion`.

## 8. Responsive
- **Breakpoints:**

| Width | Change |
|---|---|
| ≤ 1100px | KPI grid goes to 3 columns; landing body goes to 1 column |
| ≤ 900px | Card grids go to 1 column |
| ≤ 640px | Phone layout (below) |
| ≤ 560px | KPIs go to 2 columns |

- **Phone layout (≤ 640px):**
  - tables become stacked cards using `data-label` attributes;
  - F-keys wrap to rows of three;
  - charts shrink to fewer ticks, day numbers, and compact `$1.2K` labels.
- **Never horizontal page scroll.** Wide tables scroll inside their own panel.

## 9. Front-end gotchas (things that bit us)
1. **CSP is strict:**
   - `default-src 'self'`; scripts, styles and fonts come only from the same origin.
   - No CDNs and no Google Fonts: self-host everything.
   - No inline `<style>` or `<script>` in HTML. Set dynamic styles through React's `style` prop, which the CSP allows.
2. **Password gate:** every path except `/login.html`, `/login.js`, `/login.css`, `/fonts/` and `/healthz` needs the `gm_auth` cookie. Assets a new page needs before login must go in that list.
3. **Caching:**
   - HTML is `no-cache, no-store`, data and config JSON are `no-store`, and `/assets/*` is `immutable`.
   - Never put data in `/assets`, and never cache JSON.
   - Fetch with `cache: "no-store"`.
4. **Live data:** feeds are fetched at runtime, every 60s and on tab focus, never bundled. A non-JSON response (nginx serving `index.html`) counts as "missing".
5. **Routing:** hash routing only (`#/desk`), with no router library. Deep links survive the login redirect via the hash.
6. **No browser storage** (localStorage etc.) for anything that must be right. State comes from the server files.
7. **New deploys:** open tabs reload themselves when the bundle hash changes. Don't break the `index-*.js` entry naming.
8. **Desk scrolling:**
   - Only the work area scrolls; add the `desk-page` class when a page should behave like the desk.
   - Screen-reader tables must be `position: absolute` inside a `position: relative` parent, or they create a second scrollbar.
9. **Text in charts** is sized in CSS px through classes (`.fs-tick`, `.ep-label`), not SVG units.
10. **Numbers:**
    - Format at display time.
    - Keep raw values exact (strings for big integers).
    - Never round twice.
11. **Dependencies:** exact-pinned, with a lockfile; no new runtime deps without review. Charts and routing are hand-rolled on purpose.

## Optional implementation step
Save the sections above as `DESIGN.md` at the repo root and link it from `README.md` and `DEV_README.md`. Commit and push to `v1`. This is docs only, so tests and the build are unaffected; run the source gates anyway.
