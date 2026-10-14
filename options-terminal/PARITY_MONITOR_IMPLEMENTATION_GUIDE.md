# `parity_monitor` — Put-Call Parity Tab: Implementation Guide

**Status:** Draft v0.1 · **Owner:** Alejandro · **Date:** 2026-10-14
**Audience:** developers taking the second model live across `contracts/`, `models/`, `api/` and `frontend/`.
**Read first:** `README.md`, `CONTRACTS.md`, `MODELS.md`, `API.md`, `FRONTEND.md`, and `BS_PRICER_IMPLEMENTATION_GUIDE.md` (this slot reuses its Deribit adapter, conventions and formatting rules).

---

## Amendments (2026-10-14, implementation)

These came out of building v0.1 against the existing code. Where they conflict with the text below, they win.

- **A-1 Order-book confirmation (D-13) and M-5.** A model may not do I/O in `run`, but which order books to fetch depends on screening the data. Models gain an optional `followup_inputs(scope, inputs) -> list[InputSpec]` (`core/base.py`). The runner fetches the declared inputs, calls `followup_inputs` (pure screening via `core/quant/parity.py`), fetches those, applies the same staleness checks, then runs. Every input goes to the snapshot store, so replay and recordings cover the order books too.
- **A-2 Fee table in the payload.** `scalar_set` values cannot be strings, so `fee_schedule_version` is not in `context`. A `fees` table section carries the fee table the run used, with its `schedule_version`. The frontend's maker/taker toggle recomputes from it (§7.4).
- **A-3 Snapshot cache (D-11).** One Deribit client already serves every slot and asset in a `run-due` pass. The cache is keyed on (method, params, tick) where the tick is that pass. A cached response is reused only within the tick and while younger than `TICK_MAX_REUSE_S`, so a reused response never carries a stale server time into a 1-minute model.
- **A-4 Fees.** Fee formulas used by `core/quant/parity.py` (per 1 unit of underlying, in the option's native unit):
  - option: `min(option_rate × U, cap × option_price)`, with U = 1 coin for inverse books and U = index price (USD) for linear books;
  - future/perp: `rate × notional`. For inverse books the notional is K USD, so the fee in coin is `rate × K / F`. For linear books it is `rate × F` USD;
  - delivery, for structures held to expiry: exactly one option leg finishes in the money, and it pays `option_delivery × U`. The cap isn't applied, because the option's value at expiry is unknown, so the uncapped fee is the conservative bound. A dated-future hedge also pays `future_delivery` on its notional. A perp pays no delivery fee;
  - fees are subtracted from the gross edge as paid (`net = gross − fees`). That is exact at `usd_rate = 0`, Deribit's convention. `core/quant/parity.py:net_metrics` derives `net_edge_usd` (inverse: coin × index), `net_edge_bps_notional` (vs the K USD notional) and `net_edge_ann` (simple). The frontend's maker/taker toggle uses the same derivation, held together by `contracts/fixtures/parity/vectors.json`.

  `parity_deribit` stays **out of `registry.yaml` until the fee table is filled**: the runner validates the whole registry each pass, so null fees in a registered manifest would also stop `bs_pricer`.
- **A-5 Sanity checks** `parity_band_ordering` and `implied_vs_venue_forward` are in `core/validation.py`. The bs_pricer checks assume that slot's sections.

---

## 0. Summary

| | |
|---|---|
| **Slot** | `parity_monitor` |
| **First implementation** | `parity_deribit` v1.0.0 |
| **Assets** | `BTC`, `ETH`, `SOL`, `HYPE` |
| **Cadence** | Every 1 minute per asset; stale after 3 minutes |
| **What it answers** | (1) What forward and carry are the options implying, and how does that compare to futures and perp funding? (2) Where is parity violated by enough to trade **after fees, at executable prices**? (3) What USD rate can the desk lock by converting coin inventory? |
| **New API code** | None. Read-only through the generic endpoints. |
| **Fidelity** | **F2.** The math is fully specified and tested (§3). The data work is executable prices on four legs and synchronous snapshots. |

**What this tab is not:** an execution engine. Real parity breaks on Deribit last seconds; a 1-minute monitor finds *persistent* dislocations, implied-rate regimes and inventory carry, and points a trader at where to look. Every edge is labelled "indicative at snapshot" and must be re-checked live before trading.

### What ships, by folder

| Folder | Adds |
|---|---|
| `contracts/` | `slots/parity_monitor.yaml`, fixtures per asset, `contracts/fixtures/parity/` vectors |
| `models/` | `core/quant/parity.py`, Deribit adapter additions (futures, perps, order books), shared snapshot cache, `library/parity_deribit/` |
| `api/` | Nothing new |
| `frontend/` | Parity tab in the asset view, parity columns on the cross-asset monitor |

---

## 1. From the reference write-up to Deribit

The parity guide you provided is correct for equities, but five points change for our venue:

| Write-up says | On Deribit | Consequence |
|---|---|---|
| C − P = S − K·e^(−rT) (spot form) | Options are priced off each expiry's **forward**; the hedge is a future or perp, not borrowed stock | Use the forward form (the write-up's §2 futures variant) as the primary identity |
| One parity formula | BTC/ETH options are **inverse** (premium and settlement in coin); SOL/HYPE are **linear** (USDC) | Two formula sets (§3). The inverse form isn't in the write-up and is where bugs will come from. |
| Conversion/reversal are risk-free | Only true when a **same-expiry dated future** exists. BTC/ETH have dated futures for most expiries; **SOL and HYPE have only perpetuals** (checked 2026-10-14). | Every opportunity carries a hedge type (§2.3) that sets its residual risk |
| American options turn parity into an inequality | Deribit options are **European** | Parity is an equality; the American bound doesn't apply |
| Dividends / FX rates adjust the formula | No dividends. Staking/lending yield on the coin acts like q **only if you hedge with spot** | Matters for the inventory view (§3.5), not the futures-hedged trades |

The "capital wedge" point carries over directly: margin on the short option leg and the hedge is what makes most apparent edges uneconomic. It's handled as a user-set capital assumption in the display (§7.4) because v1 has no margin model.

---

## 2. Market data

### 2.1 Additions to the Deribit adapter

| Purpose | Endpoint | Calls per tick per asset |
|---|---|---|
| Option bid/ask (coin or USDC) + `underlying_price` + `underlying_index` | `public/get_book_summary_by_currency?currency=…&kind=option` (already used by `bs_pricer`) | Shared |
| Dated futures and perp bid/ask, mark | `public/get_book_summary_by_currency?currency=…&kind=future` | 1 (shared between SOL and HYPE via USDC) |
| Perp funding | `public/ticker?instrument_name=<perp>` (`current_funding`, `funding_8h`; confirm field semantics in M1) | 1 |
| Spot index | `public/get_index_price` (already used) | Shared |
| **Executable confirmation** for candidate violations | `public/get_order_book?instrument_name=…&depth=5` per leg | ≤ 3 × `max_confirmations` (default 8) |

Rules (extend the `D-` rules in the pricer guide):

- **D-11 Shared snapshot cache.** Within one scheduler tick, identical adapter calls return the same response. `bs_pricer` and `parity_monitor` see the same option book; SOL and HYPE share one USDC call. Key: (endpoint, params, tick_id).
- **D-12 Snapshot skew.** Record `snapshot_at` per response. If the option book and the futures book are more than `max_snapshot_skew_s` apart (default 3 s), flag `snapshot_skew` on every opportunity computed from that pair. Unconfirmed opportunities are never shown as executable when skew is flagged.
- **D-13 Order-book confirmation.** For the top `max_confirmations` screened candidates per asset (by annualized edge), fetch the order books for all legs back to back. Recompute the edge from those books. Record the time spread across the legs' timestamps; if > 1 s, flag `leg_time_spread`.
- **D-14 Book summary has no sizes.** Top-of-book sizes come only from order-book confirmation. Unconfirmed rows show size as "unconfirmed".
- **D-15** Instrument matching is by metadata (`expiration_timestamp`, `strike`, `option_type`, `base_currency`), never by constructed names (pricer D-2).

### 2.2 Same-expiry future matching

For each option expiry:

1. If a dated future with the same `expiration_timestamp` exists → hedge instrument = that future.
2. Else → hedge instrument = the asset's perpetual.
3. Record Deribit's `underlying_index` for the expiry as `venue_forward_source`, i.e. the future name or a synthetic (`SYN.…`). Display only.

Verify in M1 that Deribit options and futures with the same expiry settle to the same delivery price. The `locked_future` classification depends on it; if the check fails, downgrade those rows to `basis_risk` and document it.

### 2.3 Hedge types

| Hedge type | When | Residual risk shown to users |
|---|---|---|
| `locked_future` | Same-expiry dated future exists (most BTC/ETH expiries) | Venue/counterparty; leg execution; for inverse, the coin value of the locked edge (§3.2) |
| `perp_funding` | No dated future: **all SOL and HYPE expiries**, plus BTC/ETH expiries without a future | Cumulative funding until expiry, plus perp-to-delivery convergence. The trade isn't locked; the display shows the breakeven funding rate. |
| `spot_inventory` | Inventory view only (§3.5) | Forgone staking/lending yield on the coin; custody |

---

## 3. Methodology

Notation per (asset, expiry, strike K): call bid/ask c_b, c_a; put bid/ask p_b, p_a; hedge bid/ask F_b, F_a; T in years (calendar, ACT/365, 24/7); DF = e^(−rT) with r = `usd_rate` param, default 0 (Deribit's convention, as in the pricer).

### 3.1 Linear options (SOL, HYPE; prices in USD per unit)

| Quantity | Formula |
|---|---|
| Implied forward, mid | F_impl = K + (c_mid − p_mid)/DF |
| **Parity band** | F_impl_bid = K + (c_b − p_a)/DF, F_impl_ask = K + (c_a − p_b)/DF |
| **Conversion** (sell C, buy P, buy hedge at F_a), USD at expiry per unit | E_conv = (c_b − p_a)/DF − (F_a − K) |
| **Reversal** (buy C, sell P, sell hedge at F_b), USD at expiry per unit | E_rev = (F_b − K) − (c_a − p_b)/DF |

A conversion is gross-profitable iff F_impl_bid > F_a; a reversal iff F_impl_ask < F_b. Hedge size: 1 unit of the linear future/perp per 1 unit of option.

### 3.2 Inverse options (BTC, ETH; prices in coin per 1-coin option)

Coin payoffs: call (S_T − K)⁺/S_T, put (K − S_T)⁺/S_T. Hedge with the **inverse** future at USD notional **N = K per 1 coin of options**. A long inverse future bought at F₀ pays N·(1/F₀ − 1/S_T) coin.

| Quantity | Formula |
|---|---|
| Implied forward, mid | F_impl = K / (1 − (c_mid − p_mid)) |
| **Parity band** | F_impl_bid = K / (1 − (c_b − p_a)), F_impl_ask = K / (1 − (c_a − p_b)) |
| **Conversion**, coin locked at expiry per 1 coin | E_conv = (c_b − p_a) − 1 + K/F_a |
| **Reversal**, coin locked at expiry per 1 coin | E_rev = 1 − K/F_b − (c_a − p_b) |
| USD equivalent | E_usd ≈ E_coin × S (display), flagged `coin_denominated_edge` |

Same profitability rule as §3.1 (F_impl_bid > F_a, or F_impl_ask < F_b). The edge is locked **in coin**, so its USD value moves with spot. It's small and second-order, but it is shown.

Hedge sizing: inverse future contracts = K / contract_size_usd per coin of options (BTC futures are 10 USD per contract; read `contract_size` from metadata, never hard-code). Rounding to whole contracts leaves a residual. Report `hedge_rounding_residual_usd`; flag it if > 0.5% of notional.

**Verified:** §3.1 and §3.2 were checked against simulated terminal payoffs on 2,000 random (K, F₀, S_T, quote) draws for both conversion and reversal. The lock is exact.

### 3.3 Fees and net edge

Net edge = gross edge − fees on all three legs (options at the option fee rate subject to the venue's per-option cap as a fraction of the option price; hedge at the futures/perp rate; plus delivery/settlement fees if held to expiry).

- **F-1** Fee rates are manifest params under `fees:` (maker and taker per leg type, option cap, delivery fees), **populated from Deribit's current fee schedule and reviewed quarterly**. They are not hard-coded and not guessed. The run records the fee table it used in `inputs`.
- **F-2** Default assumption: **taker on all legs** (what a monitor can actually execute). A display toggle shows the maker/taker mix (§7.4).
- **F-3** Net edge ≤ 0 is the normal state and is shown as such; the opportunity table filters on net edge > threshold.

### 3.4 Implied carry per expiry

Per expiry, a robust **options-implied forward**:

1. Strikes with \|ln(K/F_venue)\| ≤ `atm_band` (default 0.10), both call and put two-sided.
2. Per strike: F_impl (mid) and band width.
3. Expiry F_impl = weighted median, weight = 1 / band width. Require ≥ 3 strikes; else `insufficient_strikes` for that expiry.

Then, all annualized as simple rates over T:

| Metric | Definition |
|---|---|
| `implied_basis_ann` | (F_impl / S − 1) / T |
| `future_basis_ann` | (F_hedge_mid / S − 1) / T, `locked_future` expiries only |
| `options_vs_future_bps` | (F_impl / F_hedge_mid − 1) × 10⁴. The core dislocation signal. |
| `funding_ann` | Perp funding annualized per Deribit's field definition |
| `breakeven_funding_ann` | `perp_funding` expiries: the constant annualized funding at which the best net conversion/reversal edge is zero. Compare to current `funding_ann`. |

### 3.5 Inventory conversion: lockable USD rate

For a desk holding coin (e.g. the ETH inventory), holding 1 unit + selling the call + buying the put at strike K delivers exactly K USD at expiry. This is the write-up's spot-form identity used as a funding trade:

| Settlement | Lockable USD rate (continuous, annualized) |
|---|---|
| Linear | r_lock = ln( K / (S − c_b + p_a) ) / T |
| Inverse | r_lock = ln( K / (S · (1 − (c_b − p_a))) ) / T |

Net of option fees. Per expiry, report the best strike within `atm_band` and its rate. S is the index: this values existing inventory at its opportunity cost and assumes no spot trade.

The frontend computes **pickup** from two user inputs, the USDC alternative rate and the coin yield forgone (staking/lending):
- pickup vs USDC = r_lock − r_usdc_alt
- pickup net of forgone yield = r_lock − r_usdc_alt − q_coin_forgone

The conversion only beats simply holding the coin if the second number is positive.

Verified: inventory + short synthetic pays exactly K USD at expiry for the inverse case on the same simulation.

### 3.6 Greek parity identities (property tests for `bs_pricer`)

The write-up's §3 identities, rewritten for our Black-76 forward conventions, become property tests in `core/quant/tests/test_black76.py` (amends pricer guide §5.5; the tests live in `models/tests/test_black76.py`):

| Identity | Our convention |
|---|---|
| Forward delta | Δ_C − Δ_P = DF |
| Spot delta | Δ_C − Δ_P = DF · F/S |
| Gamma | Γ_C = Γ_P |
| Vega | ν_C = ν_P |
| Theta (forward held fixed) | Θ_C − Θ_P = r · DF · (F − K) / 365 |
| Price | C − P = DF · (F − K) |

With r = 0 these reduce to Δ_C − Δ_P = 1 (forward) and Θ_C = Θ_P. Rho isn't computed in v1.

### 3.7 Reference implementation

`models/terminal_models/core/quant/parity.py`. Tested as described above:

```python
import math

# Linear (USDC-settled): prices USD per unit
def lin_band(K, c_bid, c_ask, p_bid, p_ask, DF=1.0):
    return K + (c_bid - p_ask) / DF, K + (c_ask - p_bid) / DF

def lin_conversion(K, c_bid, p_ask, F_ask, DF=1.0):
    return (c_bid - p_ask) / DF - (F_ask - K)

def lin_reversal(K, c_ask, p_bid, F_bid, DF=1.0):
    return (F_bid - K) - (c_ask - p_bid) / DF

# Inverse (coin-settled): prices in coin per 1-coin option; hedge notional = K USD
def inv_band(K, c_bid, c_ask, p_bid, p_ask):
    return K / (1 - (c_bid - p_ask)), K / (1 - (c_ask - p_bid))

def inv_conversion(K, c_bid, p_ask, F_ask):
    return (c_bid - p_ask) - 1 + K / F_ask

def inv_reversal(K, c_ask, p_bid, F_bid):
    return 1 - K / F_bid - (c_ask - p_bid)

# Inventory lock: hold 1 unit, sell call, buy put
def lin_inventory_rate(K, S, c_bid, p_ask, T):
    return math.log(K / (S - c_bid + p_ask)) / T

def inv_inventory_rate(K, S, c_bid, p_ask, T):
    return math.log(K / (S * (1 - (c_bid - p_ask)))) / T
```

Guards the production version MUST add: return `null` + reason when 1 − (c − p) ≤ 0 (inverse), when any input ≤ 0, or when T < `min_tte_hours`.

---

## 4. Contracts

### 4.1 `contracts/slots/parity_monitor.yaml`

```yaml
slot_id: parity_monitor
title: Put-call parity monitor
description: >
  Options-implied forwards and carry vs futures and funding, executable
  conversion/reversal edges after fees, and lockable USD rate on inventory.
payload_kind: composite
payload_version: "1"
scope:
  dimensions: [underlying]
  allowed:
    underlying: [BTC, ETH, SOL, HYPE]
max_age_seconds: 180
conventions: >
  Inverse (BTC/ETH) and linear (SOL/HYPE) parity per PARITY_MONITOR_IMPLEMENTATION_GUIDE.md §3.
  Edges per 1 unit of underlying; inverse edges in coin with USD equivalent.
  Rates simple annualized unless noted; r_lock continuous. Taker fees on all legs.
display:
  default_view: parity_monitor
  order: 2
```

### 4.2 Payload — composite sections

| Section | Kind | Content |
|---|---|---|
| `context` | `scalar_set` | spot, funding_ann, n_expiries, n_locked_expiries, n_candidates, n_confirmed, best_net_edge_ann, snapshot_skew_s, fee_schedule_version (see A-2: in `fees`) |
| `carry_curve` | `curve` | x = tenor_days; series `implied_basis_ann`, `implied_basis_ann_bid`, `implied_basis_ann_ask`, `future_basis_ann` (null where no future), `funding_ann` (flat), `r_lock_best` |
| `expiries` | `table` | expiry_ts, dte, hedge_type, hedge_instrument, venue_forward_source, F_impl, F_impl_band_bps, F_hedge_bid/ask, options_vs_future_bps, implied_basis_ann, future_basis_ann, breakeven_funding_ann, n_strikes_used, r_lock_best, r_lock_best_strike |
| `bands` | `table` | One row per (expiry, strike) with two-sided call and put: k, F_impl_bid, F_impl_ask, band_bps vs hedge mid, hedge_bid, hedge_ask, gross_conv, gross_rev |
| `opportunities` | `table` | Columns below; net edge > `min_net_edge_bps` only, max 200 rows |
| `fees` | `table` | A-2: the fee table the run used, one row per leg type, with `schedule_version` |

`opportunities` columns: `expiry_ts`, `strike`, `structure` (`conversion`/`reversal`), `hedge_type`, `hedge_instrument`, leg prices (`call_px`, `put_px`, `hedge_px`, each with `_unit` coin/usd), `gross_edge`, `fees`, `net_edge` (native unit), `net_edge_usd`, `net_edge_bps_notional`, `net_edge_ann`, `confirmed` (boolean), `exec_size_units` (null if unconfirmed), `hedge_contracts`, `hedge_rounding_residual_usd`, `snapshot_age_s`, `flags` (string, comma-joined codes).

### 4.3 Fixtures and vectors

- `fixtures/valid/composite/parity_{btc,eth,sol,hype}.json`. The BTC fixture includes one `locked_future` confirmed opportunity; SOL/HYPE include only `perp_funding`; one fixture carries `snapshot_skew`.
- `fixtures/parity/vectors.json`: band, conversion, reversal and inventory-rate cases for both settlement types, including the guard cases. Python-generated; consumed by the frontend tests (the display recomputes pickup from these values) and by any future Rust evaluation.

---

## 5. Models

```
models/terminal_models/
├── core/quant/parity.py                 # §3.7 + guards + robust implied forward (§3.4)
├── adapters/deribit/
│   ├── futures.py                       # dated futures + perp book, funding
│   ├── orderbook.py                     # D-13 confirmations
│   └── snapshot_cache.py                # D-11
└── library/parity_deribit/
    ├── manifest.yaml
    ├── model.py
    ├── README.md
    └── tests/
```

```yaml
implementation_id: parity_deribit
version: 1.0.0
fills_slot: parity_monitor
schedule: "* * * * *"
timeout_seconds: 40
max_input_age_seconds: 30
params:
  usd_rate: 0.0
  atm_band: 0.10
  min_tte_hours: 1
  min_strikes_for_forward: 3
  max_snapshot_skew_s: 3
  max_confirmations: 8
  min_net_edge_bps: 0          # rows kept in `opportunities`
  fees:                        # FILL FROM DERIBIT'S CURRENT FEE SCHEDULE — reviewed quarterly
    schedule_version: "TBD"
    option_taker: null         # fraction of underlying per option
    option_maker: null
    option_cap_pct_of_price: null
    future_taker: null
    future_maker: null
    perp_taker: null
    perp_maker: null
    option_delivery: null
    future_delivery: null
sanity_checks:
  - parity_band_ordering       # F_impl_bid ≤ F_impl_ask for every strike; violation = bad data
  - implied_vs_venue_forward   # |F_impl / venue underlying_price − 1| < 1%; else warning
```

**M1 blocker:** `validate-registry` MUST fail while any `fees.*` value is null, so the model can't run with made-up fees.

Model flow per asset: build option pairs by (expiry, strike) → match hedge (§2.2) → bands and gross edges for every pair → expiry implied forwards and carry → net edges → rank → order-book confirm top N → recompute → inventory rates → sections.

Tests:
- Unit: every §3 formula against vectors; guard cases; the robust forward on a synthetic chain with one outlier strike (median must ignore it).
- Payoff simulation test for the lock (as in §3.2), kept in CI.
- Golden per asset from recorded snapshots (including order books).
- Skew case: futures book 5 s older → no row marked executable.
- `perp_funding` case: breakeven funding computed and finite.
- Determinism ×10; replay.

Registry: shadow first with `production: null`, same as the pricer. Promote after §9 step 6.

---

## 6. API

No new endpoints. Served through `GET /v1/slots/parity_monitor/latest?underlying=…` and `/runs`.

- Payload size: `bands` can reach ~1,500 rows for BTC. Expected payload ~400–700 KB; inside CONTRACTS C-6.
- With `max_age_seconds = 180`, API A-12 gives `Cache-Control: max-age ≤ 30`, and frontend polling (F-3) lands at 30 s. That's appropriate for this slot.

---

## 7. Frontend

Design and security per `front_end_design.md` and `building_secure_front_ends.md`. Formatting per pricer guide §7.8, plus:

| Value | Format |
|---|---|
| Basis / rates | % annualized, 2 dp (`6.85% ann.`) |
| Dislocations | bps, signed, 1 dp (`+12.4 bp`) |
| Coin edges | coin with 6 dp and USD equivalent (`0.000412 BTC ≈ $41`) |

### 7.1 Cross-asset monitor additions

Add to the pricer's monitor row (or a sibling table on the Parity landing view): **Implied basis 30d · Future basis 30d · Options vs futures (bp) · Funding (ann.) · Best net edge (ann.) · Confirmed opps · Inventory r_lock 30d**. SOL/HYPE show "perp only" in the future-basis column instead of a blank.

### 7.2 Asset view → Parity tab

Four panels, in this order:

**1. Carry curve (PMs).** x = tenor; lines for options-implied basis (with shaded bid–ask band), future basis (markers, BTC/ETH), perp funding (dashed horizontal) and best lockable inventory rate. A user-entered USDC lending rate appears as a reference line (client-side input, remembered per viewer). This answers "where is carry richest, and do options agree with futures?" at a glance.

**2. Parity band chart (traders).** Expiry selector. x = strike; for each strike, a vertical bar from F_impl_bid to F_impl_ask expressed in bp vs hedge mid; horizontal lines at hedge bid and ask. A bar entirely above the hedge ask is a conversion; entirely below the hedge bid is a reversal. Bars are coloured by gross vs net-of-fees status, and confirmed rows get a marker. This is the fastest view of where parity is tight and where it's loose.

**3. Opportunities table (traders).** Columns: expiry · strike · structure · hedge type (badge) · legs with prices · net edge (native and USD) · bp of notional · annualized · size · confirmed · age · flags.
- Default filter: `confirmed = true` and `hedge_type = locked_future`. Toggles reveal unconfirmed and `perp_funding` rows; those rows are visually distinct and carry their residual risk in a tooltip (§2.3).
- `perp_funding` rows show **breakeven funding vs current funding** instead of a plain annualized edge.
- Display toggles: taker/maker mix (recomputes fees client-side from the run's recorded fee table), and capital assumption (% of notional as margin, user-entered) to show annualized return on capital alongside return on notional. This is the "capital wedge."
- Every row: "Indicative at {snapshot time}. Re-check live before trading."

**4. Inventory carry (PMs/treasury).** Inputs held client-side only: holding size, USDC alternative rate, coin yield forgone (staking/lending). Table per expiry: best strike, r_lock, pickup vs USDC, pickup net of forgone yield, USD locked at expiry for the holding, and option fees. Default sort is by pickup net of forgone yield. Covered-call-style framing ("you give up upside above K and are protected below K") appears as a one-line explanation, not a verdict.

### 7.3 States

| State | Behavior |
|---|---|
| `snapshot_skew` flagged | Banner on the tab; opportunities table shows no executable rows |
| No confirmed opportunities | Normal and expected. Show "No executable parity breaks after fees at {time}"; the bands chart still renders |
| Expiry `insufficient_strikes` | Carry point omitted; expiry listed with reason in the expiries table |
| Fee table missing/outdated | Impossible in production (M1 blocker); in staging show a red banner |

---

## 8. Tests

| Layer | Test |
|---|---|
| Contracts | Fixtures valid/invalid; vectors regenerate identically |
| Models | §5 list; payoff-simulation lock test; registry fails on null fees |
| API | Generic endpoint integration on parity fixtures (no new code) |
| Frontend | Each panel from each asset fixture; hedge-type badges and filters; fee-mix and capital toggles recompute correctly against vectors; client-side inputs never sent in any request |
| E2E | Monitor → BTC → Parity → confirmed opportunity visible → toggle maker → edge changes; SOL shows perp-only treatment |

---

## 9. Go-live runbook

| # | Step | Done when |
|---|---|---|
| 1 | Contracts PR (slot, fixtures, vectors) | CI green |
| 2 | Fee table populated from Deribit's schedule; reviewed by trading | `fees.schedule_version` set; `validate-registry` passes |
| 3 | Models PR (parity core, adapter additions, snapshot cache, implementation) as shadow | §5 tests green; local runs for all four assets |
| 4 | Verify same-expiry option/future settlement alignment (§2.2) | Documented in adapter README |
| 5 | Frontend PR (monitor columns, Parity tab) | Fixture renders; design/security sign-off |
| 6 | Shadow validation, 3 days: implied forward within 1% of venue forward on > 95% of expiries; confirmed opportunities spot-checked live by a trader at least 10 times; inventory r_lock sanity-checked against a hand calculation for ETH | Trader + quant sign-off |
| 7 | Promote in registry | Merged and deployed |
| 8 | Alerts: stale > 3 min, failure rate > 5%/h, `snapshot_skew` on > 20% of ticks in an hour | Fired in staged test |

---

## 10. Backlog

1. **OTC collar / risk-reversal check:** add an evaluation endpoint (pricer §6 pattern) that takes a counterparty's call and put quotes at a strike and returns their implied forward vs market. This catches quotes that are skewed through the forward rather than through the vol. Directly useful for the Auros program.
2. Box spreads (two-strike parity) as a pure rate instrument; no hedge leg needed.
3. Margin model so return on capital is computed rather than user-assumed.
4. Bybit as a second venue for SOL/HYPE, which would enable cross-venue parity.
5. History panel: options-vs-futures dislocation and r_lock over time, from `/runs`.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Phantom arbs from non-synchronous quotes | Snapshot skew rule, order-book confirmation, "indicative" labelling |
| Inverse/linear formula mix-up | Separate functions, payoff-simulation test, per-settlement fixtures |
| Wrong or stale fee assumptions inflate edges | Fees from the schedule only, versioned in each run, null blocks the registry, quarterly review |
| Users read `perp_funding` rows as risk-free | Hidden by default, distinct styling, breakeven-funding framing |
| Monitor mistaken for an execution signal | Cadence and labelling; this tab doesn't route orders |
| Settlement mismatch between options and futures | M1 verification, downgrade path |

---

## 12. Assessment (quant-dev)

| Lens | Result | Note |
|---|---|---|
| Functional | PASS | Parity, carry, executable edges and inventory rate for all four assets |
| Security | PASS | No new endpoints; user inputs (holdings, rates) stay client-side |
| Performance | PASS | ~5–30 venue calls per asset-minute, within public limits; no API change |
| Complexity | CLEAN | Reuses the adapter and pricer conventions; the only new data concepts are hedge matching and order-book confirmation |
| Edge cases | COVERED | Skew, missing futures, thin strikes, guard cases, fee rounding, contract rounding |
| Compliance | CLEAR | Read-only analytics, no order routing |

**Verdict: SHIP to developers**, with the fee table (step 2) as the one blocker before the model can run.

## 13. Compound

- **Rule:** Parity checks run on executable bid/ask, never on marks. Venue marks come from one vol surface, so parity holds on them by construction and tells you nothing.
- **Rule:** Inverse options parity: F = K / (1 − (c − p)), hedged with inverse futures at USD notional K per coin. Add it to the desk's standard formulas.
- **Rule:** Every arbitrage output carries a hedge type that states its residual risk. "Risk-free" is a property of the hedge, not of the formula.
- **Rule:** No model runs on placeholder fees. Null fees block the registry.
- **Reusable:** shared snapshot cache (D-11) for every Deribit model; the payoff-simulation test pattern for any static-replication claim.
