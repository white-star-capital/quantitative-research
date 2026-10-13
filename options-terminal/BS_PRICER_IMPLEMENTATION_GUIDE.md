# `bs_pricer` — Black-Scholes Pricer: Implementation Guide

**Status:** Draft v0.2 · **Owner:** Alejandro · **Date:** 2026-10-13
**Audience:** developers taking the first model live across `contracts/`, `models/`, `api/` and `frontend/`.
**Read first:** `README.md`, `CONTRACTS.md`, `MODELS.md`, `API.md`, `FRONTEND.md`. This guide only adds what is specific to the pricer; it does not repeat the base rules.

> **v0.2 changes from v0.1.** (1) Evaluation uses the generic route `GET /v1/slots/{slot_id}/evaluate`, dispatched by the slot YAML's `evaluation.evaluator`, so the API still hard-codes no slot (API A-27). The URL for this slot is unchanged. (2) Removed the go-live item "network restriction in place": README §8 Q1 is decided (research use, no restriction). (3) §5.3 uses the real `InputSpec(name, source, params)` signature. (4) The base-document amendments listed below are now made in those documents.

---

## 0. Summary

| | |
|---|---|
| **Slot** | `bs_pricer` |
| **First implementation** | `bs_pricer_black76` v1.0.0 |
| **Assets (scope `underlying`)** | `BTC`, `ETH`, `SOL`, `HYPE` |
| **Market data** | Deribit public API for all four. **No Volmex dependency.** |
| **Cadence** | Every 5 minutes per asset; stale after 15 minutes |
| **Outputs** | Market context, term structure, smiles, vol surface, per-expiry summary, full priced chain with greeks |
| **User inputs** | Price any strike/expiry, and evaluate an OTC quote against market, via `GET /v1/slots/bs_pricer/evaluate` |
| **Fidelity** | **F2.** Clear scope, multi-component. The one F3-flavoured risk is SOL/HYPE chain quality (§11). |

This is also the reference path for every future model: once `bs_pricer` is live, all of M0–M4 from `README.md` §4 has been exercised end to end.

### What ships, by folder

| Folder | Adds |
|---|---|
| `contracts/` | `slots/bs_pricer.yaml`, three vocabulary terms + `boolean` column type, payload schemas, pricing parity vectors, fixtures per asset |
| `models/` | `core/quant/` (Black-76, IV solver, interpolation, evaluate reference), `adapters/deribit/`, `library/bs_pricer_black76/`, registry entry, working runner |
| `api/` | `src/pricing/` (Rust port of the reference math), `GET /v1/slots/{slot_id}/evaluate`, log redaction for that route |
| `frontend/` | Generic renderers, cross-asset vol monitor, pricer slot view (Chain / Vol / Pricer & OTC / Data tabs) |

### Amendments to the base requirements

The OTC comparison needs computation at request time against arbitrary strikes and expiries. That doesn't fit "the API never computes," so this guide introduces one narrow, contracted exception. The base documents are updated to match:

- `README.md` §3.1: the API MAY expose **evaluation endpoints**. These are pure, deterministic functions over a *stored* run, with the math specified in Python and parity-tested in Rust against shared vectors. They never call Python, never fetch market data and never write.
- `API.md` §1, §3.2: evaluation endpoints are in scope under those rules.
- `CONTRACTS.md` §3: vocabulary additions and `boolean` column type (§4.2 here); `contracts/fixtures/pricing/` parity vectors; optional slot `evaluation` block.
- `FRONTEND.md` §1: slots MAY register a bespoke view as an additive layer; the generic renderers must still render the slot.
- `MODELS.md` M-12: a slot's first implementation may run shadow-only (`production: null`).

---

## 1. From the draft (`eth_quoter_v1.py`) to `bs_pricer`

The draft is a working covered-call screen for desk ETH inventory. Most of its pricing logic carries over; the desk-specific workflow does not belong in a public pricer.

### Keep

- Normal CDF via `erf`, and linear interpolation **in total variance** across expiries (the draft already does this for OTC quotes).
- OTC quote input with exactly one premium field (per unit, total, or % of spot). This maps to `premium_unit` in §6.
- Quote staleness (30 min) and spot-moved (1%) flags, the vol edge metric (IV_quote − IV_mark), comparison against the screen bid, and the payoff chart with a distribution overlay.

### Fix

| Draft behavior | Problem | `bs_pricer` behavior |
|---|---|---|
| `bs_call(S, K, T, vol)` prices off **spot** | Deribit IVs are quoted against each expiry's forward. With an 8% annualized basis, a 29-day ETH 2660 call at 55% IV is **$82.45 on spot vs $88.05 on the forward** (S = 2,473.59), ~7% of premium. (Illustrative inputs.) | Black-76 on the per-expiry forward (§3.4) |
| `implied_vol` clamps to 0.0001 or 5.0 when the premium is outside bounds | Silent wrong number: a quote below intrinsic shows as "0% vol" | Returns `null` plus a reason code |
| Interpolates IV linearly in **strike** space and extrapolates flat **without telling anyone** | Wrong shape on wings; users can't see they're off-grid | Total variance in log-moneyness; every extrapolation flagged |
| `deribit_name()` builds instrument names from strike and date | Breaks on fractional strikes (Deribit writes 9.2 as `9d2`), which SOL/HYPE will hit | Join on instrument metadata; never construct names |
| `get_historical_volatility` for RV | Deribit only provides this for BTC and ETH | Compute RV ourselves from hourly candles for all four (§3.6) |
| Calls only; delta only | Puts needed (protective puts, cash-secured puts); PMs need full greeks | Calls and puts; delta (forward and spot), gamma, vega, theta, P(ITM) |
| Payoff density `lognormal_pdf(x, mu=ln(spot), …)` | Wrong drift: median ends up at spot, mean above it | μ = ln F − ½σ²T |
| Listed-strike table forces expiry to 23:59 UTC but uses the 08:00 expiry's IV | ~16 h of time mismatched with the vol | Listed strikes use the venue expiry time; custom/OTC expiries take an explicit time and are interpolated |
| Headline APY compounds the premium as if re-earned every cycle | Overstates yield for PMs | APR headline; APY secondary and labelled |
| `urllib` with no retry; prints and local CSV | Not operable | Adapter with timeouts/retries; stored payload; API |

### Remove

- **Volmex EVIV / EVIV1D** inputs and `fallback_iv()`. Replaced by constant-maturity ATM IV computed from the chain (§3.6).
- **Synthetic Friday expiries** when Deribit is down. That is fabricated market data. If the venue is down, the run fails and users see the last good output marked stale.
- `MANUAL_SPOT` / `MANUAL_RV30` overrides. No hand-entered market data in production. (What-if spot shocks can become a frontend scenario feature later.)

### Move out (follow-on slot `covered_call_screen`, behind auth)

Inventory size, cost basis, the strike ladder around cost, the ACCEPT/NEGOTIATE/REJECT policy tied to cost basis, the Auros checklist, the roll rule and the quote-request text. These encode desk positions and policy. With no API key in v1, they must not be served by a public endpoint. They can be rebuilt on top of `bs_pricer` once auth exists.

---

## 2. Market data (Deribit, all four assets)

Deribit lists options on BTC, ETH, SOL and HYPE. A live check on 2026-10-13 confirmed `SOL_USDC` and `HYPE_USDC` option books and the `sol_usdc` / `hype_usdc` index names.

| Asset | Options book | Settlement | Spot index | Notes |
|---|---|---|---|---|
| BTC | `currency=BTC` (inverse, `BTC-…`) | BTC | `btc_usd` | Premium quoted in BTC |
| ETH | `currency=ETH` (inverse, `ETH-…`) | ETH | `eth_usd` | Premium quoted in ETH |
| SOL | `currency=USDC`, filter `base_currency=SOL` (`SOL_USDC-…`) | USDC (linear) | `sol_usdc` | Premium quoted in USDC |
| HYPE | `currency=USDC`, filter `base_currency=HYPE` (`HYPE_USDC-…`) | USDC (linear) | `hype_usdc` | Premium quoted in USDC; listed June 2026, thinnest book |

Deribit also lists linear `BTC_USDC` / `ETH_USDC` options. v1 uses the inverse books for BTC/ETH as the primary books; confirm relative depth during M1 and record the choice in the adapter README.

### 2.1 Endpoints (all public, no key)

| Purpose | Endpoint | Calls per run |
|---|---|---|
| Instrument metadata: strike, expiry, type, `contract_size`, settlement | `public/get_instruments?currency={BTC\|ETH\|USDC}&kind=option&expired=false` | 1 (cache 1 h; listings change daily) |
| Marks, bid/ask, underlying (forward), OI | `public/get_book_summary_by_currency?currency={BTC\|ETH\|USDC}&kind=option` | 1 |
| Spot | `public/get_index_price?index_name=…` | 1 |
| Hourly candles for RV | `public/get_tradingview_chart_data` on the asset's perpetual (`BTC-PERPETUAL`, `ETH-PERPETUAL`, `SOL_USDC-PERPETUAL`, `HYPE_USDC-PERPETUAL`); confirm names in M1 | 1 |
| DVOL cross-check (BTC/ETH only, display only) | `public/get_index_price?index_name={btcdvol_usdc\|ethdvol_usdc}` | 1 |

Note that SOL and HYPE share the `USDC` book summary call; the adapter SHOULD fetch it once per cadence tick and serve both scopes from the same snapshot.

### 2.2 Adapter rules (`models/terminal_models/adapters/deribit/`)

- **D-1** Timeout 10 s per call, 3 retries with exponential backoff and jitter on network errors and `5xx`; respect `429`.
- **D-2** Join book summary rows to instrument metadata by `instrument_name`. Never construct instrument names.
- **D-3** `contract_size` comes from metadata per instrument. It differs across assets; never hard-code it.
- **D-4** Record `snapshot_at` = Deribit's server timestamp on each response (`usOut`, or the row's `creation_timestamp`), not local fetch time.
- **D-5** Convert all prices to **USD per 1 unit of underlying**:
  - Inverse (BTC, ETH): `usd = price_coin × underlying_price` (the row's forward; this is Deribit's convention for marks).
  - Linear (SOL, HYPE): `usd = price_usdc` (treat USDC = USD; record the assumption).
- **D-6** Persist the raw response bodies to the snapshot store keyed by input hash (MODELS M-9) so any run can be replayed exactly.

### 2.3 Failure policy (replaces the Volmex fallback)

- **D-7** Each asset is its own scope, hence its own run. A HYPE failure never affects BTC.
- **D-8** Any required call failing after retries → failed run, `error.code = "venue_unavailable"`. The API keeps serving the last good run with `stale = true`.
- **D-9** An expiry with fewer than `min_nodes_per_expiry` valid nodes (§3.1) is dropped with warning `expiry_dropped_insufficient_nodes`. If no expiry survives → failed run, `insufficient_market_data`.
- **D-10** No synthetic or manually entered market data, ever.

---

## 3. Methodology

All formulas are per **1 unit of underlying**, in USD. Time is calendar time, 24/7: `T = (expiry − valuation_time) / (365 × 86400 s)`. `valuation_time` = the run's `as_of`.

### 3.1 Node selection

A **node** is a listed option whose mark IV is used to build the smile.

1. Keep **OTM** options only: calls with K ≥ F, puts with K < F (F = that expiry's `underlying_price`).
2. Require `mark_iv > 0`.
3. Drop expiries with T < `min_tte_hours` (default 1 h).
4. Compute `bid_iv` and `ask_iv` by inverting Black-76 on the USD bid/ask (§3.5) when present. Book summary gives prices, not IVs, and per-instrument ticker calls don't scale.
5. Node quality: `good` if two-sided with `ask_iv − bid_iv ≤ max_node_spread_vp` (default 10 vol points); `wide` if two-sided and wider; `one_sided` / `no_quote` otherwise. All qualities are kept as nodes (mark IV is still the venue's best estimate), but quality is shown everywhere and counted in the expiry summary.
6. Expiry needs ≥ `min_nodes_per_expiry` nodes: default 5 for BTC/ETH, 3 for SOL/HYPE (manifest params).

### 3.2 Smile per expiry

For expiry i with forward F_i and nodes at log-moneyness k_ij = ln(K_ij / F_i):

- σ_i(k) interpolates **linearly in total variance** w = σ²T_i between adjacent nodes (equivalently linear in σ², since T_i is fixed).
- Outside [k_min, k_max]: flat σ at the nearest node, flagged `strike_extrapolated`.

### 3.3 Across expiries

For target tenor T and moneyness k = ln(K / F(T)):

- Exact match (|T − T_i| < 60 s): use σ_i(k).
- T_a < T < T_b: w(k) = w_a(k) + (w_b(k) − w_a(k)) · (T − T_a) / (T_b − T_a), with w_a(k) = σ_a(k)² T_a. Then σ = √(w / T).
- T < T_first: σ = σ_first(k), flagged `tenor_extrapolated_short`.
- T > T_last: σ = σ_last(k), flagged `tenor_extrapolated_long`. Evaluation refuses T > 2 × T_last (§6).
- If w_b(k) < w_a(k) anywhere on the k grid: run warning `calendar_arbitrage` (no clamping).

**Forward at arbitrary T:** ln F linear in T through the points (0, ln S), (T_1, ln F_1), …, (T_n, ln F_n). Beyond T_n, extend the last segment's slope (flagged `tenor_extrapolated_long`).

This is deliberately simple (sticky-strike, piecewise linear). A fitted parametrization (e.g. SVI) is a future implementation in the same slot, and swapping it in is exactly what the registry is for.

### 3.4 Pricing: Black-76

With DF = e^(−rT), d1 = [ln(F/K) + ½σ²T] / (σ√T), d2 = d1 − σ√T:

- Call = DF · [F·N(d1) − K·N(d2)]
- Put = DF · [K·N(−d2) − F·N(−d1)]

`r` = manifest param `usd_rate`, **default 0.0**. With r = 0 our fair value reproduces Deribit's marks, which gives a built-in correctness check (§5.4). Non-zero r is supported but makes fair diverge from venue marks by the discount.

"Black-Scholes" in this guide means Black-76: the same model written on the forward. It's the right form because crypto option IVs are quoted against the forward, not spot.

### 3.5 Greeks and conventions (per 1 unit, USD)

| Field | Formula | Convention shown to users |
|---|---|---|
| `delta` | Call DF·N(d1); Put −DF·N(−d1) | Forward delta (hedge with that expiry's future) |
| `delta_spot` | `delta` × F/S | Spot delta (hedge with spot or the perp) |
| `gamma_1pct` | DF·φ(d1)/(F·σ√T) × (F/S)² × S × 0.01 | Change in spot delta for a 1% spot move |
| `vega_1pt` | DF·F·φ(d1)·√T / 100 | USD per 1 vol point |
| `theta_1d` | [−DF·F·φ(d1)·σ/(2√T) + r·V] / 365 | USD per calendar day, forward held fixed |
| `p_itm` | Call N(d2); Put N(−d2) | Risk-neutral probability of finishing ITM |
| `breakeven` | Call K + premium; Put K − premium | Spot at expiry |
| `yield_ann` | premium / S × 365 / dte | Simple annualized (APR). APY is a secondary, labelled column. |

Deltas are non-premium-adjusted. Premium-adjusted delta for coin-margined BTC/ETH positions is a v1.1 item.

**Implied vol solver:** bisection on [0.01, 5.0], price tolerance 1e-10, max 200 iterations. Deterministic and robust; speed is irrelevant at these volumes. Returns `null` with a reason instead of clamping:

| Reason | When |
|---|---|
| `below_intrinsic` | Price ≤ DF × intrinsic |
| `above_max_price` | Price ≥ DF·F (call) or DF·K (put) |
| `below_min_vol` / `above_max_vol` | Root outside [0.01, 5.0] |

**Reference implementation** (Python; tested: put-call parity to 1e-9, greeks vs finite differences, IV round-trip to 1e-8). This is `models/terminal_models/core/quant/black76.py` and is the spec the Rust port must match:

```python
import math

SQRT2PI = math.sqrt(2 * math.pi)
def N(x): return 0.5 * math.erfc(-x / math.sqrt(2))
def n(x): return math.exp(-0.5 * x * x) / SQRT2PI

def black76(F, K, T, sigma, r, is_call, S):
    """USD per 1 unit of underlying."""
    if not (F > 0 and K > 0 and S > 0):
        raise ValueError("non_positive_input")
    DF = math.exp(-r * T)
    if T <= 0 or sigma <= 0:
        intr = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return {"price": DF * intr}
    sd = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if is_call:
        price, delta, p_itm = DF * (F * N(d1) - K * N(d2)), DF * N(d1), N(d2)
    else:
        price, delta, p_itm = DF * (K * N(-d2) - F * N(-d1)), -DF * N(-d1), N(-d2)
    fs = F / S
    gamma_spot = DF * n(d1) / (F * sd) * fs * fs
    return {
        "price": price,
        "delta": delta,
        "delta_spot": delta * fs,
        "gamma_1pct": gamma_spot * S * 0.01,
        "vega_1pt": DF * F * n(d1) * math.sqrt(T) / 100,
        "theta_1d": (-DF * F * n(d1) * sigma / (2 * math.sqrt(T)) + r * price) / 365,
        "p_itm": p_itm,
    }

def implied_vol(F, K, T, price, r, is_call, S, lo=0.01, hi=5.0, tol=1e-10, maxit=200):
    DF = math.exp(-r * T)
    intrinsic = DF * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    upper = DF * (F if is_call else K)
    if price <= intrinsic: return None, "below_intrinsic"
    if price >= upper:     return None, "above_max_price"
    f = lambda s: black76(F, K, T, s, r, is_call, S)["price"] - price
    a, b = lo, hi
    if f(a) > 0: return None, "below_min_vol"
    if f(b) < 0: return None, "above_max_vol"
    for _ in range(maxit):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) < 1e-12:
            return m, None
        a, b = (a, m) if fm > 0 else (m, b)
    return 0.5 * (a + b), None
```

### 3.6 Market context metrics (replace Volmex)

| Metric | Definition |
|---|---|
| `atm_iv_{1d,7d,30d,90d}` | σ(k = 0, T = τ) from §3.3. Constant maturity, computed identically for all four assets. |
| `rv_{7d,30d}` | √(8760 × mean(r_h²)) over the last N×24 hourly log returns of perp closes (zero-mean estimator, 24/7 annualization) |
| `vrp_30d` | `atm_iv_30d − rv_30d`, in vol points. The desk playbook gate (≥ 5 vp) is a display threshold, not model logic. |
| `term_ratio` | `atm_iv_1d / atm_iv_30d`. `event_flag = term_ratio ≥ 1.3` (the draft's `EVENT_RATIO_FLAG`). Replaces Volmex EVIV1D/EVIV. |
| `rr25_30d`, `bf25_30d` | RR = σ(25Δ call) − σ(25Δ put); BF = ½(σ25C + σ25P) − σ_ATM. Forward delta. For each bracketing expiry, solve the strike where \|delta\| = 0.25 using that expiry's smile, then interpolate total variance to 30d. |
| `dvol` | BTC/ETH only, display only. DVOL is a 30d variance-style index, so it normally sits above ATM IV; don't warn on the gap. |
| `spot_change_24h` | From the same hourly candles |

`atm_iv_1d` is usually extrapolated short when the nearest expiry is more than 1 day out; the flag is shown next to it.

### 3.7 Grids for display

- **k grid:** −0.50 to +0.50 in steps of 0.025 (41 points). Smile and surface sections evaluate σ on this grid, **`null` outside each expiry's node range**. The display never extrapolates; only evaluation does, with a flag.
- **Chain:** every listed strike per expiry with \|forward delta\| between 0.02 and 0.98, calls and puts. Listed nodes use their mark IV; non-node rows (ITM side) use σ_i(k) from the OTM nodes, so a call and put at the same strike share one IV and put-call parity holds exactly.

---

## 4. Contracts (`contracts/`)

### 4.1 Slot definition — `contracts/slots/bs_pricer.yaml`

```yaml
slot_id: bs_pricer
title: Options pricer (Black-Scholes)
description: >
  Black-76 fair values, greeks and vol surface from Deribit marks, with
  evaluation of any strike/expiry and OTC quotes against market.
payload_kind: composite
payload_version: "1"
scope:
  dimensions: [underlying]
  allowed:
    underlying: [BTC, ETH, SOL, HYPE]
max_age_seconds: 900
conventions: >
  Prices USD per 1 unit of underlying. Black-76 on per-expiry forward, r from
  manifest (default 0). IV decimal. k = ln(K/F). Calendar time, ACT/365, 24/7.
  Greeks per section 3.5 of BS_PRICER_IMPLEMENTATION_GUIDE.md.
display:
  default_view: bs_pricer      # bespoke view; generic composite renderer as fallback
  order: 1
evaluation:
  evaluator: black76_surface_v1  # API dispatches GET /v1/slots/{slot_id}/evaluate on this
  quote_max_age_minutes: 30
  spot_move_flag_pct: 1.0
  max_tenor_multiple: 2.0
```

### 4.2 Vocabulary additions (contract PR)

| Addition | Use |
|---|---|
| `quantity: vol_spread` | Differences in vol (VRP, RR, BF, vol edge). Transmitted as decimal; displayed as vol points ("+5.2 vp"). |
| `quantity: probability` | `p_itm` |
| `quantity: yield_annualized` | `yield_ann`, APY |
| `quantity: rate` (exists) | Annualized forward basis |
| Column `type: boolean` | `is_node` and similar flags in tables |

### 4.3 Payload — composite sections

| Section key | Kind | Content |
|---|---|---|
| `context` | `scalar_set` | spot, spot_change_24h, atm_iv_1d/7d/30d/90d, rv_7d/30d, vrp_30d, term_ratio, event_flag, rr25_30d, bf25_30d, dvol (BTC/ETH), perp_last (for `spot_vs_perp`), usd_rate, n_expiries, n_nodes, n_nodes_good. Values may carry `flags` (e.g. `tenor_extrapolated_short`). |
| `term_structure` | `curve` | x = tenor_days of listed expiries; series `atm_iv`, `iv_25d_put`, `iv_25d_call` |
| `smile` | `curve` | x = k grid; one series per expiry (key `exp_YYYYMMDD`), `null` outside node range |
| `surface` | `surface` | x = k grid, y = tenor_days, z = IV (`null` outside node range) |
| `expiries` | `table` | expiry_ts, label, dte, forward, basis_ann, atm_iv, rr25, bf25, n_nodes, n_good, k_min, k_max, median_spread_vp |
| `chain` | `table` | Columns below |

`chain` columns:

| key | type / unit |
|---|---|
| `instrument_name` | string |
| `expiry_ts` | timestamp |
| `strike` | number / strike |
| `option_type` | string (`call`/`put`) |
| `k` | number / moneyness_log |
| `is_node` | boolean |
| `node_quality` | string (`good`/`wide`/`one_sided`/`no_quote`/`interpolated`) |
| `iv_mark`, `iv_bid`, `iv_ask` | number / implied_vol (decimal) |
| `venue_bid_usd`, `venue_ask_usd`, `venue_mark_usd` | number / price_usd |
| `fair_usd` | number / price_usd |
| `fair_pct_spot` | number / ratio (decimal) |
| `delta`, `delta_spot` | number / delta (`convention: forward`, `spot`) |
| `gamma_1pct` | number / gamma (`convention: delta_change_per_1pct_spot`) |
| `vega_1pt` | number / vega (`convention: usd_per_1vol_point`) |
| `theta_1d` | number / theta (`convention: usd_per_calendar_day`) |
| `p_itm` | number / probability |
| `breakeven` | number / price_usd |
| `yield_ann`, `apy` | number / yield_annualized |
| `open_interest` | number / count |
| `contract_size` | number / other |

Expected payload size: ~300–600 KB for BTC; smaller for the others. Within CONTRACTS C-6.

### 4.4 Fixtures and parity vectors

- `contracts/fixtures/valid/composite/bs_pricer_{btc,eth,sol,hype}.json`: one frozen run per asset. HYPE's fixture MUST include `wide` and `one_sided` nodes and a dropped expiry warning.
- `contracts/fixtures/invalid/composite/`: a `NaN` IV, a chain row with a missing `fair_usd` key, a `smile` series with the wrong length.
- `contracts/fixtures/pricing/`, generated by `python -m terminal_models.tools.gen_pricing_vectors` from the Python reference:
  - `black76.json`: ≥ 500 cases across moneyness, tenor (1 h to 2 y), vol (5%–300%), r ∈ {0, 0.05}, calls and puts.
  - `implied_vol.json`: round-trips plus every error reason.
  - `interpolation.json`: σ and F at strikes/tenors on, between and beyond nodes of a frozen run, including every flag.
  - `evaluate.json`: full `evaluate` request → expected response, built from the frozen BTC and HYPE runs.

---

## 5. Models (`models/`)

### 5.1 Files

```
models/terminal_models/
├── core/quant/
│   ├── black76.py          # §3.4–3.5 (reference implementation above)
│   ├── smile.py            # §3.2–3.3 interpolation + forward curve
│   ├── metrics.py          # §3.6 context metrics
│   └── evaluate.py         # §6 evaluation semantics (reference for the Rust port)
├── adapters/deribit/
│   ├── client.py           # D-1, D-4, D-6
│   ├── chains.py           # D-2, D-3, D-5: normalized OptionRow list per asset
│   └── README.md           # book choice per asset, conventions
├── library/bs_pricer_black76/
│   ├── manifest.yaml
│   ├── model.py
│   ├── README.md           # methodology summary + known limits (§11)
│   └── tests/
│       ├── fixtures/       # recorded Deribit snapshots per asset
│       └── test_model.py
└── tools/gen_pricing_vectors.py
```

### 5.2 Manifest

```yaml
implementation_id: bs_pricer_black76
version: 1.0.0
fills_slot: bs_pricer
owner: alejandro
description: Black-76 on Deribit marks; piecewise-linear total-variance surface.
schedule: "*/5 * * * *"
timeout_seconds: 60
max_input_age_seconds: 120
params:
  usd_rate: 0.0
  min_tte_hours: 1
  min_nodes_per_expiry: { BTC: 5, ETH: 5, SOL: 3, HYPE: 3 }
  max_node_spread_vp: 10
  chain_delta_range: [0.02, 0.98]
  k_grid: { min: -0.5, max: 0.5, step: 0.025 }
  rv_windows_days: [7, 30]
  atm_tenors_days: [1, 7, 30, 90]
  event_ratio_flag: 1.3
  candle_lookback_days: 31
bounds:                       # keyed by value key (MODELS §3.3); abs_ bounds the absolute value
  iv_mark: { min: 0.01, max: 5.0 }
  z: { min: 0.01, max: 5.0 }
  fair_usd: { min: 0 }
  abs_delta: { max: 1.0 }     # forward delta only: delta_spot = delta·F/S may exceed 1
sanity_checks:
  - calendar_arbitrage
  - put_call_parity
  - mark_reproduction
  - spot_vs_perp
```

### 5.3 `model.py` shape

```python
class BsPricerBlack76(Model):
    def required_inputs(self, scope):
        u = scope["underlying"]
        return [
            InputSpec("chain", "deribit", {"kind": "chain", "underlying": u}),   # metadata + book summary, joined
            InputSpec("index", "deribit", {"kind": "index", "underlying": u}),
            InputSpec("candles", "deribit", {"kind": "candles_1h", "underlying": u, "lookback_days": 31}),
            *([InputSpec("dvol", "deribit", {"kind": "dvol", "underlying": u})] if u in ("BTC", "ETH") else []),
        ]

    def run(self, ctx) -> dict:
        rows = ctx.inputs["chain"].data
        expiries = build_expiries(rows, ctx.params, ctx.as_of)   # nodes, forwards, drops → ctx.warn()
        surface = Surface(expiries)                               # core/quant/smile.py
        return composite([
            context_section(surface, ctx),
            term_structure_section(surface),
            smile_section(surface, ctx.params["k_grid"]),
            surface_section(surface, ctx.params["k_grid"]),
            expiries_section(surface),
            chain_section(rows, surface, ctx),
        ])
```

No I/O inside `run` (MODELS M-5). Warnings go through `ctx.warn(code, message)`.

### 5.4 Sanity checks (in `core/validation.py`)

| Check | Rule | Level |
|---|---|---|
| `mark_reproduction` | Node rows: \|fair_usd − venue_mark_usd\| / venue_mark_usd < 1% when `usd_rate = 0` | Warning per asset with count and worst instrument. **Hard failure if > 20% of nodes miss**, since that means a units or conversion bug. |
| `put_call_parity` | call − put − DF·(F − K) within 1e-6 × F for every strike pair | Hard (it holds by construction) |
| `calendar_arbitrage` | §3.3 | Warning |
| `spot_vs_perp` | \|index / perp last − 1\| < 2% | Warning |
| Bounds | Manifest `bounds` | Hard |

### 5.5 Tests

- Unit: black76 (parity, finite-difference greeks, limits T→0 and σ→0), implied_vol (round-trip and each reason), smile and forward interpolation (each flag), metrics (RV on a synthetic path with known vol, RR/BF on a synthetic smile with known skew).
- Model: golden test per asset from recorded snapshots; HYPE thin-book case; one asset with an expiry dropped; venue outage → failed run.
- Determinism: golden test run 10× with identical output (MODELS M-24).
- Replay: `runner replay <run_id>` on a stored fixture run reproduces the envelope exactly.

### 5.6 Registry

```yaml
slots:
  bs_pricer:
    production: null                   # first model: start as shadow, see §9
    shadow:
      - { implementation: bs_pricer_black76, version: 1.0.0 }
    scopes:
      - { underlying: BTC }
      - { underlying: ETH }
      - { underlying: SOL }
      - { underlying: HYPE }
```

`production: null` is allowed by MODELS M-12 for a slot in its initial shadow period. That's the rule for every first model in a slot.

---

## 6. API (`api/`)

### 6.1 Reading model output

No new code. `/v1/slots/bs_pricer/latest?underlying=ETH` and `/runs` work through the generic endpoints. During the shadow period, add `role=shadow&implementation=bs_pricer_black76`.

### 6.2 `GET /v1/slots/{slot_id}/evaluate` (here: `/v1/slots/bs_pricer/evaluate`)

Prices any strike and expiry against a stored run. With a premium, it evaluates an OTC quote. The route is generic: the API looks up the slot's `evaluation.evaluator` and dispatches to that evaluator; slots without one return 404 `no-evaluator`.

| Param | Required | Notes |
|---|---|---|
| `underlying` | yes | BTC \| ETH \| SOL \| HYPE |
| `run_id` | no | Defaults to the latest production run (during the shadow period: pass `role=shadow&implementation=…`, or a `run_id`). Pinning makes the answer reproducible. |
| `option_type` | yes | `call` \| `put` |
| `expiry` | yes | ISO 8601 UTC **with time** (`2026-11-06T23:59:00Z`). Date-only is rejected: OTC fixing times differ from Deribit's 08:00 UTC, and the time matters for short tenors. |
| `strike` | yes | > 0, within [0.2·F, 5·F] |
| `size` | no | Units of underlying, default 1 |
| `premium` | no | Enables quote evaluation |
| `premium_unit` | with premium | `usd_per_unit` \| `usd_total` \| `pct_spot` \| `coin_per_unit` |
| `side` | with premium | `sell` \| `buy`, from our perspective |
| `spot_ref` | no | Counterparty's reference spot; required for `pct_spot` |
| `quoted_at` | no | ISO UTC; enables the staleness flag |

There is **no counterparty parameter**. The API doesn't need it, and unknown params are rejected (API A-3).

**Semantics** (normative reference: `core/quant/evaluate.py`):

1. Load the run; `valuation_time = as_of`, S = run spot.
2. T from `expiry`; 400 `expiry_in_past` if T ≤ 0; 400 `tenor_out_of_range` if T > 2 × T_last.
3. F(T) and σ(k, T) per §3.3; σ_bid/σ_ask interpolated the same way from node bid/ask IVs where all bracketing nodes have them, else `null`.
4. Fair value and greeks per §3.4–3.5, per unit and × size.
5. If `premium` is given:
   - Normalize to USD per unit (`pct_spot` uses `spot_ref`; `coin_per_unit` uses S).
   - F_ref = F × spot_ref / S if `spot_ref` is given (keeps the basis), else F.
   - `iv_quote` = implied_vol(F_ref, …); `null` plus reason if unsolvable.
   - `pct_of_fair` = premium / fair.
   - `edge_per_unit` = (premium − fair) for `sell`, (fair − premium) for `buy`. Positive is always good for us.
   - `vol_edge_vp` = (iv_quote − σ_mark) × 100, sign flipped for `buy`. Positive is always good for us.
   - `screen_position`: screen bid/ask prices from σ_bid/σ_ask → `below_bid` \| `inside` \| `above_ask` \| `n/a`, plus `screen_position_frac` = (premium − bid) / (ask − bid).
   - `fair_at_spot_ref`: fair repriced with F_ref.
6. `distribution`: lognormal pdf at that expiry's ATM IV, μ = ln F − ½σ²T, 200 points over F·e^(±3σ√T). Labelled `lognormal_atm` (it ignores skew).

**Response:**

```json
{
  "data": {
    "run_id": "…", "as_of": "2026-10-13T17:00:00Z", "valuation_time": "2026-10-13T17:00:00Z",
    "underlying": "ETH", "spot": 2473.59,
    "contract": { "option_type": "call", "expiry": "2026-11-06T23:59:00Z", "strike": 2660,
                  "size": 100, "T": 0.0661, "dte": 24.12 },
    "market": { "forward": 2486.10, "iv_mark": 0.552, "iv_bid": 0.541, "iv_ask": 0.566,
                "interpolation": "interpolated",
                "bracketing_expiries": ["2026-10-30T08:00:00Z", "2026-11-27T08:00:00Z"],
                "screen_bid_usd": 58.9, "screen_ask_usd": 63.7 },
    "fair": { "per_unit_usd": 61.4, "total_usd": 6140.0, "pct_spot": 0.0248,
              "greeks_per_unit": { "delta": 0.33, "delta_spot": 0.33, "gamma_1pct": 0.022,
                                   "vega_1pt": 2.6, "theta_1d": -1.4, "p_itm": 0.30 },
              "greeks_position": { "...": "× size, signed for side when given" },
              "breakeven": 2721.4, "yield_ann": 0.375 },
    "quote": { "side": "sell", "premium_per_unit_usd": 60.59, "premium_total_usd": 6059.0,
               "iv_quote": 0.551, "iv_quote_status": "ok", "pct_of_fair": 0.987,
               "edge_per_unit_usd": -0.81, "edge_total_usd": -81.0, "vol_edge_vp": -0.1,
               "screen_position": "inside", "screen_position_frac": 0.35,
               "fair_at_spot_ref": 61.4, "yield_ann": 0.370 },
    "distribution": { "basis": "lognormal_atm", "x": [], "pdf": [] },
    "flags": [ { "code": "quote_stale", "message": "Quote is 5 days old" } ]
  },
  "meta": { "api_version": "1", "role": "production", "stale": false, "age_seconds": 140,
            "max_age_seconds": 900, "served_at": "2026-10-13T17:02:20Z" }
}
```

Numbers are illustrative only; they are not computed.

**Flags:** `quote_stale` (> `quote_max_age_minutes`), `spot_moved` (\|S/spot_ref − 1\| > `spot_move_flag_pct`), `run_stale`, `strike_extrapolated`, `tenor_extrapolated_short`, `tenor_extrapolated_long`, `low_quality_nodes` (any bracketing node not `good`), `expiry_time_differs_from_venue` (informational).

**Errors (problem+json, 400):** `expiry_in_past`, `tenor_out_of_range`, `strike_out_of_range`, `premium_requires_side`, `premium_requires_unit`, `pct_spot_requires_spot_ref`, `invalid_premium` (≤ 0), `invalid_spot_ref`, `invalid_option_type`, `invalid_size`, `run_not_found`, `run_failed` (pinned run has status failed).

`premium_unit=pct_spot` takes the premium in percent (`2.45` = 2.45% of `spot_ref`). `quote_stale` compares `quoted_at` with the run's `as_of`, so the answer is reproducible for a pinned run.

### 6.3 Rust implementation

```
api/src/pricing/
├── mod.rs
├── black76.rs      # price, greeks
├── iv.rs           # bisection, same bounds/tolerance/reasons as Python
├── surface.rs      # build from a stored run's `expiries` + `chain` (is_node rows); §3.2–3.3
└── evaluate.rs     # §6.2 semantics
```

- **P-1** f64 throughout. This is valuation, not settlement or token accounting, so the BigNumber rule doesn't apply. Settlement-grade amounts never pass through this code.
- **P-2** Build the `Surface` once per `run_id` and cache it (`moka`, keyed by run_id, ~16 entries). Runs are immutable, so the cache needs no invalidation.
- **P-3** **Parity:** CI runs every vector in `contracts/fixtures/pricing/`. Tolerances: price relative 1e-9 (absolute 1e-12 near zero), IV absolute 1e-8, greeks relative 1e-8, flags and error codes exact. Python is the reference; a disagreement is a Rust bug unless the Python tests prove otherwise.
- **P-4** Target p95 < 10 ms server-side with a warm surface cache.

### 6.4 Privacy and security for this route

OTC quotes are commercially sensitive, and GET puts them in URLs.

- **P-5** Access logs for `/evaluate` record path, status, latency and `request_id` only, **never the query string**. Applies to the API's `TraceLayer` and **to the reverse proxy/load balancer config** (deployment requirement; verify in M4).
- **P-6** `Cache-Control: no-store` and `Referrer-Policy: no-referrer` on `/evaluate` responses.
- **P-7** Separate rate-limit bucket: default 60 req/min per IP, burst 20.
- **P-8** The frontend never sends the counterparty label (§7.6).

If query-string exposure is still unacceptable after review, the fallback is a non-persisting `POST /evaluate` with the same semantics. That would be the API's only non-GET route, so it's a deliberate decision, not a default.

---

## 7. Frontend (`frontend/`)

Visual design, components and security follow `front_end_design.md` and `building_secure_front_ends.md` (FRONTEND.md §0). This section defines **what** to show and how the numbers are formatted. Layout and styling come from the design doc.

`bs_pricer` registers a bespoke view. The generic composite renderer must still render the slot (FRONTEND F-18 and the amended §1).

### 7.1 Who looks at what

| User | Primary question | Primary view |
|---|---|---|
| PM | Where is vol rich or cheap across our assets? Is premium worth selling? | Cross-asset monitor; context tiles; term structure vs RV |
| Trader | What's fair for this strike, where's the screen, is this OTC quote good? | Chain; Pricer & OTC |
| Quant | Is the surface sane? Where is the data thin? What did the model assume? | Vol tab; node quality; Data tab |

### 7.2 Cross-asset vol monitor (landing view for this slot)

One row per asset (BTC, ETH, SOL, HYPE), built from four `/latest` calls:

| Spot · 24h | ATM 7d | ATM 30d | ATM 90d | RV 30d | VRP 30d | RR25 30d | BF25 30d | Term ratio | Data | Updated |
|---|---|---|---|---|---|---|---|---|---|---|

- VRP coloured against a **user-set gate** (default 5 vp, stored per viewer in localStorage with try/catch). Show the sign and value, not colour alone.
- Term ratio shows an **Event** badge when `event_flag` is true.
- Data column: node count with good/total, and a warning icon if the run has warnings.
- Updated: age plus stale badge from `meta.stale`.
- Row click opens the asset view.

### 7.3 Asset view header

Tiles from `context`: spot (24h change), ATM 30d IV, RV 30d, VRP, RR25, BF25, term ratio, DVOL (BTC/ETH). Freshness, `as_of`, implementation/version and warnings per FRONTEND F-9/F-10.

Tabs: **Chain** · **Vol** · **Pricer & OTC** · **Data**.

### 7.4 Chain tab (traders)

- Expiry selector chips: date, DTE, ATM IV. Default: nearest expiry ≥ 7 DTE.
- **T-layout:** calls left, strike centre, puts right. A marker line sits at the forward between strikes; ITM cells are shaded.
- Column presets (switchable, remembered per viewer):
  - **Trader:** Bid · Fair · Ask (USD), IV mark, Δ
  - **Greeks:** Δ (spot), Γ 1%, Vega, Θ/day
  - **Yield:** Fair, % spot, Yield (ann.), P(ITM), Breakeven
  - **Vol:** IV bid · mark · ask, spread (vp), node quality
- Toggles:
  - Units: USD / % of spot / coin. Coin only for BTC/ETH, where premiums settle in coin.
  - Quantity: per unit / per contract (× `contract_size`) / per position (user enters size).
- Non-`good` nodes are dimmed, with a tooltip giving quality and spread. ITM rows marked `interpolated` show IV in italics.
- Row click → opens Pricer & OTC prefilled with that contract.

### 7.5 Vol tab (quants and PMs)

1. **Term structure:** ATM IV by tenor with a 25Δ put/call band, a dashed RV 30d reference line, and the VRP labelled at 30d.
2. **Smile overlay:** up to 4 expiries. x-axis toggle: log-moneyness / strike (K = F·e^k, converted per expiry from `expiries.forward`) / % of forward. Node points as markers with bid–ask IV error bars; interpolated line between.
3. **Surface heatmap:** x = moneyness, y = tenor, sequential colour scale with legend; `null` cells blank. 3D toggle optional.
4. **Skew table:** per expiry ATM IV, RR25, BF25, basis (ann.), nodes good/total, median spread.

### 7.6 Pricer & OTC tab (everyone)

**Inputs:** underlying (from view), call/put, expiry date **and** time (default 08:00 UTC; a one-click 23:59 option for OTC conventions), strike, size. For quote evaluation, optionally add side, premium + unit, spot ref, quoted at, and a **counterparty label held only in browser memory, never sent** (P-8).

**Outputs:**

1. **Verdict strip:** large `vol_edge_vp` and `edge_total_usd`, both signed "good for us", `pct_of_fair`, and a band classification against **user-set thresholds** (defaults from the desk playbook: floor 90% of fair, walk-away 80% of fair). Labels: *At/above floor*, *Negotiate zone*, *Below walk-away*. When selling, a quote below the screen bid also lands in *Negotiate zone*. Bands are display only; the API returns neutral metrics.
2. **Screen ladder:** a horizontal bar from screen bid to screen ask with markers for fair and the OTC quote. This is the fastest way for a trader to see "inside the market or not."
3. **Fair & greeks:** per unit and position, with signs from the side.
4. **Payoff at expiry:** P&L vs spot at expiry for the position, with the `distribution` pdf shaded underneath, and markers at spot, strike and breakeven. Users can optionally enter a cost basis held client-side only, which adds the covered-call vs holding comparison from the draft.
5. **Flags:** every `flags[]` entry in plain words.
6. **Blotter:** add multiple quotes. Each row calls `/evaluate`. The table shows label, expiry, strike, size, premium, fair, % fair, vol edge, screen position, band and flags, ranked by `vol_edge_vp`. Kept in session memory, with CSV export generated client-side (the counterparty label only ever lives in the export).

With no premium entered, the tab works as a plain pricer for any strike and expiry.

### 7.7 Data tab

`run_id`, implementation/version, `inputs.sources` with `snapshot_at`, code SHA, warnings, expiries table, node quality counts, download JSON and CSV per section.

### 7.8 Formatting conventions

| Value | Format | Example |
|---|---|---|
| IV | % with 1 dp | 55.2% |
| Vol differences (VRP, RR, BF, vol edge) | signed vol points, 1 dp | +5.2 vp |
| USD premium | 4 significant figures, at least 2 dp below $100 | $1,234 · $61.40 · $0.8523 |
| USD totals | thousands separators, no decimals ≥ $1,000 | $6,059 |
| Spot / strike | Asset precision: BTC 0 dp, ETH 1 dp, SOL 2 dp, HYPE 3 dp | 2,473.6 |
| Δ | 2 dp, signed | −0.33 |
| Γ 1% | 3 dp | 0.022 |
| Vega / Θ | USD, 2 significant figures minimum | $2.60 / −$1.40 per day |
| P(ITM), % of fair, yield | % with 0–1 dp | 30% · 98.7% · 37.5% ann. |
| Expiry | `06 Nov 26 08:00 UTC`; DTE to 1 dp under 7 days | 24 d · 2.4 d |
| Timestamps | UTC, local time on hover | |

Signs and colours are **side-aware**: "good for us" is positive and green whether we're buying or selling. Always pair colour with a sign or icon.

### 7.9 States specific to this slot

| State | Behavior |
|---|---|
| One asset stale or failed | Monitor row shows stale/failed badge; other assets unaffected |
| `/evaluate` 400 | Show the problem `detail` inline on the offending field |
| Quote IV `null` | Show the reason ("below intrinsic"); still show premium vs fair |
| Extrapolation flags | Badge next to the IV ("off-grid"), tooltip explains |
| HYPE thin book | Node-quality summary visible in header; dimmed chain rows |
| Shadow period | Views read the shadow implementation and show a SHADOW badge |

---

## 8. Test matrix

| Layer | Test | Gate |
|---|---|---|
| Contracts | Fixtures valid/invalid; pricing vectors present and regenerated reproducibly | CI |
| Models | Unit, golden per asset, thin-book, outage, determinism ×10, replay | CI |
| Models | `mark_reproduction` on recorded snapshots for all four assets | CI |
| API | Integration on fixture runs: latest, runs, evaluate (each param path, each error, each flag) | CI |
| API | Pricing parity vectors (P-3) | CI, blocking |
| API | Query string absent from logs for `/evaluate` (assert on captured log output) | CI |
| Frontend | Each section renders from each asset fixture; chain toggles; formatting table §7.8 as unit tests | CI |
| Frontend | OTC panel against mock `/evaluate` responses incl. every flag and error | CI |
| E2E | Seeded stack: monitor → ETH → chain row → pricer prefilled → add quote → blotter → CSV | Pre-release |
| UAT | Desk script, §9 step 6 | Go-live |

---

## 9. Go-live runbook

| # | Step | Owner | Done when |
|---|---|---|---|
| 1 | **Contracts PR:** slot YAML, vocabulary additions, fixtures per asset, pricing vectors | Models + API + FE reviewers | CI green on contract tests |
| 2 | **Models PR:** `core/quant`, Deribit adapter, `bs_pricer_black76`, registry as shadow | Models | All §5.5 tests green; local runs for all four assets write to local Postgres |
| 3 | **API PR:** `pricing/`, `/evaluate`, log redaction, rate-limit bucket | API | Parity vectors pass; §6.2 integration tests pass |
| 4 | **Frontend PR:** monitor, asset view, four tabs, against mock API | FE | Fixture renders, OTC panel tests, design and security sign-off |
| 5 | **Staging deploy:** migrations, registry sync, scheduler running `run-due` every 5 min, proxy log redaction configured | Platform | 24 h of runs for all four assets, < 1% failed |
| 6 | **Shadow validation (3 days, 24/7 market):** | Desk + Quant | All below hold |
|   | • `mark_reproduction` warnings < 5% of nodes for BTC/ETH; SOL/HYPE rates documented | | |
|   | • ATM 30d tracks DVOL direction for BTC/ETH (level gap expected) | | |
|   | • UAT: re-price the draft's sample quote (ETH 6 Nov 2660C, 100 ETH, $6,059 total, spot ref 2,473.59) and explain any difference from `eth_quoter_v1.py` (forward vs spot, expiry time) | | |
|   | • PM, trader and quant each complete a task on staging and sign off | | |
| 7 | **Promote:** registry PR `production: bs_pricer_black76 1.0.0`, link the validation notes | Alejandro | Merged and deployed |
| 8 | **Monitoring live:** alerts on (a) any scope stale > 15 min, (b) failed-run rate > 5%/h per asset, (c) `mark_reproduction` failure, (d) `/evaluate` 5xx, (e) API p95 above targets | Platform | Alerts fire in a staged test |
| 9 | **Retire the draft:** archive `eth_quoter_v1.py`; open backlog item for `covered_call_screen` (§1, Move out) | Alejandro | Done |

### Go-live checklist

- [ ] No Volmex references anywhere (`grep -ri volmex` returns nothing in `models/`)
- [ ] All four assets producing successful runs every 5 minutes
- [ ] `/latest` and `/evaluate` meet latency targets
- [ ] Query strings for `/evaluate` absent from API **and** proxy logs (checked in staging)
- [ ] Counterparty label confirmed absent from network requests (browser devtools check)
- [ ] Stale handling tested by stopping the scheduler for 20 minutes

---

## 10. Backlog after v1

Ordered by value:

1. `covered_call_screen` slot (inventory, cost basis, policy verdicts) once auth exists.
2. Premium-adjusted delta for BTC/ETH coin-margined positions.
3. Bybit adapter as a second source for HYPE/SOL cross-checks and as an outage fallback.
4. SVI implementation as a shadow candidate (exercises the swap path).
5. Vanna/volga, scenario grid (spot × vol shocks) in the pricer tab.
6. Smile-consistent density (Breeden–Litzenberger) to replace the lognormal overlay.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| SOL/HYPE books are thin; marks there are largely Deribit's model, not trades | Node quality everywhere; per-asset `min_nodes_per_expiry`; HYPE fixture tests; the monitor shows good/total nodes |
| Single venue: a Deribit outage makes all four assets stale | Accepted for v1; stale is visible; Bybit adapter in backlog |
| Inverse vs linear conversion bug produces plausible-looking wrong prices | `mark_reproduction` check is hard-fail above 20% of nodes; fixtures cover both kinds |
| OTC quote leakage via URLs | P-5 to P-8; POST fallback documented |
| Users mistake interpolated/extrapolated prices for screen prices | Interpolation status and flags on every evaluation; display grids never extrapolate |
| Lognormal distribution overlay misleads on skewed assets | Labelled `lognormal_atm`; replacement in backlog |
| No auth: anyone on the network can use the pricer | Pricer exposes only public market-derived data; desk-specific logic deliberately excluded (§1) |

---

## 12. Assessment (quant-dev)

| Lens | Result | Note |
|---|---|---|
| Functional | PASS | Four assets, no Volmex, OTC comparison, views for all three user types, full path across all components |
| Security | FLAG | OTC quotes in GET query strings; mitigated by P-5 to P-8 and a documented POST fallback |
| Performance | PASS | 5–6 venue calls per tick; evaluation < 10 ms with surface cache |
| Complexity | CLEAN | One narrow amendment to the base architecture; interpolation kept deliberately simple |
| Edge cases | COVERED | Thin books, outages, extrapolation, unsolvable IV, stale quotes, expiry-time mismatch |
| Compliance | CLEAR for internal; NEEDS_REVIEW if external | Read-only analytics; desk policy excluded from public outputs |

**Verdict: SHIP to developers.** Steps 1–4 of §9 can start now.

## 13. Compound

- **Rule:** Price crypto options on the expiry's forward (Black-76), never on spot. Add a mark-reproduction check whenever we price off a venue's IVs: it's the cheapest units-bug detector there is.
- **Rule:** Solvers return `null` plus a reason, never a clamped bound.
- **Rule:** Never construct venue instrument names; join on metadata.
- **Rule:** Evaluation endpoints are allowed only as pure functions over a stored run, with Python as the reference and Rust held to it by shared vectors. Reuse this pattern for every future "price this for me" feature.
- **Reusable:** `core/quant/black76.py`, `smile.py` and the Deribit adapter serve every future options slot; the pricing parity-vector setup serves any Python↔Rust math.
- **Failure pattern from the draft:** silent fallbacks (synthetic expiries, clamped IVs, unflagged extrapolation) produce confident wrong numbers. Fail visibly instead.
