# `vrp_signal`: RV-Forecast & VRP Signal Implementation Guide

**Status:** Draft v0.1 · **Owner:** Alejandro · **Date:** 2026-10-28
**Audience:** developers adding the third model across `contracts/`, `models/`, `api/` and `frontend/`.
**Read first:** `README.md`, `CONTRACTS.md`, `MODELS.md`, `API.md`, `FRONTEND.md`, `BS_PRICER_IMPLEMENTATION_GUIDE.md` and `PARITY_MONITOR_IMPLEMENTATION_GUIDE.md`. This slot reuses the Deribit adapter, the bs_pricer surface code and the parity view patterns.

---

## Amendments (2026-10-28, design review against the existing code)

These came out of mapping the guide onto the terminal. Where they conflict with the text below, they win.

- **VRP-A-1 Model family: HAR-RV, not NeuralProphet.**
  - **Why:** every NeuralProphet release, 0.9.0 and 1.0.0rc10 included, requires `numpy<2`. `models/` is on numpy 2.5.4, and the runner imports every model into one interpreter. NeuralProphet hasn't had a release since June 2024, and torch would add 1–5 GB to the image.
  - **v1 model (`rv_har`):** a HAR regression in log variance, which is §9's own fallback promoted to primary.
    - Regressors: lags of 1, 5 and 22 days, day-of-week dummies, and per-type event-window dummies.
    - One direct OLS per horizon h = 1..60, solved with `numpy.linalg.lstsq`.
    - The cone comes from empirical quantiles (0.1, 0.5, 0.9) of each horizon's in-sample residuals.
  - **Baselines in the same run:** plain HAR (no day-of-week or event terms) and EWMA (λ = 0.94).
  - §5's NeuralProphet config stays below for reference only.
- **VRP-A-2 Storage and API: a slot, not a table.** `vrp_signal` is a slot like `parity_monitor`: envelopes go to `model_runs`, and the generic read-only API serves them. There is no `vrp_signal` table, migration or Rust route; the API's catalog picks up the new slot on its own (API A-27–A-31). §7's routes map as follows:

  | §7 route | Existing route |
  |---|---|
  | `signal/{asset}` | `GET /v1/slots/vrp_signal/latest?underlying=X` |
  | `signal/{asset}/{tenor}` | one row of that payload's `signal` table |
  | `history/...` | the payload's `history` section, or `/v1/slots/vrp_signal/runs?underlying=X&from=&to=&include_payload=true` |
  | `decomp/...` | the `comp_*` columns of `signal` |
- **VRP-A-3 No recompute endpoint.** The API is GET-only (A-9). An off-cycle run is an ops action: `runner run --slot vrp_signal --underlying BTC`.
- **VRP-A-4 Refit every run; no model files.**
  - **Why:** a model may not do I/O in `run` (M-5), and replay must reproduce every run from its input hash (M-4). A fitted-model file loaded from disk would bypass the hash.
  - **How:** each run refits from its `rv_daily` input, which holds completed UTC days only. That input's content changes only when a day closes, so all runs within a day fit the same model: the "nightly refit" happens when the day turns. The hourly schedule is the "intraday re-score": the forecast stays fixed while IV moves.
  - The guide's `model_version` is `implementation_version` + `code.git_sha` + `inputs.hash`, all already on the envelope. `runner replay` proves any signal. Rows are append-only (C-10), so history is never regenerated.
- **VRP-A-5 z-score basis.** No IV history is stored anywhere: a run can't read past runs, and Deribit has no historical option IVs. v1 therefore computes `vrp_z` only for **BTC and ETH at 30 days**, against DVOL:
  - The current point is `DVOL_t − RV̂30_t`.
  - The history comes from an in-run expanding-origin HAR over the same `rv_daily` input, so the run stays pure and replayable.
  - z uses the trailing 90 days of that history.
  - Every other cell has `vrp_z = null` and a `z_unavailable` flag, never 0. A self-history input, the slot reading its own past runs, is the follow-up that gives a true ATM-IV z on every tenor.
- **VRP-A-6 IV source.** A slot cannot read another slot's output.
  - `build_expiries` moves from `bs_pricer_black76/model.py` to `core/quant/surface.py`, and bs_pricer is bumped to 1.0.1. Its output is unchanged.
  - `vrp_signal` reads the same `chain` kind and takes ATM IV at 7, 14, 30 and 60 days with the same interpolation, passing extrapolation flags through.
  - It runs at minute 5 of each hour, so the per-tick snapshot cache (Parity A-3) gives it the same chain as bs_pricer.
- **VRP-A-7 Assets: BTC, ETH, SOL and HYPE.**
  - RV history comes from Deribit perpetual candles when Deribit lists a perpetual, otherwise from Hyperliquid candles (`adapters/hyperliquid`).
  - No placeholder prices: a scope with too little history fails with `insufficient_history`.
  - DVOL, and therefore z, exists only for BTC and ETH.
- **VRP-A-8 Data: the `rv_daily` input kind.**
  - Takes `venue`, `instrument`, `resolution` and `lookback_days`.
  - Pages through candles and returns completed UTC days only (day boundary 00:00), as `{day[], rv[], n_bars[]}` with `RV_d = Σ r²` over log close-to-close returns.
  - A day below `min_coverage` of its expected bars is `null`, never filled.
  - Completed days are immutable, so the adapter caches them per (venue, instrument, resolution, day) under `SNAPSHOT_DIR/rv_daily/` and only fetches new days.
  - Hyperliquid serves only about the latest 5,000 candles per interval (unverified; check live). That is roughly 17 days at 5-min bars and 208 days at hourly bars, so `resolution` is set per asset in the manifest. The cache grows the history over time.
  - Deribit's 5-min depth and row cap must be checked live before BTC/ETH move from hourly to 5-min bars.
- **VRP-A-9 Event calendar.**
  - **Where it lives:** the manifest's `params.events`, not a database table. Each entry is `{type, scope, ts, source}`.
  - **Rule-derived events:** monthly and quarterly opex are derived (Deribit expiry: last Friday of the month, 08:00 UTC; quarterly in Mar/Jun/Sep/Dec) by `core/quant/calendar.py`.
  - **Windows:** `params.event_windows` is the per-type config map.
  - **Edits:** every calendar edit is reviewed in a PR and bumps the version (M-11), because it changes output.
  - This is reference data, not market data, so D-10 doesn't apply. Each entry carries a `source`.
- **VRP-A-10 Signal P&L gate (§9) on a proxy.** There are no historical option prices. The P&L backtest is a variance-swap proxy, BTC/ETH only:
  - short 30-day variance struck at DVOL²;
  - P&L ∝ `DVOL² − RV²` over the realized 30 days, jump days included.
  - It is labelled a proxy, not tradeable P&L.
- **VRP-A-11 Consumers out of scope.** The CID hedge engine and covered-call program aren't in this repo. Phase 6 here delivers a stable slot contract at `/v1/slots/vrp_signal/latest`.
- **VRP-A-12 Payload keys.**
  - **Strings:** `scalar_set` can't hold strings, so `asof` and `model_version` come from the envelope (`as_of`, `implementation_version`, `code.git_sha`, `inputs.hash`).
  - **z unit:** `zscore` is added to the unit vocabulary for `vrp_z`.
  - **Event flag:** the guide's `event_flag` is `event_in_horizon`, to avoid confusion with bs_pricer's EVENT badge (term-structure inversion).
  - **New columns:**
    - `rich` (band discipline, `iv_impl > rv_high`);
    - `event_hump` (`comp_event` share above `event_hump_share`);
    - `signal_ok` (z available, not hump-flagged, history sufficient).
- **VRP-A-13 Degenerate forecasts.**
  - A NaN, non-positive variance or a forecast outside `rv_sanity` raises `degenerate_forecast`, and the run fails.
  - The API keeps serving the last good run with `meta.stale`, and the view disables the signal when stale.
  - "Alert" means a failed run in `/runs?status=failed`; paging-style alerting is out of scope.
- **VRP-A-14 Horizon alignment.** The forecast covers whole UTC days starting at the next 00:00; an option runs to its 08:00 UTC expiry. The mismatch is under a day and is accepted for v1.

---

## 1. Scope & positioning

This module adds one thing to the terminal: a calendar- and event-aware realized-volatility (RV) forecast per asset and tenor, turned into a vol-risk-premium (VRP) rich/cheap signal. Everything already shipped stays as-is; this bolts on beside the existing Python models.

**What it adds**

- A new Python model, `rv_forecaster` (NeuralProphet), alongside the current model set. *(VRP-A-1: `rv_har`.)*
- A scheduled job: nightly refit, intraday re-score. *(VRP-A-4.)*
- A `vrp_signal` table in the existing DB, plus new read-only endpoints on the Axum API. *(VRP-A-2.)*
- One frontend panel: per asset/tenor, forecast RV vs implied, the VRP z-score, and the baseline / seasonal / event-hump decomposition behind it.

**What it deliberately does NOT do.** These stay in the modules that already own them:

- No vol surface, skew, smile, or term structure. It consumes your IV marks; it does not build them.
- No tail or jump distribution. Central estimate only; tail reserves stay in risk.
- No sizing and no order placement. It emits an advisory signal; the trader and the hedge engine decide size.
- Not a price or returns forecaster. The target is realized variance, never the underlying's direction.

**One line for the dev team:** this forecasts the *realized* side of the VRP trade and shows *why* it's rich or cheap. It never decides how much risk goes on.

## 2. Architecture & data flow

The whole module is one scheduled Python job plus a table and a few endpoints. Everything it touches already lives in the terminal. Data comes in at the top; the job writes one signal row set per run, and the API serves it to the panel and to the book.

```
  Price feed            Event calendar          IV marks
  (5-min OHLC)          (FOMC/opex/unlock)      (from surface)
       |                      |                      |
     prices                 events                implied
       v                      v                      v
+-- rv_forecaster -- nightly refit + hourly re-score (Python) -------------+
|                                                                          |
|   RV builder      -->    NeuralProphet     -->    VRP calc               |
|   (log daily var)        (fit + 60d path)         (vs IV, z, decomp)     |
|                                                                          |
+------------------------------------+-------------------------------------+
                                     v
                          vrp_signal table  (Postgres, existing DB)
                                     v
                          Axum API  (new read endpoints)
                                     v
          +--------------------------+--------------------------+
          v                          v                          v
   Frontend panel            CID hedge engine           Covered-call timing
   (RV vs IV, cone,          (hedged-quantity            (write / roll gate)
    decomp)                   input)
```

## 3. Data layer: the RV target

The model forecasts **log daily realized variance**. Log stabilizes the variance of the series and guarantees positivity when you exponentiate back. Variance, not vol, aggregates linearly over a tenor, so build the target in variance space and convert to vol only at the end.

**Estimator.** Realized variance from intraday log returns. Crypto trades 24/7, so there is no overnight gap to patch. Default sampling is 5-minute bars: fine enough to estimate daily variance, coarse enough to keep microstructure noise down. Make the bar size a config param; realized kernels and two-scale estimators are a v2 refinement, not needed for v1.

With 5-minute log returns on day d:

```latex
r_{d,i} = \ln\!\left(\frac{P_{d,i}}{P_{d,i-1}}\right), \qquad
RV_d = \sum_{i=1}^{n_d} r_{d,i}^{2}
```

`RV_d` is the realized variance for day d in daily units (n_d ≈ 288 bars). The model target is its log:

```latex
y_d = \ln(RV_d)
```

**Annualized vol**, for reading and for comparison to IV (365 calendar days, 24/7):

```latex
\sigma_d^{\text{ann}} = \sqrt{365 \cdot RV_d}
```

**Realized vol over an option's life** (what you score against implied): forecast each future day's variance, sum over the T-day tenor, and annualize:

```latex
\widehat{RV}(T) = \sqrt{\frac{365}{T}\sum_{k=1}^{T}\hat{v}_{k}}, \qquad \hat{v}_k = e^{\hat{y}_k}
```

**Notes for the builder.**
- Jumps inflate `RV_d`. That is correct here: you want *total* realized variance, including jump variance.
- Winsorize only obvious bad prints (feed glitches), not real moves.
- Timestamp everything in UTC, and close the day at 00:00 UTC so the nightly job runs on a sealed day.

**Input dataframe** handed to NeuralProphet (one per asset):

| column | type | meaning |
| --- | --- | --- |
| `ds` | timestamp (UTC, daily) | day close, 00:00 UTC |
| `y` | float | log daily realized variance |
| `log_funding` | float (optional) | lagged regressor, if used |
| `log_perp_vol` | float (optional) | lagged regressor, if used |

Event columns are not added here. They are injected via the events dataframe (next section).

## 4. Event calendar

Known catalysts are the one thing a HAR-style model can't handle and this one can. Each event type gets its own regressor with a window, so the model learns a distinct realized-vol hump for FOMC vs a token unlock rather than lumping them into one "event" bump. *(VRP-A-1: the HAR in v1 carries the same per-type event-window dummies.)*

**Table `event_calendar`** *(VRP-A-9: manifest `params.events`)*:

| column | type | meaning |
| --- | --- | --- |
| `event_id` | uuid | primary key |
| `asset` | text | `BTC`, `ETH`, or `GLOBAL` (macro hits all) |
| `event_type` | text (enum) | `fomc`, `cpi`, `opex_monthly`, `opex_quarterly`, `token_unlock`, `halving`, `custom` |
| `ts` | timestamptz | event time, UTC |
| `window_before` | int | days before to flag (i) |
| `window_after` | int | days after to flag (j) |
| `magnitude` | float, nullable | optional scale (e.g. unlock size as % of float) |

**Mapping to NeuralProphet.**
- One named event per `event_type`, each with its own window, via `add_events(name, lower_window=-i, upper_window=+j)`.
- Occurrences go in an events dataframe (`event`, `ds`), merged into history with `create_df_with_events`.
- Known future events **must** be present in the future events dataframe, or the model can't apply the hump when predicting forward.

```python
events_df = pd.DataFrame({
    "event": calendar_rows.event_type,      # one name per type
    "ds":    calendar_rows.ts.dt.floor("D"),
})
```

**Scoping rules the loader must enforce:**

- `GLOBAL` macro events (FOMC, CPI) attach to every asset's events frame; `token_unlock` attaches only to its own asset.
- Windows are per type, not global: FOMC `[-1, +1]`, quarterly opex `[-2, +1]`, large unlock `[-3, +2]`. Keep these in a config map, not hard-coded.
- **Funding resets are NOT events here.** At 8-hourly cadence on a daily model they are intraday structure: they belong to session seasonality in the hourly model, not the daily event set. Only promote something to an event if it is sparse relative to the model frequency.
- Country/exchange holidays: skip. Crypto doesn't close.

## 5. NeuralProphet model spec (reference only; VRP-A-1)

One daily model per asset. Start linear (no hidden layers): it's interpretable and, per the paper's ablations, close to the NN on short/medium horizons. Add a small NN only if the backtest shows a real gain at longer tenors.

| param | value | why |
| --- | --- | --- |
| `n_lags` | 30 | AR memory for vol clustering; set generously, let regularization prune |
| `n_forecasts` | 60 | max tenor in days you'll score |
| `ar_reg` | 0.5 | sparsifies the AR weights → interpretable clustering lags (1, 7, 30…) |
| `weekly_seasonality` | 6 | Fourier terms; day-of-week + weekend effect |
| `yearly_seasonality` | auto | on if ≥ 2y data, else off |
| `daily_seasonality` | False | no intraday structure at daily freq |
| `quantiles` | [0.1, 0.5, 0.9] | median path + a crude cone for the band rule |
| `normalize` | "standardize" | log target is already variance-stabilized |
| `loss_func` | "Huber" | robust to RV spikes |

**Reading the output.**
- Components are additive in log space, so an event hump's effect on variance is multiplicative: "FOMC multiplies expected variance by e^{event}".
- **Pin versions** and verify kwarg names against the installed build.

## 6. Forecast → VRP signal

Take the forward path from the latest origin, convert log-variance to variance, aggregate over the tenor, and compare to implied.

```latex
\widehat{RV}(T) = \sqrt{\frac{365}{T}\sum_{k=1}^{T}\hat{v}_{k}}, \qquad
\mathrm{VRP}(T) = \mathrm{IV}_{\text{impl}}(T) - \widehat{RV}(T), \qquad
z = \frac{\mathrm{VRP}(T) - \mu_{90}}{\sigma_{90}}
```

- `IV_impl(T)` is the ATM implied for that tenor, pulled from the existing surface.
- `μ_90` and `σ_90` are the trailing-90-day mean and std of `VRP(T)` *(VRP-A-5)*.
- **Positive z = implied rich vs forecast realized → favors selling vol.**
- Band discipline (Section 10): only call it rich when `IV_impl > RV_high`, the upper-cone forecast built the same way from the 0.9-quantile path.

**Component decomposition.** Split the forecast variance over the window into three shares, so the trader sees *where* the forecast comes from:

- `comp_baseline`: trend + intercept, and the AR terms
- `comp_seasonal`: weekly + yearly
- `comp_event`: sum of event components

v1 attributes counterfactually:
- event share = 1 − RV̂(T, event terms zeroed) / RV̂(T), in variance;
- seasonal share is computed the same way;
- the shares are clipped to ≥ 0 and the baseline is the remainder.

If `comp_event` is a large share, the "richness" may just be the market correctly pricing a known catalyst. Do not read it as free premium. Surface this as the headline next to the z-score.

**Signal row** (one per tenor; VRP-A-12 names):

| column | type | meaning |
| --- | --- | --- |
| `tenor_days` | int | 7, 14, 30, 60 |
| `iv_impl` | float | ATM implied, annualized |
| `rv_forecast` | float | median forecast RV, annualized |
| `rv_low` / `rv_high` | float | 0.1 / 0.9 cone, annualized |
| `vrp` | float | iv_impl − rv_forecast |
| `vrp_z` | float, nullable | trailing-90d z-score (VRP-A-5) |
| `comp_baseline` / `comp_seasonal` / `comp_event` | float | variance shares [0,1] |
| `event_in_horizon` | bool | any event inside the tenor |
| `rich` | bool | `iv_impl > rv_high` |
| `event_hump` | bool | `comp_event > event_hump_share` |

## 7. API contract

**Boundary decision: DB-backed, not a live Python call in the request path.** The Python job writes rows to Postgres, and Axum serves straight from the database. That keeps the request path pure Rust, deterministic and auditable: every signal the frontend shows is a persisted row with a model version. *(VRP-A-2: the rows are `model_runs` envelopes, served by the existing generic endpoints. VRP-A-3: no recompute endpoint.)*

**Response shape** of the guide's `/signal/{asset}/{tenor}`. The same data is one row of `signal` in `/v1/slots/vrp_signal/latest?underlying=BTC`:

```json
{
  "asset": "BTC", "tenor_days": 30, "asof": "2026-10-28T00:30:00Z",
  "iv_impl": 0.58, "rv_forecast": 0.49, "rv_low": 0.41, "rv_high": 0.63,
  "vrp": 0.09, "vrp_z": 1.8,
  "decomp": {"baseline": 0.70, "seasonal": 0.12, "event": 0.18},
  "event_flag": true, "model_version": "2026-10-28.a1b2c3d"
}
```

## 8. Scheduling, persistence, reproducibility

*VRP-A-4 replaces the two-cadence and saved-model design: one hourly schedule, refit per run from completed days, with reproducibility proven by `runner replay`.*

**Failure handling:** if a refit fails or produces a degenerate forecast (NaN, non-positive variance after exp, or forecast RV outside a sanity band), hold the previous row set and raise an alert. Never write a bad signal silently. The frontend shows the `asof` staleness, so the trader knows the forecast didn't update. *(VRP-A-13.)*

## 9. Validation & backtest

**Read this before building the rest.** Do not ship on the strength of a good-looking fit. Two separate backtests gate this module: forecast accuracy, then signal P&L.

**Forecast accuracy: benchmark against the right baselines.**

- Method: expanding-origin backtest. Use k folds: train on the past, roll the origin forward without refitting inside a fold, and refit per fold. Report MASE and RMSSE averaged over horizons.
- Baselines that matter, beyond naive: **EWMA** (RiskMetrics, λ = 0.94) and the real bar, **HAR-RV** (Corsi 2009: regress daily RV on its 1-day, 5-day and 22-day averages).

**The honest flag:** HAR-RV is the workhorse of realized-vol forecasting and is *hard to beat* on raw accuracy. The reasons to ship an event-aware model anyway, and the claims to hold it to:
1. **Event humps:** the win should show up *in event windows*.
2. **An interpretable decomposition.**
3. **One framework** across assets and tenors.

If it can't at least match HAR ex-event **and** beat it inside event windows, plain HAR is the right answer.

**Acceptance gates:**

| test | gate |
| --- | --- |
| RMSSE vs plain HAR, ex-event days | within 5% |
| RMSSE vs plain HAR, event windows | lower (beat) |
| Forecast RV bias | near zero on ordinary days (expect downward bias around tails; see §10) |
| Signal P&L Sharpe vs naive-sell | positive, net of tails (VRP-A-10 proxy) |

**Signal P&L backtest.** A good RV forecast is not a profitable signal. Backtest the actual rule, "sell vol when `vrp_z` > threshold, gated by the band and event rules", **including the jump days**. If the edge evaporates once the tails are in, the signal is timing noise. This test, not the accuracy test, decides whether it touches the book. `tools/validate_rv.py` runs these on recorded data and writes a report. Promotion from shadow is a PR that cites the report.

## 10. Risk guardrails & integration

The single rule that governs everything here: **this is a central estimate.** You get paid on the mean of realized vol and you die on the tail, and the model smooths the tail. It will *systematically under-forecast* tail realized vol.

**Hard guardrails (enforced in code, not convention):**

- **Sizing is never driven by this signal alone.** The signal is a gate on *direction*, never a lever on *size*.
- **Band discipline.** Treat as rich only when `iv_impl > rv_high`.
- **Event-hump rule.** If the `comp_event` share exceeds a threshold, flag it and don't auto-read the richness as premium.
- **Staleness.** If the forecast didn't refresh, the signal is disabled, not shown green.

**Integration points** (out of scope here, VRP-A-11):
- covered-call write/roll timing (`vrp_z` + band as a gate);
- the CID hedge engine (`rv_forecast` as the hedged-quantity input).

## 11. Build checklist (as implemented)

- **P0 Contracts:**
  - `contracts/slots/vrp_signal.yaml`;
  - the `zscore` unit;
  - this guide.
- **P1 Core math:**
  - `core/quant/rv.py`: RV, HAR, cone, decomposition, EWMA, backtest metrics;
  - `core/quant/calendar.py`;
  - `core/quant/surface.py`.
- **P2 Adapters:**
  - Deribit `rv_daily` and `dvol_history`;
  - the Hyperliquid adapter (`rv_daily`);
  - recorder support.
- **P3 Model:** `library/rv_har/` (manifest, model, sanity check `vrp_band_ordering`, tests).
- **P4 Fixtures and registry:** shadow registration for BTC/ETH/SOL/HYPE.
- **P5 Frontend:**
  - `views/vrp_signal/`;
  - a VRP tab on asset pages;
  - VRP monitor columns.
- **P6 Validation:** `tools/validate_rv.py` runs the §9 gates on recordings. Phase 6 risk wiring is out of scope.
- **Live checks** before trusting VRP-A-8:
  - Deribit's 5-min candle depth and row cap;
  - Hyperliquid's candle cap;
  - whether Deribit lists SOL and HYPE perpetuals.
