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
  - It runs at minute 5 of each hour, in the same `run-due` pass as bs_pricer and parity_monitor and after them (registry order). The per-tick snapshot cache (Parity A-3) reuses a chain only for `TICK_MAX_REUSE_S` (20 s), so vrp usually fetches its own chain a few seconds to a minute after bs_pricer's. The monitor's "ATM 30d" (bs_pricer) and "VRP fcst 30d" (this slot's `iv_impl`) can therefore come from chains up to about a minute apart; each envelope's `inputs` records which.
  - Run cost: the expanding-origin history (§12.6) refits every `history_refit_every` origins, so a run at the shipped params takes a few seconds and the four scopes don't push the pass past the next minute's parity run (tested at shipped params).
- **VRP-A-7 Assets: BTC, ETH, SOL and HYPE.**
  - RV history comes from Deribit perpetual candles when Deribit lists a perpetual, otherwise from Hyperliquid candles (`adapters/hyperliquid`).
  - No placeholder prices: a scope with too little history fails with `insufficient_history`.
  - DVOL, and therefore z, exists only for BTC and ETH.
- **VRP-A-8 Data: the `rv_daily` input kind.**
  - Takes `venue`, `instrument`, `resolution` and `lookback_days`.
  - Pages through candles and returns completed UTC days only (day boundary 00:00), as `{day[], rv[], n_bars[]}` with `RV_d = Σ r²` over log close-to-close returns.
  - A day below `min_coverage` of its expected bars is `null`, never filled. A return that reaches back past more than one bar into an earlier day (the first return after a missing day) is dropped, so a missing day's variance never lands on the next day.
  - Completed days are immutable, so the adapter caches them per (venue, instrument, resolution, day) under `SNAPSHOT_DIR/rv_daily/` and only fetches new days. Only a day with at least `min_coverage` of its bars is cached; an empty or short day (outage, transient `no_data`, a page cut short by a lower row cap than assumed) is fetched again on the next run.
  - A HAR lag mean averages the non-null days in its window and needs 80% of them (4 of 5, 18 of 22; the 1-day lag needs its day), so one null day no longer blocks the forecast origin for 22 days.
  - Hyperliquid serves only about the latest 5,000 candles per interval (unverified; check live). That is roughly 17 days at 5-min bars and 208 days at hourly bars, so `resolution` is set per asset in the manifest. The cache grows the history over time.
  - Deribit's 5-min depth and row cap must be checked live before BTC/ETH move from hourly to 5-min bars.
- **VRP-A-9 Event calendar.**
  - **Where it lives:** the manifest's `params.events`, not a database table. Each entry is `{type, scope, ts, source}`.
  - **Rule-derived events:** monthly and quarterly opex are derived (Deribit expiry: last Friday of the month, 08:00 UTC; quarterly in Mar/Jun/Sep/Dec) by `core/quant/calendar.py`.
  - **Windows:** `params.event_windows` is the per-type config map.
  - **Edits:** every calendar edit is reviewed in a PR and bumps the version (M-11), because it changes output.
  - **Coverage:** `params.events_listed_through` gives, per listed scheduled type (FOMC, CPI), the date through which its list is complete. When the horizon plus the widest window reaches past it, recent fit days and the horizon would read "no event" where there may be one, so the run sets `signal_enabled` false with `event_calendar_incomplete` and flags every signal row (§12.3, §12.4). `tools/validate_rv.py` fails its `event_calendar_complete` gate on the same test over the sample.
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

**`tools/validate_rv.py` (as implemented, P6).** It runs only on a recording that includes rv_har:

```sh
uv run python -m terminal_models.tools.validate_rv --recording <dir> --out <report.json> \
  [--assets BTC,ETH] [--folds 8] [--z-threshold 1.0] [--bias-tolerance 0.10]
```

- **Inputs.** Each asset's `rv_daily` and, for BTC/ETH, `dvol_history` are replayed through the production adapters with the run's own InputSpecs. Nothing is fetched live and nothing is filled.
- **Backtest.** It uses every origin with `min_history_days[asset]` of prior RV, split into `--folds` contiguous folds. Each fold refits on days ≤ its first origin and holds the fit through the fold.
- **Forecasters.** All three use the `core/quant/rv.py` functions and the rv_har manifest params:
  - rv_har (HAR + day-of-week + events, median path as in `signal`);
  - plain HAR;
  - EWMA(0.94).
- **Accuracy.** MASE and RMSSE are computed on two bases, each split all / ex-event / in-event:
  - per horizon 1..60 on log daily variance, then averaged over horizons; the scale is the in-sample one-step naive before the first origin;
  - per tenor 7/14/30/60 on annualized RV over days t+1..t+T, with the previous T-day window as naive.
  - A horizon pair is in-event when its target day is in an in-scope event window; a tenor window is in-event when any of its days is.
- **Bias.** Mean forecast − actual in annualized daily vol, on ex-event target days, pooled over horizons.
- **P&L proxy (BTC/ETH, 30d).**
  - The rule is short when `vrp_z` > `--z-threshold`, DVOL > RV^0.9(30) (band) and `comp_event(30)` ≤ `event_hump_share`. Naive is always short. `vrp_z` is scored as in §12.6 on the backtest's 30d forecasts.
  - P&L = DVOL² − realized RV² over the next 30 days, jump days included.
  - Entries are non-overlapping, every 30 days back from the last matured window. Sharpe = mean/sd(ddof = 1) · √(365/30); a flat period counts as 0 for the rule.
- **Gates.** Evaluated per asset as `pass`, `fail`, `unavailable` (too little data, which counts as a fail) or `not_applicable` (no DVOL basis):
  - RMSSE(rv_har) ≤ 1.05 × RMSSE(HAR) ex-event;
  - RMSSE(rv_har) < RMSSE(HAR) in event windows;
  - |bias| / mean actual ≤ `--bias-tolerance` (0.10 by default: the table above gives no number);
  - rule Sharpe > 0 **and** > naive Sharpe.
- **Report.** The JSON report holds the recording's manifest sha256, per-input hashes (the runner's `input_hash`), the params hash, the git SHA, the config, per-asset metrics and the gates.
- **Exit code.** 0 when every asset passes, 1 otherwise, 2 when the recording is unusable.

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

## 12. Payload (v1)

The exact composite payload `rv_har` emits and `views/vrp_signal` reads. It is legal under `contracts/schemas/payloads/composite/v1.json` and the C-4 shape checks in `core/validation.py` (`validate_payload_full`). Keys are stable (C-5); labels may change. Every section is always present, in this order, and every column, item and series listed is always present, even when all its values are `null`. A value that can't be computed is `null` with a flag, never 0 (C-3).

### 12.1 Time conventions

- **Day close.** A UTC day d runs `[00:00, 24:00)`. Its close is the 00:00 UTC that ends it. Every day timestamp in this payload is a **close**.
- **`C`**, the forecast origin, is the close of the last completed UTC day in the `rv_daily` input. At run time `C` is normally the most recent 00:00 UTC before `as_of`.
- **Forecast day k** (k = 1..60) is `[C + (k−1)·1d, C + k·1d)`. Day 1 is the UTC day in progress at `as_of`. This is the "next 00:00" of VRP-A-14.
- **Tenor T** covers forecast days 1..T.

### 12.2 Forecast definitions

- `ŷ_k`: the direct h = k HAR+events forecast of log daily variance from origin `C` (VRP-A-1).
- `q_k(p)`: the empirical p-quantile of the h = k in-sample residuals, for p ∈ `params.quantiles` = [0.1, 0.5, 0.9].
- `v̂_k^p = exp(ŷ_k + q_k(p))`.
- `RV^p(T) = sqrt(365 / T · Σ_{k=1..T} v̂_k^p)`.
- `rv_forecast = RV^0.5`, `rv_low = RV^0.1` and `rv_high = RV^0.9`. Summing per-day quantiles gives a crude, comonotone cone, which is what §6's band rule asks for.
- **Decomposition**, all on the 0.5 path, in variance:
  - `V(T) = Σ_{k≤T} v̂_k^0.5`;
  - `V_noevt` uses the same fit with the event dummies zeroed over days 1..T;
  - `V_nodow` uses the same fit with every forecast day at the average day of the week: each day-of-week dummy at 1/7 (`core/quant/rv.py:counterfactuals`). Not the Monday reference level, which would make the share depend on the reference day and, with low-vol weekends, pin it at 0. Over whole weeks the share is therefore near 0; it is positive when the horizon is weighted toward high-vol days.
  - `s_evt = max(0, 1 − V_noevt/V)` and `s_sea = max(0, 1 − V_nodow/V)`.
  - If `s_evt + s_sea > 1`, both are divided by their sum.
  - `comp_event = s_evt`, `comp_seasonal = s_sea` and `comp_baseline = 1 − s_evt − s_sea`.
  - The three sum to 1 within 1e-12. "Seasonal" is day of week only in v1.
- **Event window overlap.** An in-scope event (GLOBAL, or this underlying) with UTC day `D`, `window_before` b and `window_after` a covers days `D−b .. D+a`.

### 12.3 Section `signal` (`table`)

One row per tenor in `params.tenors` ([7, 14, 30, 60] in v1), ascending.

| key | type | unit (quantity, scale) | null? | definition |
|---|---|---|---|---|
| `tenor_days` | number | `tenor_days`, `absolute` | no | T |
| `iv_impl` | number | `implied_vol`, `decimal` | yes, with `iv_unavailable` | `core/quant/metrics.py:atm_iv` at T days from `as_of` on the chain surface (VRP-A-6) |
| `rv_forecast` | number | `realized_vol`, `decimal` | no | `RV^0.5(T)` |
| `rv_low` | number | `realized_vol`, `decimal` | no | `RV^0.1(T)` |
| `rv_high` | number | `realized_vol`, `decimal` | no | `RV^0.9(T)` |
| `vrp` | number | `vol_spread`, `decimal` | iff `iv_impl` is null | `iv_impl − rv_forecast` |
| `vrp_z` | number | `zscore`, `absolute` | yes, with `z_unavailable` or `z_insufficient_history` | the 30d row for BTC/ETH only, on the DVOL basis: equals the last value of `history.vrp_z_30d` (VRP-A-5) |
| `comp_baseline` | number | `ratio`, `decimal` | no | §12.2 |
| `comp_seasonal` | number | `ratio`, `decimal` | no | §12.2 |
| `comp_event` | number | `ratio`, `decimal` | no | §12.2 |
| `event_in_horizon` | boolean | none | no | some in-scope event's window overlaps days 1..T |
| `rich` | boolean | none | iff `iv_impl` is null | `iv_impl > rv_high` (band discipline) |
| `event_hump` | boolean | none | no | `comp_event > params.event_hump_share` |
| `signal_ok` | boolean | none | no | `vrp_z` and `iv_impl` are non-null, `event_hump` is false, and `context.signal_enabled` is true |
| `flags` | string | none | no | Comma-joined codes, `""` when none (the parity `opportunities.flags` precedent). Codes are listed below. |

Row flag codes:
- `iv_unavailable`: no ATM IV at T, so `iv_impl`, `vrp` and `rich` are null.
- `tenor_extrapolated_short` / `tenor_extrapolated_long`: passed through from `atm_iv`.
- `z_unavailable`: no z basis for this cell. That is every tenor ≠ 30, and every SOL/HYPE row.
- `z_insufficient_history`: a basis exists (BTC/ETH 30d), but the trailing window has fewer than `params.z_min_obs` values, its standard deviation is 0, or DVOL at `C` is missing.
- `z_basis_dvol`: `vrp_z` is non-null and was scored on `DVOL − rv_forecast`, not on this row's `vrp` (VRP-A-5).
- `event_calendar_incomplete`: a listed event type's calendar stops before the horizon (VRP-A-9), so `event_in_horizon` and `comp_event` may miss an event; `signal_ok` is false.

### 12.4 Section `context` (`scalar_set`)

| key | value | unit (quantity, scale, convention) | null? | definition |
|---|---|---|---|---|
| `last_complete_day` | number | `timestamp`, `absolute`, `unix_seconds` | no | `C` as Unix epoch seconds |
| `n_days` | number | `count`, `absolute` | no | non-null RV days in the fit sample |
| `n_days_missing` | number | `count`, `absolute` | no | days in the lookback that are null (below `min_coverage`, VRP-A-8) |
| `fit_r2` | number | `ratio`, `decimal` | no | in-sample R² of the h = 1 HAR+events regression, in log variance |
| `resid_sd` | number | `other`, `absolute`, `log_variance` | no | standard deviation of the h = 1 residuals, in log daily variance |
| `rv_trailing_30d` | number | `realized_vol`, `decimal` | yes, with `insufficient_days` | backward-looking: `sqrt(365 · mean(RV_d))` over the non-null days among the 30 ending at `C`. Null if fewer than `params.window_min_days` are non-null. |
| `rv_ewma_30d` | number | `realized_vol`, `decimal` | no | EWMA(λ = 0.94) baseline 30d forecast: `sqrt(365 · σ²_EWMA,C)`, flat |
| `rv_har_30d` | number | `realized_vol`, `decimal` | no | plain-HAR baseline (no day-of-week or event terms) `RV^0.5(30)` |
| `dvol` | number | `implied_vol`, `decimal` | yes, with `dvol_unavailable` | Deribit DVOL daily close at `C` (BTC/ETH). SOL/HYPE are always null. |
| `vrp_30d_dvol` | number | `vol_spread`, `decimal` | iff `dvol` is null | `dvol − signal[30].rv_forecast`. Equals the last value of `history.vrp_30d`. |
| `vrp_z_mean` | number | `vol_spread`, `decimal` | iff `signal[30].vrp_z` is null | the trailing-window mean behind `vrp_z` |
| `vrp_z_sd` | number | `vol_spread`, `decimal` | iff `signal[30].vrp_z` is null | the trailing-window sample standard deviation (ddof = 1) |
| `vrp_z_n` | number | `count`, `absolute` | no | non-null values in that window; 0 off-basis |
| `event_hump_share` | number | `ratio`, `decimal` | no | the threshold in use, `params.event_hump_share` |
| `signal_enabled` | boolean | `other`, `absolute` | no | false when the RV input lags (`C` is earlier than the latest 00:00 UTC ≤ `as_of`; flag `rv_input_lagging`) or a listed event type's calendar stops before the horizon (flag `event_calendar_incomplete`, VRP-A-9). The item carries one flag per reason. |

**The view's gate.** The signal shows only when `signal_enabled` is true, the run is not `meta.stale` and the row's `signal_ok` is true (§10 staleness).

Strings never appear in `context` (VRP-A-12). The RV venue and instrument are in `envelope.inputs.sources`, and the model version is on the envelope.

### 12.5 Section `cone` (`curve`)

- `x`: label "Horizon (days)", unit `tenor_days`/`absolute`, values `1, 2, …, 60` (no nulls; the schema forbids them on an axis).
- Series, each with 60 values:

| key | unit | null? | value at x = T |
|---|---|---|---|
| `rv_forecast` | `realized_vol`, `decimal` | no | `RV^0.5(T)`, cumulative over days 1..T. At 7/14/30/60 it equals `signal.rv_forecast`. |
| `rv_low` | `realized_vol`, `decimal` | no | `RV^0.1(T)` |
| `rv_high` | `realized_vol`, `decimal` | no | `RV^0.9(T)` |
| `atm_iv` | `implied_vol`, `decimal` | sparse | `signal.iv_impl` at x ∈ `params.tenors`; null at every other x, and where `iv_impl` is null |
| `rv_path` | `realized_vol`, `decimal` | no | `sqrt(365 · v̂_T^0.5)`: forecast day T on its own, not cumulative (the plan's "path") |
| `rv_path_ex_event` | `realized_vol`, `decimal` | no | `rv_path` with the event dummies zeroed. The gap to `rv_path` is the event hump. |

### 12.6 Section `history` (`timeseries`)

- `t`: daily closes at 00:00 UTC (`YYYY-MM-DDT00:00:00Z`), strictly increasing and gap-free.
  - The last value is `C`; there are `min(params.history_days, available)` values, 365 in v1.
  - A day with no data keeps its `t` and gets nulls.
- **Expanding-origin forecast.** At each origin `t`, the HAR+events forecast uses a fit on days ≤ `t` only: no look-ahead. The model is refit every `params.history_refit_every` (5) origins, counted back from `C`, and held in between, so `C` is always fit fresh and the last value equals `signal[30].rv_forecast`.
  - The forecasts are computed for `history_days + z_window_days` origins so that z exists from the first emitted point. Only the last `history_days` are emitted.
  - An origin with fewer than `params.min_history_days` of prior data gets nulls.

| key | unit | null? | value at `t` |
|---|---|---|---|
| `rv_realized_30d` | `realized_vol`, `decimal` | yes | backward-looking realized: `sqrt(365 · mean(RV_d))` over the non-null days among the 30 ending at `t`; null below `window_min_days` |
| `rv_forecast_30d` | `realized_vol`, `decimal` | yes | the as-of forecast made at origin `t` for days `(t, t+30d]`, `RV^0.5(30)`. Its last value equals `signal[30].rv_forecast`. |
| `rv_forecast_30d_matured` | `realized_vol`, `decimal` | yes | `rv_forecast_30d` at `t − 30d`, i.e. the forecast for the window that ends at `t`. Compare it to `rv_realized_30d` (forecast against outcome). |
| `dvol` | `implied_vol`, `decimal` | yes | DVOL daily close at `t`; all null for SOL/HYPE |
| `vrp_30d` | `vol_spread`, `decimal` | yes | `dvol − rv_forecast_30d` |
| `vrp_z_30d` | `zscore`, `absolute` | yes | `(vrp_30d_t − mean) / sd` over the non-null `vrp_30d` in the `params.z_window_days` (90) days **before** `t` (exclusive of `t`, ddof = 1). Null below `z_min_obs` values or when sd = 0. |

### 12.7 Section `events` (`table`)

Every in-scope event, from the manifest plus rule-derived opex, whose window overlaps forecast days 1..max(tenors). Rows are sorted by `ts`, then `type`. The table may have zero rows.

| key | type | unit | null? | definition |
|---|---|---|---|---|
| `ts` | timestamp | none | no | event time, ISO UTC |
| `type` | string | none | no | one of `fomc`, `cpi`, `opex_monthly`, `opex_quarterly`, `token_unlock`, `halving`, `custom` |
| `scope` | string | none | no | `GLOBAL` or the underlying |
| `window_before` | number | `count`, `absolute` | no | days b, from `params.event_windows[type]` |
| `window_after` | number | `count`, `absolute` | no | days a |
| `horizon_day` | number | `count`, `absolute` | no | the forecast day k that contains `ts`. It is ≤ 0 for a past event whose window still overlaps day 1. |
| `source` | string | none | no | the manifest entry's `source`, or `rule` for derived opex |

### 12.8 Params referenced

These are `params.*` in the `rv_har` manifest, with v1 defaults:
- `tenors` [7, 14, 30, 60]
- `quantiles` [0.1, 0.5, 0.9]
- `event_hump_share` 0.35
- `z_window_days` 90
- `z_min_obs` 60
- `window_min_days` 24
- `history_days` 365
- `history_refit_every` 5
- `min_history_days` (per asset)
- `events`
- `events_listed_through`
- `event_windows`

### 12.9 Shape example (ILLUSTRATIVE values, not market data)

Arrays are truncated with `…`; a real payload has 4 signal rows, 60 cone points and up to 365 history points.

```json
{
  "sections": [
    {"key": "signal", "label": "Signal by tenor", "payload_kind": "table", "payload_version": "1", "payload": {
      "columns": [
        {"key": "tenor_days", "label": "Tenor", "type": "number", "unit": {"quantity": "tenor_days", "scale": "absolute"}},
        {"key": "iv_impl", "label": "ATM IV", "type": "number", "unit": {"quantity": "implied_vol", "scale": "decimal"}},
        {"key": "rv_forecast", "label": "RV forecast", "type": "number", "unit": {"quantity": "realized_vol", "scale": "decimal"}},
        {"key": "rv_low", "label": "RV low (q0.1)", "type": "number", "unit": {"quantity": "realized_vol", "scale": "decimal"}},
        {"key": "rv_high", "label": "RV high (q0.9)", "type": "number", "unit": {"quantity": "realized_vol", "scale": "decimal"}},
        {"key": "vrp", "label": "VRP", "type": "number", "unit": {"quantity": "vol_spread", "scale": "decimal"}},
        {"key": "vrp_z", "label": "VRP z", "type": "number", "unit": {"quantity": "zscore", "scale": "absolute"}},
        {"key": "comp_baseline", "label": "Baseline share", "type": "number", "unit": {"quantity": "ratio", "scale": "decimal"}},
        {"key": "comp_seasonal", "label": "Seasonal share", "type": "number", "unit": {"quantity": "ratio", "scale": "decimal"}},
        {"key": "comp_event", "label": "Event share", "type": "number", "unit": {"quantity": "ratio", "scale": "decimal"}},
        {"key": "event_in_horizon", "label": "Event in horizon", "type": "boolean"},
        {"key": "rich", "label": "Rich (IV > RV high)", "type": "boolean"},
        {"key": "event_hump", "label": "Event hump", "type": "boolean"},
        {"key": "signal_ok", "label": "Signal OK", "type": "boolean"},
        {"key": "flags", "label": "Flags", "type": "string"}
      ],
      "rows": [
        [7, 0.5, 0.4, 0.3, 0.6, 0.1, null, 0.8, 0.1, 0.1, false, false, false, false, "z_unavailable"],
        [30, 0.5, 0.4, 0.3, 0.6, 0.1, 1.0, 0.5, 0.1, 0.4, true, false, true, false, "z_basis_dvol"],
        "…"
      ]}},
    {"key": "context", "label": "Context", "payload_kind": "scalar_set", "payload_version": "1", "payload": {
      "items": [
        {"key": "last_complete_day", "label": "Last complete day", "value": 1800000000, "unit": {"quantity": "timestamp", "scale": "absolute", "convention": "unix_seconds"}},
        {"key": "dvol", "label": "DVOL", "value": null, "unit": {"quantity": "implied_vol", "scale": "decimal"}, "flags": ["dvol_unavailable"]},
        {"key": "signal_enabled", "label": "Signal enabled", "value": true, "unit": {"quantity": "other", "scale": "absolute"}},
        "…"
      ]}},
    {"key": "cone", "label": "RV cone vs ATM IV", "payload_kind": "curve", "payload_version": "1", "payload": {
      "x": {"label": "Horizon (days)", "unit": {"quantity": "tenor_days", "scale": "absolute"}, "values": [1, 2, "…", 60]},
      "series": [
        {"key": "rv_forecast", "label": "RV forecast", "unit": {"quantity": "realized_vol", "scale": "decimal"}, "values": [0.4, 0.4, "…"]},
        {"key": "atm_iv", "label": "ATM IV", "unit": {"quantity": "implied_vol", "scale": "decimal"}, "values": [null, null, "…"]},
        "…"
      ]}},
    {"key": "history", "label": "History (30d)", "payload_kind": "timeseries", "payload_version": "1", "payload": {
      "t": ["2000-01-01T00:00:00Z", "2000-01-02T00:00:00Z", "…"],
      "series": [
        {"key": "rv_realized_30d", "label": "Realized 30d", "unit": {"quantity": "realized_vol", "scale": "decimal"}, "values": [0.4, null, "…"]},
        {"key": "vrp_z_30d", "label": "VRP z (30d, DVOL)", "unit": {"quantity": "zscore", "scale": "absolute"}, "values": [null, null, "…"]},
        "…"
      ]}},
    {"key": "events", "label": "Events in horizon", "payload_kind": "table", "payload_version": "1", "payload": {
      "columns": [
        {"key": "ts", "label": "Time", "type": "timestamp"},
        {"key": "type", "label": "Type", "type": "string"},
        {"key": "scope", "label": "Scope", "type": "string"},
        {"key": "window_before", "label": "Days before", "type": "number", "unit": {"quantity": "count", "scale": "absolute"}},
        {"key": "window_after", "label": "Days after", "type": "number", "unit": {"quantity": "count", "scale": "absolute"}},
        {"key": "horizon_day", "label": "Forecast day", "type": "number", "unit": {"quantity": "count", "scale": "absolute"}},
        {"key": "source", "label": "Source", "type": "string"}
      ],
      "rows": [["2000-01-28T08:00:00Z", "opex_monthly", "GLOBAL", 1, 1, 27, "rule"]]}}
  ]
}
```
