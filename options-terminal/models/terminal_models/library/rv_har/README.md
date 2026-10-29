# rv_har

Fills slot `vrp_signal`. Full methodology: [`RV_FORECAST_IMPLEMENTATION_GUIDE.md`](../../../../RV_FORECAST_IMPLEMENTATION_GUIDE.md) §3–6 and its amendments VRP-A-1…14. Payload: §12.

**Registered as shadow** (`production: null`) in `models/registry.yaml` for BTC, ETH, SOL and HYPE. Promotion is a separate PR citing the §9 validation report (P6).

**Method.**
- Target: log daily realized variance, `y_d = ln RV_d`, on completed UTC days from the `rv_daily` input. A null or zero day is missing and is never filled. Each lag mean averages the non-null days of its window and needs 80% of them (4 of 5, 18 of 22; the 1-day lag needs its day).
- Model (VRP-A-1): one direct OLS per horizon h = 1..`horizon_days`, `y_{t+h} = b0 + Σ b_l · mean(y_{t−l+1..t}) + c · cal_{t+h}`, lags `har_lags`. The calendar columns are day-of-week dummies (Monday is the reference level) and one window dummy per event type, taken at the target day. Refit every run (VRP-A-4).
- Cone: empirical `quantiles` of each horizon's in-sample residuals. `RV^p(T) = √(365/T · Σ_{k≤T} exp(ŷ_k + q_k(p)))`; `rv_forecast`, `rv_low` and `rv_high` are p = 0.5, 0.1 and 0.9.
- Decomposition on the median path, in variance: event share with the event dummies zeroed over the horizon, seasonal share with every forecast day at the average day of the week (each day-of-week dummy 1/7, not the Monday reference), baseline the remainder (§12.2).
- ATM IV per tenor: `core/quant/surface.py:build_expiries` and `core/quant/metrics.py:atm_iv` on the same `chain` kind as bs_pricer (VRP-A-6). It runs after bs_pricer and parity in the minute-5 pass, usually past the tick cache's 20 s reuse, so its chain is often a fresh fetch up to about a minute after bs_pricer's. Surface params are copied from bs_pricer. Extrapolation flags pass through to the row's `flags`.
- z (VRP-A-5): BTC/ETH at 30 days only. An expanding-origin HAR+events forecast (fit on days ≤ t, refit every `history_refit_every` origins counted back from C, so C is always fresh) gives `RV̂30_t`; `vrp_30d_t = DVOL_t − RV̂30_t`; z is scored against the `z_window_days` values before t (ddof = 1), and needs `z_min_obs` of them. Every other cell is null with `z_unavailable`.
- Baselines in `context`: plain HAR (no calendar terms) 30d and EWMA (`ewma_lambda`).
- `resid_sd` is the h = 1 residual standard deviation with ddof = 1.

**Inputs:** Deribit `chain` and `index` (as bs_pricer, so the tick cache shares them), `rv_daily` from the venue in `params.rv_sources[asset]`, and Deribit `dvol_history` for BTC/ETH (optional: without it z is null with `z_insufficient_history`). A day is cached only once it has `min_coverage` of its bars; short or empty days are refetched every run.

**RV sources.** BTC/ETH use Deribit perp candles at 60-minute bars until Deribit's 5-minute depth and row cap are verified live (VRP-A-8). SOL/HYPE use Hyperliquid hourly candles, because no Deribit SOL/HYPE perpetual has been seen yet (`adapters/deribit/README.md`). Hyperliquid's ~5,000-candle cap is about 208 days at 1h, so `min_history_days` is 150 there and 365 for BTC/ETH; the completed-day cache grows the history over time.

**Event calendar.** `params.events` lists FOMC statement days, CPI release days and the 2024 halving, each with its source; Deribit monthly and quarterly opex are derived by rule (`core/quant/calendar.py`). The list is marked VERIFY: it was entered without access to federalreserve.gov or bls.gov. CPI dates from November 2025 onward and FOMC dates for 2027 are not listed yet. `params.events_listed_through` records how far each list is complete (FOMC 2026-12-31, CPI 2025-10-31); while a list stops before the horizon, `signal_enabled` is false with `event_calendar_incomplete`. Every calendar edit bumps the version (M-11).

**Output:** composite `signal`, `context`, `cone`, `history`, `events` (guide §12).

**Fails visibly, never silently:**
- fewer non-null RV days than `min_history_days[asset]`, too few rows for a horizon, or no forecast origin (C itself has no RV, or a lag window ending at C has too few days) → `insufficient_history`;
- non-finite or non-positive forecast variance, non-finite fit diagnostics, or a cone value outside `rv_sanity` → `degenerate_forecast` (VRP-A-13);
- no chain expiry with enough nodes → `insufficient_market_data`;
- `rv_low ≤ rv_forecast ≤ rv_high` broken on any tenor → `vrp_band_ordering` (a model bug);
- RV input lagging (its last day closes before the latest 00:00 UTC ≤ `as_of`) → `signal_enabled` false with `rv_input_lagging`, and no row is `signal_ok`;
- an event list in `events_listed_through` ending before the horizon → `signal_enabled` false with `event_calendar_incomplete`, the same flag on every row, and no row is `signal_ok`.

## Validation (guide §9, P6)

`tools/validate_rv.py` gates promotion from shadow. It runs only on a recording that includes rv_har (`tools/record_deribit.py`), replaying each asset's `rv_daily` and BTC/ETH `dvol_history` through the production adapters:

```sh
cd options-terminal/models
uv run python -m terminal_models.tools.validate_rv --recording <recording dir> \
  --out terminal_models/library/rv_har/validation/<recording name>.json \
  [--assets BTC,ETH] [--folds 8] [--z-threshold 1.0] [--bias-tolerance 0.10]
```

- Expanding-origin backtest over every origin with `min_history_days[asset]` of prior RV, in `--folds` contiguous folds; each fold refits on days ≤ its first origin and holds that fit (no look-ahead). It compares rv_har (HAR + day-of-week + events), plain HAR and EWMA(`ewma_lambda`), using the core/quant/rv.py functions and this manifest's params.
- MASE and RMSSE per horizon 1..`horizon_days` on log daily variance (averaged over horizons) and per tenor on annualized RV over days t+1..t+T, each split all / ex-event / in-event (target day, or any day of the tenor window, inside an event window). Bias is on ex-event target days, in annualized daily vol.
- Signal P&L for BTC/ETH at 30d is a variance-swap proxy (VRP-A-10): short at DVOL², P&L = DVOL² − realized RV² over the next 30 days, jump days included. The rule is short when vrp_z > `--z-threshold`, DVOL > RV^0.9(30) and comp_event(30) ≤ `event_hump_share`; naive is always short. Entries are non-overlapping (every 30 days back from the last matured window); Sharpe = mean/sd · √(365/30).
- Gates: every list in `events_listed_through` complete through the sample (`event_calendar_complete`); RMSSE ≤ 1.05 × HAR ex-event; RMSSE < HAR in event windows; |relative bias| ≤ `--bias-tolerance` (the guide gives no number); rule Sharpe > 0 and above naive. A gate without enough data is `unavailable`, which fails; SOL/HYPE have no DVOL, so their P&L gate is `not_applicable`.
- The report records the recording's manifest sha256, each input's hash (the runner's `input_hash`), the params hash, the git SHA and the config. Exit 0 when every asset passes, 1 otherwise (the report is written either way), 2 when the recording can't be used.

The promotion PR commits the report and cites it.

## To confirm on first live access

- [ ] Deribit 5-minute candle depth and the `get_tradingview_chart_data` row cap; then move BTC/ETH to `resolution: "5"` (version bump).
- [ ] Hyperliquid candle cap per interval.
- [ ] Whether Deribit lists SOL/HYPE perpetuals; if so, decide whether to switch their RV source.
- [ ] Every `params.events` date against the Fed and BLS calendars; add the missing CPI dates (Nov 2025 on) and the 2027 FOMC dates, and move `events_listed_through`.
- [ ] Hyperliquid accepts a `candleSnapshot` `endTime` in the future (the clock window runs to `as_of` + 10 min).
- [ ] Run time of the minute-5 pass on real lookbacks (a shipped-params run is ~2 s here).
- [ ] `tools/validate_rv.py` on the first recording that includes rv_har; commit the report.

## Tests

`tests/` run the model on TEST-ONLY synthetic inputs (`tests/synthetic.py`). Golden tests on recorded snapshots run once a recording that includes rv_har is committed (`tests/test_golden.py`).
