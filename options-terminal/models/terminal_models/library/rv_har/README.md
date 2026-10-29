# rv_har

Fills slot `vrp_signal`. Full methodology: [`RV_FORECAST_IMPLEMENTATION_GUIDE.md`](../../../../RV_FORECAST_IMPLEMENTATION_GUIDE.md) §3–6 and its amendments VRP-A-1…15. Payload: §12.

**Registered as shadow** (`production: null`) in `models/registry.yaml` for BTC, ETH, SOL and HYPE. Promotion is a separate PR citing the §9 validation report (P6).

**Method.**
- Target: log daily realized variance, `y_d = ln RV_d`, on completed UTC days from the `rv_daily` input. A null or zero day is missing and is never filled. Each lag mean averages the non-null days of its window and needs 80% of them (4 of 5, 18 of 22; the 1-day lag needs its day).
- Model (VRP-A-1): one direct OLS per horizon h = 1..`horizon_days`, `y_{t+h} = b0 + Σ b_l · mean(y_{t−l+1..t}) + c · cal_{t+h}`, lags `har_lags`. The calendar columns are day-of-week dummies (Monday is the reference level) and one window dummy per event type, taken at the target day. Refit every run (VRP-A-4).
- Cone: empirical `quantiles` of each horizon's in-sample residuals. `RV^p(T) = √(365/T · Σ_{k≤T} exp(ŷ_k + q_k(p)))`; `rv_low` and `rv_high` are p = 0.1 and 0.9.
- Central forecast (VRP-A-15): the mean, not the median, because implied variance prices the mean. Duan smearing `s_k = mean(exp(e_k))` over horizon k's residuals, `v̂_k = exp(ŷ_k) · s_k` (`core/quant/rv.py:smearing_factors`, `mean_path`), and `rv_forecast = √(365/T · Σ_{k≤T} v̂_k)`. The same path feeds the decomposition, the plain-HAR baseline, the z history and `tools/validate_rv.py`.
- Decomposition on the mean path, in variance: event share with the event dummies zeroed over the horizon, seasonal share with every forecast day at the average day of the week (each day-of-week dummy 1/7, not the Monday reference), baseline the remainder (§12.2).
- ATM IV per tenor: `core/quant/surface.py:build_expiries` and `core/quant/metrics.py:atm_iv` on the same `chain` kind as bs_pricer (VRP-A-6). It runs after bs_pricer and parity in the minute-5 pass, usually past the tick cache's 20 s reuse, so its chain is often a fresh fetch up to about a minute after bs_pricer's. Surface params are copied from bs_pricer. Extrapolation flags pass through to the row's `flags`.
- z (VRP-A-5): BTC/ETH at 30 days only. An expanding-origin HAR+events forecast (fit on days ≤ t, refit every `history_refit_every` origins counted back from C, so C is always fresh) gives `RV̂30_t`; `vrp_30d_t = DVOL_t − RV̂30_t`; z is scored against the `z_window_days` values before t (ddof = 1), and needs `z_min_obs` of them. Every other cell is null with `z_unavailable`.
- Baselines in `context`: plain HAR (no calendar terms) 30d and EWMA (`ewma_lambda`).
- `resid_sd` is the h = 1 residual standard deviation with ddof = 1.

**Inputs:** Deribit `chain` and `index` (as bs_pricer, so the tick cache shares them), `rv_daily` from the venue in `params.rv_sources[asset]`, and Deribit `dvol_history` for BTC/ETH (optional: without it z is null with `z_insufficient_history`). A day is cached only once it has `min_coverage` of its bars; short or empty days are refetched every run.

**RV sources.** BTC/ETH use Deribit perp candles at 60-minute bars until Deribit's 5-minute depth and row cap are verified live (VRP-A-8). SOL/HYPE use Hyperliquid hourly candles, because no Deribit SOL/HYPE perpetual has been seen yet (`adapters/deribit/README.md`). Hyperliquid's ~5,000-candle cap is about 208 days at 1h, so `min_history_days` is 150 there and 365 for BTC/ETH; the completed-day cache grows the history over time.

**Event calendar.** `params.events` lists FOMC statement days, CPI release days and the 2024 halving, each with its source; Deribit monthly and quarterly opex are derived by rule (`core/quant/calendar.py`). The FOMC and CPI dates were checked via web search against federalreserve.gov and bls.gov on 2026-10-29 (guide §4). `params.events_listed_through` records how far each list is complete (FOMC 2027-12-31, CPI 2027-01-13); coverage is checked per tenor: a tenor whose window reaches past a list's date gets `event_calendar_incomplete` on its row and is not `signal_ok`, and `signal_enabled` turns false only once even the shortest tenor is uncovered. Every calendar edit bumps the version (M-11).

*Maintenance.* `context.calendar_days_left` is the longest tenor's margin in days (negative once it has lost coverage), and the view warns below 21. With the shipped lists the CPI list runs out first: tenors drop from 2026-11-13 (60d) to 2027-01-05 (7d, signal disabled) until the 2027 CPI dates are added. When BLS publishes its 2027 schedule, add each release at 08:30 ET in UTC (13:30Z under EST, 12:30Z under EDT) with its source page, move `events_listed_through.cpi`, and bump the minor version. FOMC runs out at 2027-12-31 (the 2027 schedule was published on 2025-09-05); add 2028 the same way before 2027-10.

**Output:** composite `signal`, `context`, `cone`, `history`, `events` (guide §12).

**Fails visibly, never silently:**
- fewer non-null RV days than `min_history_days[asset]`, too few rows for a horizon, or no forecast origin (C itself has no RV, or a lag window ending at C has too few days) → `insufficient_history`;
- non-finite or non-positive forecast variance, non-finite fit diagnostics, or a cone value outside `rv_sanity` → `degenerate_forecast` (VRP-A-13);
- no chain expiry with enough nodes → `insufficient_market_data`;
- `rv_low ≤ rv_forecast ≤ rv_high` broken on any tenor → `vrp_band_ordering` (a model bug);
- RV input lagging (its last day closes before the latest 00:00 UTC ≤ `as_of`) → `signal_enabled` false with `rv_input_lagging`, and no row is `signal_ok`;
- an event list in `events_listed_through` ending before a tenor's window → that row flagged `event_calendar_incomplete` and not `signal_ok`; before even the shortest tenor's → `signal_enabled` false with the same flag.

## Validation (guide §9, P6)

`tools/validate_rv.py` gates promotion from shadow. It runs only on a recording that includes rv_har (`tools/record_deribit.py`), replaying each asset's `rv_daily` and BTC/ETH `dvol_history` through the production adapters:

```sh
cd options-terminal/models
uv run python -m terminal_models.tools.validate_rv --recording <recording dir> \
  --out terminal_models/library/rv_har/validation/<recording name>.json \
  [--assets BTC,ETH] [--folds 8] [--z-threshold 1.0] [--bias-tolerance 0.10]
```

- Expanding-origin backtest over every origin with `min_history_days[asset]` of prior RV, in `--folds` contiguous folds; each fold refits on days ≤ its first origin and holds that fit (no look-ahead). It compares rv_har (HAR + day-of-week + events), plain HAR and EWMA(`ewma_lambda`), using the core/quant/rv.py functions and this manifest's params.
- MASE and RMSSE per horizon 1..`horizon_days` on log daily variance (averaged over horizons) and per tenor on annualized RV over days t+1..t+T, each split all / ex-event / in-event (target day, or any day of the tenor window, inside an event window). Bias is on ex-event target days, in RMS vol (`√(365 · mean variance)`, the scale the mean forecast is unbiased on, VRP-A-15).
- Signal P&L for BTC/ETH at 30d is a variance-swap proxy (VRP-A-10): short at DVOL², P&L = DVOL² − realized RV² over the next 30 days, jump days included. The rule is short when vrp_z > `--z-threshold`, DVOL > RV^0.9(30) and comp_event(30) ≤ `event_hump_share`; naive is always short. Entries are non-overlapping (every 30 days back from the last matured window); Sharpe = mean/sd · √(365/30).
- Gates: every list in `events_listed_through` complete through the sample (`event_calendar_complete`); RMSSE ≤ 1.05 × HAR ex-event; RMSSE < HAR in event windows; |relative bias| ≤ `--bias-tolerance`; rule Sharpe > 0 and above naive. A gate without enough data is `unavailable`, which fails; SOL/HYPE have no DVOL, so their P&L gate is `not_applicable`.
- The report records the recording's manifest sha256, each input's hash (the runner's `input_hash`), the params hash, the git SHA and the config. Exit 0 when every asset passes, 1 otherwise (the report is written either way), 2 when the recording can't be used.

**Thresholds, pre-declared (guide §9).** Bias tolerance 0.10: about 5 vol points at 50% vol, tight enough to catch a structural bias such as the mean/median gap, loose enough for sampling noise over about 20 non-overlapping 30-day windows. z entry threshold 1.0: fixed before scoring, because choosing it on the backtest that judges it would overfit. The report's `pnl_proxy.z_sweep` shows Sharpe and `n_short` at z ∈ {0.5, 1.0, 1.5, 2.0}, marked `informational`; it never gates. Change a threshold only in a reviewed PR before the next recording is scored.

The promotion PR commits the report and cites it.

## To confirm on first live access

`make probe` (read-only) answers the first four and writes `validation/venue_probe.json`; commit it with the promotion PR, next to the `validate_rv` report.

- [ ] Deribit 5-minute candle depth and the `get_tradingview_chart_data` row cap. v1 stays hourly; moving BTC/ETH to `resolution: "5"` is a later versioned change, gated on that depth and a `validate_rv` comparison.
- [ ] Hyperliquid candle cap per interval (a lower cap is split around: more calls, same coverage).
- [ ] Hyperliquid accepts a `candleSnapshot` `endTime` in the future (the clock window runs to `as_of` + 10 min; if refused, the clock falls back to `endTime = as_of`).
- [ ] The UTC hour Deribit's `1D` DVOL candles open at. Non-zero: `dvol_history.close_offset_h` is set and every run warns `dvol_offset`.
- [ ] Whether Deribit lists SOL/HYPE perpetuals; if so, decide whether to switch their RV source.
- [x] Every `params.events` date against the Fed and BLS calendars; the CPI dates through 2027-01-13 and the 2027 FOMC dates are listed (1.1.0, checked via web search on 2026-10-29).
- [ ] The 2027 CPI dates once BLS publishes them (see *Maintenance* above).
- [ ] Run time of the minute-5 pass on real lookbacks (a shipped-params run is ~2 s here).
- [ ] `tools/validate_rv.py` on the first recording that includes rv_har; commit the report.

## Tests

`tests/` run the model on TEST-ONLY synthetic inputs (`tests/synthetic.py`). Golden tests on recorded snapshots run once a recording that includes rv_har is committed (`tests/test_golden.py`).
