# results/

Run date: 2026-09-09 · universe fingerprint `af8fe4ee067f` · reproduce with `make start`

## Verdict: no evidence of skill

The search does not beat data with no signal. **Every one of the 19 signal-free
surrogate runs found a higher in-sample Sharpe than the real data did**
(null range +4.00 to +5.93, observed +3.32), giving an empirical p-value of
1.000. Out-of-sample the observed best (+0.735) sits above the null median
(−0.195) but inside its spread, p = 0.200 — unremarkable, and not significant
at any conventional level.

An in-sample Sharpe of ~3 therefore carries no information about skill here:
the identical pipeline reliably produces *better* than that on block-
bootstrapped surrogates of the same data. Read `null_control.txt` before
quoting any number in `results.csv`.

## What was run

| | |
|---|---|
| Data | Binance daily OHLCV, 2024-01-01 → 2026-04-01 |
| Universe | 21 of 30 declared assets — see `universe.json` |
| Panel | 701 dates × 12 features × 21 assets |
| Windows | 3 walk-forward (12m train / 2m val / 3m OOS, non-overlapping) |
| Search | pop 100 × 15 generations × 3 seeds = 13,006 evaluations |
| Costs | 10 bps round-trip, inside `evaluate()` |
| Null control | 19 runs × 1 seed, 20-bar blocks |

The search is smaller than a publication run would use. It was sized from a
measured throughput probe (~33 evaluations/second/core on this panel) so that
the null control — which costs 19 further full experiments — could run at all.
The null uses the *same* evolution config as the observed run, so the
comparison is valid; scaling both up is the obvious next step, and would raise
the DSR trial count and hurdle along with it.

## Reading the columns

`NaN` in a metric column means **not measured**, never "bad result". `oos_status`
says why: `ok`, `below_min_trades`, or `nan_metrics`.

Two DSR columns are reported because the answer is an interval, and here the
two conventions disagree completely:

| | Trial set | Best value |
|---|---|---|
| `dsr` | all 9,721 finite trials of 13,006 evaluations | **0.0095** |
| `dsr_bests_only` | the 9 reported winners | **1.0000** |

Sizing the multiple-testing correction to the winners rather than to the search
turns a decisive rejection into apparent certainty. `dsr` is the honest figure;
`dsr_bests_only` is retained only to make the gap visible. `attach_dsr()` warns
when the two straddle 0.95, as they do here.

Note that even `dsr` — the conservative one — cannot detect bias shared across
all trials. It is the null control, not the DSR, that produced the verdict above.

## Per-window summary

| Window | OOS period | Median OOS Sharpe | Seeds positive | Median `dsr` |
|--------|------------|-------------------|----------------|--------------|
| 0 | 2025-07-01 → 2025-09-30 | +0.472 | 3/3 | 0.005 |
| 1 | 2025-10-01 → 2025-12-31 | +0.214 | 2/3 | 0.006 |
| 2 | 2026-01-01 → 2026-03-31 | −1.570 | 0/3 | 0.001 |

All nine rows were measurable. Worth noting for anyone comparing against older
output: the OOS trade threshold is scaled to window length (38, 20, 8 for the
three windows). Window 2's rows recorded 55, 41 and 39 OOS trades, so under the
previous unscaled 50-trade filter **two of nine rows would have been written out
as an OOS Sharpe of −inf** — the worst-fitness ranking sentinel reported as
performance, which is what made the earlier committed results uninterpretable.

## Universe

21 of 30 assets. Three were lost at fetch (`HYPEUSDT`, `AEROUSDT`, `FLUIDUSDT` —
not in the local cache and unreachable from this environment) and six more fell
below `min_obs_fraction` for having listed too late in the sample (`MORPHO`,
`POL`, `WLFI`, `SYRUP`, `ONDO`, `EUL`).

The run declared this tolerance up front (`ALLOW_PARTIAL_UNIVERSE = True`,
`MIN_ASSETS = 10`) rather than discovering it afterwards; without that the
fetcher raises. **Compare `universe_fingerprint`, not asset counts, before
comparing two runs** — results computed on different compositions are not
comparable, and this is the one bias channel neither DSR nor the null control
can see, since every trial and every surrogate inherits the same universe.

`universe.json` records `universe_matches_observed: true` for the null runs:
surrogates are bootstrapped from the same OHLCV and preserve each asset's index
exactly, so `FeatureEngine` reaches the same retention decision.

## Reproducing

Daily OHLCV parquet files must be in `data_pipeline_example/cache/` as
`{SYMBOL}_1d.parquet`. `*.parquet` is gitignored, so they are not committed
here; this run used the Git LFS cache from the sibling `risk_premium_pca`
project:

```bash
git lfs pull --include="risk_premium_pca/rp_pca/data/cache/*.parquet"
cp risk_premium_pca/rp_pca/data/cache/*USDT_1d.parquet \
   vector_genetic_programming/data_pipeline_example/cache/
make start
```

Given a network path to Binance the fetcher populates the cache itself.

## Artifacts

| File | Contents |
|------|----------|
| `results.csv` | one row per (window, seed) — Sharpe, status, DSR both ways, trial counts, universe stamp |
| `null_control.txt` | verdict, p-values, and the per-run null distribution |
| `universe.json` | realized universe, what was lost at each stage, and why |
| `pareto_front.png` | Pareto scatter for the best window |
| `equity_curves.png` | IS/OOS equity curves, top 3 individuals |
| `tree_graph.png` | GP tree of the best individual |
