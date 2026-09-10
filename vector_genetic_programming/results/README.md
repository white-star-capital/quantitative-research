# results/

Run date: 2026-09-10 · universe fingerprint `af8fe4ee067f` · reproduce with `make start`

## Verdict: no evidence of skill

The search does not beat data with no signal, in-sample or out-of-sample.

| statistic | observed | null median | null p95 | p |
|---|---|---|---|---|
| best IS Sharpe | +3.705 | +3.412 | +4.331 | **0.300** |
| best OOS Sharpe | +1.862 | +0.900 | +2.623 | **0.250** |

Out-of-sample the observed best sits between the null's median and its 95th
percentile — 4 of 19 signal-free runs did better. That is an unremarkable draw
from the null, not a finding. Read `null_control.txt` before quoting any number
in `results.csv`.

## These numbers supersede two earlier runs, which were wrong

Both previous null controls were computed against a broken surrogate. In
`block_bootstrap_ohlcv`, cross-asset co-movement was shared only over the
intersection of all 27 fetched assets; EUL lists 2025-10-13, so that
intersection was the final 171 days — after every training window — and every
training-window bar fell into a per-asset fallback that drew each asset
independently. Measured in the real training window: mean pairwise correlation
0.615 against **0.001** in the surrogate, 2.40 effective bets against **19.96**.

A 1/N cross-sectional book with twenty independent bets instead of two reaches
a far higher in-sample Sharpe, so the null was inflated in-sample and distorted
out-of-sample. The correction moves both p-values, in opposite directions:

| | broken null | corrected null |
|---|---|---|
| IS null median | +5.387 | +3.412 |
| IS p | 1.000 (every surrogate beat the real run) | 0.300 |
| OOS null median | −0.043 | +0.900 |
| OOS null p95 | +0.849 | +2.623 |
| OOS p | 0.050 | 0.250 |

**The encouraging OOS result was an artifact.** With correlated assets the null
runs reach far higher OOS Sharpe (median +0.900, p95 +2.623), so a corrected
null is a materially harder opponent and the observed +1.862 no longer stands
out. Conversely the in-sample pathology — every surrogate beating the real run
— is gone, because the surrogate is now the same problem the GP actually faces.

The surrogates behind these numbers were verified faithful: 0 fidelity breaches
across 8 windows, max divergence 28.5% against a 50% tolerance. The pipeline now
runs that check itself on every null control (`check_surrogate_fidelity`) and
refuses to present a clean p-value over a breached one.

## What was run

| | |
|---|---|
| Data | Binance daily OHLCV, 2024-01-01 → 2026-04-01 |
| Universe | 21 of 30 declared assets — see `universe.json` |
| Panel | 701 dates × 12 features × 21 assets |
| Windows | 3 walk-forward (12m train / 2m val / 3m OOS, non-overlapping) |
| Search | pop 250 × 40 generations × 3 seeds = 83,239 evaluations |
| Costs | 10 bps round-trip, inside `evaluate()` |
| Null control | 19 runs × 1 seed, 20-bar blocks, one shared warm worker pool |

## Reading the columns

`NaN` in a metric column means **not measured**, never "bad result". `oos_status`
says why: `ok`, `below_min_trades`, or `nan_metrics`. All nine rows were
measurable here.

Two DSR columns are reported because the answer is an interval, and the two
conventions disagree completely:

| | Trial set | Best value |
|---|---|---|
| `dsr` | all 67,756 finite trials of 83,239 evaluations | **0.0002** |
| `dsr_bests_only` | the 9 reported winners | **1.0000** |

Sizing the multiple-testing correction to the winners rather than to the search
turns a decisive rejection into apparent certainty. `dsr` is the honest figure;
`dsr_bests_only` exists only to make the gap visible. Note that even `dsr`
cannot detect bias shared across all trials — it was the null control, not the
DSR, that produced the verdict above, and it was a bug in the null control that
produced the two wrong verdicts before it.

## Per-window summary

| Window | OOS period | Median OOS Sharpe | Seeds positive | Median `dsr` |
|--------|------------|-------------------|----------------|--------------|
| 0 | 2025-07-01 → 2025-09-30 | +1.139 | 3/3 | 0.000 |
| 1 | 2025-10-01 → 2025-12-31 | −0.608 | 1/3 | 0.000 |
| 2 | 2026-01-01 → 2026-03-31 | −1.007 | 0/3 | 0.000 |

Only window 0 is positive, and the trend across windows is downward. The OOS
trade threshold is scaled to window length (38, 20, 8); under the former
unscaled 50-trade filter some rows would have been written out as an OOS Sharpe
of −inf rather than measured.

## Universe

21 of 30 assets. Three lost at fetch (`HYPEUSDT`, `AEROUSDT`, `FLUIDUSDT` — not
in the local cache and unreachable from this environment) and six below
`min_obs_fraction` for listing too late (`MORPHO`, `POL`, `WLFI`, `SYRUP`,
`ONDO`, `EUL`).

The run declared this tolerance up front (`ALLOW_PARTIAL_UNIVERSE = True`,
`MIN_ASSETS = 10`); without that the fetcher raises. **Compare
`universe_fingerprint`, not asset counts, before comparing two runs.**

Note that `EUL` — dropped from the panel by `min_obs_fraction` — is the asset
whose late listing collapsed the bootstrap's intersection. An asset that never
reached the GP silently broke the null control that judged it.

## Known limitations

- **19 null runs floor the p-value at 0.05.** These p-values (0.300, 0.250) are
  well clear of the floor, so the conclusion does not depend on it, but a
  positive result would.
- **Window-local correlation is matched in distribution, not exactly.** The
  bootstrap draws source dates from the whole eligible region, so an unusually
  high-correlation window gets a slightly easier surrogate (e.g. 0.736 → 0.654).
  Same directional bias as the fixed bug, roughly two orders of magnitude
  smaller. Drawing only within-window would remove it at the cost of block
  diversity.
- **The search is smaller than a publication run** and the null control costs 19
  further experiments; scaling both up raises the DSR hurdle with the trial count.
- **Three windows over two years of one asset class** is a thin basis for any
  claim either way.

## Reproducing

Daily OHLCV parquet files must be in `data_pipeline_example/cache/` as
`{SYMBOL}_1d.parquet`. `*.parquet` is gitignored, so they are not committed;
this run used the Git LFS cache from the sibling `risk_premium_pca` project:

```bash
git lfs pull --include="risk_premium_pca/rp_pca/data/cache/*.parquet"
cp risk_premium_pca/rp_pca/data/cache/*USDT_1d.parquet \
   vector_genetic_programming/data_pipeline_example/cache/
make start
```

Given a network path to Binance the fetcher populates the cache itself.
`python scripts/diagnose_null_gap.py` re-verifies that the surrogate is a fair
opponent after any change to the bootstrap.

## Artifacts

| File | Contents |
|------|----------|
| `results.csv` | one row per (window, seed) — Sharpe, status, DSR both ways, trial counts, universe stamp |
| `null_control.txt` | verdict, p-values, and the per-run null distribution |
| `universe.json` | realized universe, what was lost at each stage, and why |
| `pareto_front.png` | Pareto scatter for the best window |
| `equity_curves.png` | IS/OOS equity curves, top 3 individuals |
| `tree_graph.png` | GP tree of the best individual |
