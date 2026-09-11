# results/

Run date: 2026-09-11 · universe fingerprint `af8fe4ee067f` · reproduce with `make start`

## Verdict: no evidence of skill, and the out-of-sample result is negative

Six walk-forward windows, three seeds, a 19-run null control whose surrogate was
verified faithful in all 8 fidelity windows.

| statistic | observed | null median | null p95 | p |
|---|---|---|---|---|
| **MAX** best IS Sharpe | +3.308 | +3.771 | +4.637 | 0.600 |
| **MAX** best OOS Sharpe | +1.362 | +0.866 | +3.411 | 0.400 |
| **TYPICAL** IS Sharpe | +2.421 | +2.631 | +3.638 | 0.700 |
| **TYPICAL** OOS Sharpe | **−0.943** | −0.412 | +2.050 | **0.750** |

The typical window loses money out-of-sample, and loses *more* than a typical
signal-free surrogate does. On the max statistic the search does not even reach
the null's median in-sample. Nothing here is close to significant on any of the
four tests.

Per-window OOS medians: **−0.736, −0.319, −1.471, −2.867, −1.149, +0.066**.
Only the last window is positive, and barely. All 18 rows measurable.

## The previous run's positive OOS was a lookahead artifact

The run immediately before this one reported a *positive* typical OOS Sharpe of
+0.448 (p = 0.200) and per-window medians of +0.660, +0.820, +1.027, +0.235,
−0.978, −0.375. The only difference is one feature.

`obv_signal` was z-scored with whole-series mean and standard deviation, and
`FeatureEngine.fit_transform()` runs once on the full panel before any
walk-forward split — so every training window was normalised using statistics
that included every OOS period. It is now a trailing 20-bar z-score.

| | with the leak | causal |
|---|---|---|
| typical OOS Sharpe | +0.448 | −0.943 |
| p (typical OOS) | 0.200 | 0.750 |
| windows with positive median | 4 of 6 | 1 of 6 |

**That single contaminated feature accounted for essentially all of the apparent
out-of-sample edge.**

### The severity of the leak was badly misjudged, and the method was the problem

Before rerunning, the leak was assessed as "mild" on three grounds: its
correlation with a pure time index was 0.522 against 0.532 for the
uncontaminated `log_close`, so it was not acting as a calendar proxy; train and
OOS value ranges overlapped rather than being disjoint, so an in-sample
threshold was not trivially true or false out-of-sample; and it was an affine
transform with panel-wide constants, injecting no bar-level future information.

Every one of those observations was correct, and together they were worthless
as a severity estimate. They measured properties of the *feature* and inferred
an effect on *fitted strategies*, and a GP can exploit a globally normalised
input in ways that leave no trace in either statistic. The swing was 1.39 in
Sharpe and a sign change.

The one thing that did establish severity was rerunning the experiment. Proxy
diagnostics that feel quantitative are not a substitute, and the direction of a
bias ("it favours finding skill, so a negative result is conservative") says
nothing about its size.

## DSR and the null control disagree — believe the null control

| | value |
|---|---|
| `dsr` (78,210 trials of 100,852 evaluations) | up to **0.861** |
| `dsr_bests_only` (9 winners) | up to 0.994 |
| null control, max IS | **p = 0.600** |

A DSR of 0.86 reads as "nearly significant". The null control, running the same
pipeline on signal-free data, finds it performs *better* on noise. Both numbers
are arithmetically correct; they answer different questions. DSR corrects for
selection across trials and is blind to bias shared by every trial, which is
exactly what a null control is for. Where they disagree, the null control wins.

(The DSR values are much larger than earlier runs' ~1e-3 partly because
de-annualization was corrected from 252 to 365 periods, matching vectorbt's
`freq="1D"` convention. That change raises DSR; it does not change any verdict,
because the verdict comes from the null control.)

## What was run

| | |
|---|---|
| Data | Binance daily OHLCV, 2024-01-01 → 2026-04-01 |
| Universe | 21 of 30 declared assets — see `universe.json` |
| Panel | 701 dates × 12 features × 21 assets |
| Windows | 6 walk-forward (9m train / 2m val / 2m OOS, non-overlapping) |
| Search | pop 200 × 30 generations × 3 seeds = 100,852 evaluations |
| Costs | 10 bps round-trip, inside `evaluate()` |
| Null control | 19 runs × 1 seed, 20-bar blocks, one shared warm worker pool |

## Reading the columns

`NaN` in a metric column means **not measured**, never "bad result";
`oos_status` says why. All 18 rows were measurable in this run.

Two DSR columns are reported because the trial-set convention changes the
answer completely — 0.861 against all evaluations versus 0.994 against the nine
reported winners. `dsr` is the honest figure. Two null statistics are reported
because the max asks whether the search got lucky and the typical asks whether
the average period beats chance; here both fail, but they can disagree.

## Known limitations

- **19 null runs floor the p-value at 0.05.** These p-values (0.400–0.750) are
  nowhere near the floor, so the conclusion does not depend on it.
- **Window-local correlation is matched in distribution, not exactly.** A
  bootstrap draws source dates from the whole eligible region, so an unusually
  high-correlation window gets a slightly easier surrogate. Verified within
  tolerance in all 8 fidelity windows.
- **Six windows over two years of one asset class** is a thin basis for any
  claim either way — though a negative result needs less support than a
  positive one would.
- **Serial and pooled evaluation are not verified bit-identical.** Results come
  from `n_jobs=3`; EXP-03 is pinned for the serial path and through a stub pool,
  not across a real spawn pool. See the open item in the session notes.

## Reproducing

Daily OHLCV parquet files must be in `data_pipeline_example/cache/` as
`{SYMBOL}_1d.parquet`. `*.parquet` is gitignored; this run used the Git LFS
cache from the sibling `risk_premium_pca` project:

```bash
git lfs pull --include="risk_premium_pca/rp_pca/data/cache/*.parquet"
cp risk_premium_pca/rp_pca/data/cache/*USDT_1d.parquet \
   vector_genetic_programming/data_pipeline_example/cache/
make start
```

`python scripts/diagnose_null_gap.py` re-verifies that the surrogate is a fair
opponent after any change to the bootstrap.

## Artifacts

| File | Contents |
|------|----------|
| `results.csv` | one row per (window, seed) — Sharpe, status, DSR both ways, trial counts, universe stamp |
| `null_control.txt` | both statistics, p-values, fidelity verdict, per-run null distributions |
| `universe.json` | realized universe, what was lost at each stage, and why |
| `pareto_front.png` · `equity_curves.png` · `tree_graph.png` | best window / top 3 individuals / best tree |
