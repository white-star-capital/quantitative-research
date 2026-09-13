# results/

Run date: 2026-09-13 · universe fingerprint `af8fe4ee067f` · reproduce with `make start`

## Verdict: no evidence of skill

Six walk-forward windows with **disjoint** OOS periods, three seeds, selection on
a held-out validation slice, and a 19-run null control whose surrogate was
verified faithful in all 8 fidelity windows.

| statistic | observed | null median | null p95 | p |
|---|---|---|---|---|
| **MAX** best IS Sharpe | +3.066 | +3.046 | +4.209 | 0.500 |
| **MAX** best OOS Sharpe | +2.657 | +3.606 | +6.373 | 0.750 |
| **TYPICAL** IS Sharpe | +1.925 | +1.878 | +3.265 | 0.450 |
| **TYPICAL** OOS Sharpe | **+0.974** | +0.284 | +2.411 | **0.300** |

Per-window OOS medians: **+1.090, +0.858, −3.016, +2.657, −2.266, +1.368** —
four of six positive. All 18 rows measurable, all 18 selected on validation with
zero fallbacks.

The typical out-of-sample Sharpe is positive, and it is still not evidence of
anything. Five of nineteen signal-free surrogates reached +0.974 or better, so a
search of this size lands here three times in ten on data with no signal at all.
Nothing clears 0.05 on any of the four tests.

## Two corrections since the previous run, both of which moved the result

### The OOS windows were nested, not disjoint

`WalkForwardSplitter.split()` had no `test_end` parameter. Every test slice ran
from `test_start` to the **end of the panel**, while `results.csv` faithfully
recorded a `test_end` that was never applied. Window 0's OOS spanned ~12 months
and contained every later window's.

The only trace was `oos_min_trades`, which scales with `T_test/T_train` and fell
66, 46, 31, 20, 12, 5 across six windows that all declared the same 2-month span.
A 13:1 spread means the windows were 13:1 in length.

Everything taken across windows was affected: "six independent OOS periods" was
false, and the typical statistic was a median over six overlapping views of
largely the same stretch. Re-measured on genuinely disjoint windows, with
selection unchanged, the result got **worse**: typical OOS −1.355 (p = 0.700)
against the −0.943 (p = 0.750) reported before.

`test_end` is now required with no default — a default of "run to the end" is
what hid this for the life of the project — and three tests pin it, including an
end-to-end check that consecutive windows tile rather than nest.

### Selection used training fitness; the validation split was discarded

`run_window` computed the validation slice and threw it away (`_val_fm`,
`_val_close`), then scored `hof[0]` — the Pareto front's best by **training**
fitness. `val_months=2` was a bare embargo gap while the banner advertised
"9m train + 2m val + 2m OOS".

Not a measurement error: OOS never touched selection, so the older numbers were
honest. But it meant the framework concluded "no evidence of skill" using the
most overfitting-prone rule available. Every front member is now scored on the
held-out validation window and the best is taken.

| | select on train (`hof[0]`) | select on validation |
|---|---|---|
| TYPICAL OOS Sharpe | −1.355 | **+0.974** |
| p (typical OOS) | 0.700 | 0.300 |
| TYPICAL IS Sharpe | +2.441 | +1.925 |
| windows with positive median | 1 of 6 | 4 of 6 |

Out-of-sample rose 2.33 Sharpe while **in-sample fell** — the signature of a
real fix rather than a new leak. Selecting on training fitness was picking the
individual most fitted to the training window.

It does not change the verdict. The null median rose too (+0.052 → +0.284),
because the surrogates run the identical selection procedure. A better method
helps the null as much as the real data, which is exactly what a null control
exists to reveal.

## Earlier: a positive OOS that was a lookahead artifact

Historical, and still the largest single correction this project has made. Both
runs below predate the two fixes above, so they used nested OOS windows and
selected on training fitness — the comparison is like-for-like between
themselves, not with the headline table.

That run reported a *positive* typical OOS Sharpe of +0.448 (p = 0.200) and
per-window medians of +0.660, +0.820, +1.027, +0.235, −0.978, −0.375. The only
difference from the run that followed it is one feature.

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
out-of-sample edge**, as measured at the time. The headline result has since
returned to a positive typical OOS (+0.974) for an unrelated reason — selecting
on validation instead of training fitness — and the null control declines to
certify that one too, at p = 0.300.

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
| `dsr` (78,599 trials of 100,923 evaluations) | up to **0.965** |
| `dsr_bests_only` (18 winners) | up to 0.993 |
| null control, typical OOS | **p = 0.300** |

This is the sharpest version of the disagreement yet: **0.965 clears the
conventional 0.95 bar.** Read on its own, the Deflated Sharpe Ratio now
certifies this result. The null control, running the same pipeline on
signal-free surrogates, finds that three runs in ten do as well on data with no
signal in it.

Both numbers are arithmetically correct and they answer different questions. DSR
corrects for selection *across trials* — it asks whether the best of N tries is
better than the best of N tries should be, given the spread of those tries. It is
blind by construction to any bias shared by **every** trial, because such a bias
shifts all of them together and leaves the cross-sectional spread untouched. A
lookahead, a reused training window, a survivor-biased universe, and an
overlapping OOS period all live in that blind spot; this project has now found
four of them.

The null control is the only check here that can see them, because it rebuilds
the entire pipeline on data where the answer is known to be nothing. Where the
two disagree, the null control wins.

(The DSR figures are far larger than earlier runs' ~1e-3 partly because
de-annualization was corrected from 252 to 365 periods, matching vectorbt's
`freq="1D"` convention. That change raises DSR; it changes no verdict, because
the verdict comes from the null control.)

## What was run

| | |
|---|---|
| Data | Binance daily OHLCV, 2024-01-01 → 2026-04-01 |
| Universe | 21 of 30 declared assets — see `universe.json` |
| Panel | 701 dates × 12 features × 21 assets |
| Windows | 6 walk-forward, 9m train / 2m validation / 2m OOS, OOS periods **disjoint** |
| Selection | best Pareto-front member by **validation** Sharpe (0 fallbacks in 18 rows) |
| Search | pop 200 × 30 generations × 3 seeds = 100,923 evaluations |
| Costs | 10 bps round-trip, inside `evaluate()` |
| Null control | 19 runs × 1 seed, 20-bar blocks, one shared warm worker pool |

## Reading the columns

`NaN` in a metric column means **not measured**, never "bad result";
`oos_status` says why. All 18 rows were measurable in this run.

Two DSR columns are reported because the trial-set convention changes the
answer completely — 0.965 against all evaluations versus 0.993 against the
eighteen reported winners. `dsr` is the honest figure. Two null statistics are reported
because the max asks whether the search got lucky and the typical asks whether
the average period beats chance; here both fail, but they can disagree.

## Known limitations

**Per-seed rows are not fully independent.** In window 5, seeds 0 and 1 produced
different Pareto fronts (2 and 4 members) yet both validation-best members are
the same 3-node tree, giving identical IS and OOS Sharpe to six decimals. With
trees that small there are few distinct expressions to find, so convergence is
expected — but it means the effective number of independent observations is
lower than the 18 rows suggest, and a per-window median over 3 seeds can rest on
fewer than 3 distinct strategies.

**Disjoint OOS windows are short, and the dispersion is severe.** Two years of
data tiled into non-overlapping 2-month OOS periods gives ~61 daily bars per
window, where an annualised Sharpe carries a standard error near 2.4. Individual
seeds range −5.44 to +2.77 within a single window. That noise floor is high
enough that an edge the size of the one observed here could not be distinguished
from luck at this sample size — which is the honest reading of p = 0.300, rather
than "close to significant".

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
