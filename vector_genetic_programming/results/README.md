# results/

Run date: 2026-09-15 · universe fingerprint `af8fe4ee067f` · reproduce with `make start`

## Verdict: no evidence of skill

Three walk-forward windows with **disjoint 4-month OOS periods**, three seeds,
selection on a held-out validation slice, and a **99-run** null control whose
surrogate was verified faithful in all 8 fidelity windows.

This is the first result the project has produced that **reproduces**: it is the
first run on the deterministic crossover (see below), so re-running `make start`
returns these numbers rather than a different draw.

| statistic | observed | null median | null p95 | p |
|---|---|---|---|---|
| **MAX** best IS Sharpe | +3.066 | +2.770 | +4.141 | 0.340 |
| **MAX** best OOS Sharpe | +1.207 | +1.484 | +3.517 | 0.620 |
| **TYPICAL** IS Sharpe | +2.216 | +2.167 | +3.515 | 0.470 |
| **TYPICAL** OOS Sharpe | **−0.533** | +0.255 | +2.294 | **0.700** |

Per-window OOS medians: **+1.207, −1.759, −0.533** — one of three positive.

Two design changes were made specifically to give a real edge room to appear,
and both did their job:

* **99 null runs instead of 19.** The p-value floor was 0.05, so nothing could
  ever have been called significant; it is now 0.01. Nothing came close.
* **4-month windows instead of 2-month.** Per-window standard error fell from
  2.45 to 1.74. The typical OOS Sharpe went **down**, to −0.533 against a null
  median of +0.255 — the search performs worse than signal-free data on the
  statistic that matters.

## The features are not empty; the tradeable signal is

`scripts/diagnose_feature_ic.py` asks the data directly, in minutes rather than
the hours an evolution takes, and gives a sharper answer than the GP can.

Five features clear a Bonferroni threshold and hold their sign across both
halves of the sample with near-identical magnitudes — `vol_20d` at −0.0620 /
−0.0607, `parkinson_14` at −0.0662 / −0.0673. That is a stable cross-sectional
effect, not a regime artifact.

It does not survive contact with a portfolio:

| | result |
|---|---|
| long-only lowest-vol tercile | negative OOS — 21 assets at 0.615 mean correlation give 2.4 effective bets, so the book is a levered direction bet |
| market-neutral tercile tilt | median OOS Sharpe **+1.079** after costs — looks like an edge |
| the same strategy, 99 surrogates | **p = 0.250**, null median +0.283, p95 +1.861 |

The block bootstrap preserves each asset's volatility level and the cross-asset
correlation structure — that is what makes it a fair null — so the
low-volatility assets in a surrogate are **still** the low-volatility assets. A
long-low/short-high book inherits that structure's return asymmetry with no
predictive timing involved. The apparent edge is the structure, not information.

Which also explains the GP result. It is not failing to search hard enough;
there is nothing to find beyond what the null reproduces.

## An observation on validation selection, not yet a finding

Across the nine rows of this run, validation Sharpe is **anti**-correlated with
OOS Sharpe (pooled −0.608), and the window-level pattern is stark:

| window | mean validation Sharpe | mean OOS Sharpe |
|---|---|---|
| W0 | +0.11 | **+1.29** |
| W1 | +2.92 | −0.72 |
| W2 | +2.94 | −0.74 |

If real, it would mean selecting on validation is not merely neutral here but
actively harmful, and the natural reading is regime alternation: the validation
window sits immediately before the OOS window, and a period that looked good is
followed by one that does not.

**Three windows is not evidence.** It is recorded because it is cheap to test
on the next dataset and expensive to discover later.

## Two corrections since the previous run## Two corrections since the previous run, both of which moved the result

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
| `dsr` (38,774 trials of 50,455 evaluations) | up to **0.970** |
| `dsr_bests_only` (9 winners) | up to 0.993 |
| null control, typical OOS | **p = 0.700** |

This is the sharpest version of the disagreement yet: **0.970 clears the
conventional 0.95 bar**, on a run whose null control returns p = 0.700. Read on its own, the Deflated Sharpe Ratio now
certifies this result. The null control, running the same pipeline on
signal-free surrogates, finds that seven runs in ten do as well on data with no
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
| Windows | 3 walk-forward, 9m train / 2m validation / **4m OOS**, OOS periods **disjoint** |
| Selection | best Pareto-front member by **validation** Sharpe (0 fallbacks in 9 rows) |
| Search | pop 200 × 30 generations × 3 seeds = 50,455 evaluations |
| Costs | 10 bps round-trip, inside `evaluate()` |
| Null control | **99 runs** × 1 seed, 20-bar blocks, one shared warm worker pool |

## Reading the columns

`NaN` in a metric column means **not measured**, never "bad result";
`oos_status` says why. All 18 rows were measurable in this run.

Two DSR columns are reported because the trial-set convention changes the
answer completely — 0.970 against all evaluations versus 0.993 against the
nine reported winners. `dsr` is the honest figure. Two null statistics are reported
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
data tiled into non-overlapping 4-month OOS periods gives ~121 daily bars per
window, where an annualised Sharpe carries a standard error near 2.4. Individual
seeds range −5.44 to +2.77 within a single window. That noise floor is high
enough that an edge the size of the one observed here could not be distinguished
from luck at this sample size — which is the honest reading of p = 0.300, rather
than "close to significant".

- **99 null runs floor the p-value at 0.01.** These p-values (0.340–0.700) are
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

```bash
git clone <repo> && cd vector_genetic_programming
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
make start
```

That is the whole procedure. The market data ships in `data/` as ordinary
committed files — 27 assets, 956 KB — so there is no Git LFS step, nothing to
copy out of a sibling project, and no setup command before `make start`.

Previously the data lived in `data_pipeline_example/cache/`, which the
`*.parquet` gitignore rule excluded, so a fresh clone had none of it. Running
anything first required:

```bash
git lfs pull --include="risk_premium_pca/rp_pca/data/cache/*.parquet"
cp risk_premium_pca/rp_pca/data/cache/*USDT_1d.parquet \
   vector_genetic_programming/data_pipeline_example/cache/
```

Two commands documented in one paragraph of one file. `tests/test_data_pipeline.py`
now fails if `data/` goes missing, gets re-ignored, or is committed as LFS
pointers, so a clean clone stays runnable.

To point at your own data, drop `{SYMBOL}_1d.parquet` files into `data/` or
change `CACHE_DIR` at the top of `scripts/run.py`. Paths there resolve against
the project directory, not the working directory, so the script runs correctly
from anywhere.

`python scripts/diagnose_null_gap.py` re-verifies that the surrogate is a fair
opponent after any change to the bootstrap.

## Artifacts

| File | Contents |
|------|----------|
| `results.csv` | one row per (window, seed) — Sharpe, status, DSR both ways, trial counts, universe stamp |
| `null_control.txt` | both statistics, p-values, fidelity verdict, per-run null distributions |
| `universe.json` | realized universe, what was lost at each stage, and why |
| `pareto_front.png` · `equity_curves.png` · `tree_graph.png` | best window / top 3 individuals / best tree |
