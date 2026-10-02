**Work in Progess**

# Vector Genetic Programming (VGP)

**Evolve trading strategies from multi-asset crypto data using genetic programming.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

---

## What is VGP

VGP is an open-source research framework that uses genetic programming (DEAP) to evolve trading strategies from multi-asset crypto OHLCV data. Populations of symbolic expression trees are optimized via NSGA-II multi-objective search, maximizing Sharpe ratio and total return while penalizing tree complexity. Fitness is evaluated using vectorbt's vectorized backtesting engine with transaction costs baked into every fitness computation — the GP cannot exploit fee-free assumptions.

Lookahead is prevented structurally: every GP signal at time `t` uses only data from `t-1` and earlier via a mandatory `fshift(1)` applied in `TreeEvaluator.execute()`. Walk-forward validation across multiple random seeds, combined with Deflated Sharpe Ratio (DSR) reporting, provides statistical evidence of out-of-sample performance.

VGP is a research tool, not a trading system. The primary output is reproducible findings — evolved strategies with measurable OOS Sharpe — plus a documented, forkable codebase.

---

## Quick Start

```bash
git clone <repo-url>
cd quantitative-research/vector_genetic_programming
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make test     # full suite
make start    # the headline experiment, end to end
```

That is the whole procedure, and it needs no network. The market data ships in
`data/` as ordinary committed parquet files — 27 assets, 956 KB — so there is no
Git LFS step and nothing to copy out of a sibling project.

`data/` holds 27 of `UNIVERSE_30`'s 30 symbols. Code that loads it must pass
`allow_partial=True` with a `min_assets` floor, or `DataLoader` will reach
Binance for the missing three and raise `FetchError`. That raise is deliberate —
a partial universe changes the experiment — and the shipped scripts all declare
it, so they run offline.

`make start` runs three walk-forward windows, three seeds and a 99-run null
control. **It takes several hours**; the null control is 99 full experiments and
is most of that time. For a quick check that the install works, `make smoke`.

`make test` collects 209 tests: 207 pass and 2 skip (the optional MLflow pair).
One is marked `slow` — a cross-process determinism canary that spawns 10
subprocesses and adds ~165s. Deselect it with `pytest -m "not slow"`, which
leaves 206 passed, 2 skipped, 1 deselected.

### Is there anything in the data to find?

```bash
python scripts/diagnose_feature_ic.py
```

Asks the data directly in minutes rather than the hours an evolution takes:
per-feature information coefficient scored against signal-free surrogates, then
a pre-specified portfolio against the same null. On the committed dataset it
reaches a firmer negative than any evolution run, and explains why. Lower
`VGP_N_NULL_IC` and `VGP_N_NULL` for a fast pass — but an empirical p cannot
resolve below `1/(1+N)`, so a reduced run cannot call anything significant.

---

## Architecture

| Module | Purpose |
|--------|---------|
| `vgp/data` | `DataLoader` fetches/caches OHLCV and raises `FetchError` rather than returning a silently smaller universe; `UniverseRecord` records what a run was actually computed on; `FeatureEngine` produces `float32 [T×F×A]` feature matrix; `WalkForwardSplitter` enforces train/val/test ordering structurally |
| `vgp/gp` | `PrimitiveSetTyped` with `Vector`/`Scalar` type tokens; 14 typed primitives (arithmetic, rolling stats, conditional); `TreeEvaluator` applies structural `fshift(1)` to prevent lookahead |
| `vgp/evolution` | `EvolutionConfig`; NSGA-II loop via `varOr` + `selNSGA2`; spawn `Pool` with JIT warmup initializer; `ParetoFront` hall-of-fame; checkpoint/resume via `dill` |
| `vgp/backtest` | `EvalConfig`; `evaluate()` returns `(sharpe, total_return, -tree_size)` fitness tuple; transaction costs baked in; `< 50 trades` receives worst-possible fitness (rankable by NSGA-II); `evaluate_with_status()` additionally reports whether that tuple is a measurement or the ranking sentinel |
| `vgp/analysis` | `WalkForwardRunner` orchestrates multi-seed evolution; `compute_dsr()` / `attach_dsr()` (Bailey & Lopez de Prado 2014); `run_null_control()` re-runs the pipeline on signal-free surrogates; `aggregate_seeds()` summarizes measured seeds only; `plot_pareto_front()`, `plot_equity_curves()`, `plot_tree_graph()` |
| `vgp/trials` | `TrialAccumulator` (streaming Welford count/spread of every evaluated individual) and `TrialSet`; numpy-only so the evolution layer can record trials without importing vectorbt |

---

## Research Notes

**Fitness objectives (NSGA-II minimizes the negatives):**

- `Sharpe ratio` — annualized IS Sharpe with fees; primary quality signal
- `total_return` — cumulative return over the training window
- `-tree_size` — node count negated; penalizes overly complex trees

**Deflated Sharpe Ratio (DSR):** After every window and seed has run, `attach_dsr()` deflates each trial's **in-sample** Sharpe — the one exposed to selection bias — for the number of trials, the cross-sectional spread of their Sharpe ratios, and the skewness and excess kurtosis of the return distribution (Bailey & Lopez de Prado, 2014). DSR > 0.95 indicates significance at the 5% level after correcting for multiple testing.

DSR is a property of the whole trial set, not of one row: the hurdle `E[SR_max]` is proportional to `sigma_SR`, the standard deviation of the Sharpe ratios across trials, so it cannot be computed inside the per-seed loop. `run_window()` leaves `dsr` as NaN and `attach_dsr()` fills it in.

**What counts as a trial.** N is the number of configurations the search effectively tried, which for a genetic program is every individual *evaluated* — `pop_size × generations × seeds × windows`, typically thousands — not the handful of per-seed winners in the results table. The difference is not cosmetic: with N = 9 the hurdle multiplier is 1.52, with N = 3600 it is 3.24. On signal-free data the winners-only convention certifies noise at DSR > 0.95 while the evaluation-based one rejects the same runs below 0.05. `vgp/trials.py` accumulates every evaluation in O(1) memory (Welford) and rides inside the DEAP logbook so it survives checkpointing.

`attach_dsr()` therefore reports **two bounds**, because the honest answer is an interval:

| Column | Trial set | Reading |
|--------|-----------|---------|
| `dsr` | every individual evaluated | **Primary.** Conservative — Proposition 3 assumes independent trials, and GP individuals are correlated by descent, so effective N is below the raw count |
| `dsr_bests_only` | the per-seed winners only | Upper bound. Flattering and near-meaningless alone |

If the two straddle 0.95, significance is an artifact of how trials were counted and the experiment has not settled the question — `attach_dsr()` logs a warning when this happens.

**What DSR cannot do, at any trial count.** It corrects for selection across trials and nothing else. Bias shared by *every* trial is invisible to it, because such bias shifts all trials together and leaves `sigma_SR` unchanged: a lookahead in a primitive, one training window reused by all seeds, a survivor-biased universe, a fee model that is wrong for all of them. No refinement of the DSR formula detects these. That is what the null control below is for.

---

## Null control

`vgp/analysis/null_control.py` runs the identical pipeline on data where the answer is known to be "nothing here", and checks that it reports nothing. This is the falsification test for the headline result, and the only check here that can catch bias shared across all trials.

Surrogates come from a circular block bootstrap of log returns using the **same block indices for every asset**. Preserved: each asset's return distribution (volatility, fat tails, skew), the cross-asset correlation structure, within-block autocorrelation, and bar geometry — open/high/low/volume ride along as ratios to their own close, so every surrogate bar is a real bar's shape. Destroyed: the ordering that makes returns predictable from signals computed on earlier bars.

The verdict is an empirical p-value in the `(1 + k) / (1 + n)` form, so it never reports exactly zero — with `n` runs the floor is `1/(1+n)`, and `NullControlResult.summary()` says so explicitly when the run count cannot resolve 0.05.

**Two statistics, not one.** The *max* compares the best row in the run against the best row of each null run: could a search this size stumble on something this good by chance. The *typical* compares the median window against the null's median window: does the average period beat chance. A strategy that works in one regime and fails elsewhere clears the max and fails the typical, and for a deployable edge the typical is the harder and more relevant bar. Reporting only the max is how a regime-dependent artifact gets mistaken for an edge, so `summary()` prints both and calls out the case where they disagree.

**The surrogate verifies itself, window by window.** `check_surrogate_fidelity()` runs before the null experiments and compares observed against surrogate inside each of several contiguous windows, on cross-asset correlation and effective-bet count. The locality is the point: a surrogate once preserved 69% full-sample PC1 share while having 0.001 mean pairwise correlation inside every training window, which inflated the null until it could not be beaten in-sample. Full-sample statistics cannot validate a surrogate.

**Block size is not a free parameter.** A block bootstrap severs dependence only at block boundaries, so with L-bar blocks roughly 1 in L transitions breaks and structure shorter than L survives. Concretely: an AR(1) series with lag-1 autocorrelation 0.82 retains 0.67 under 5-bar blocks. **The block must be shorter than the horizon of the effect being tested** — with the default 20 bars, a strategy exploiting 1–5 day momentum survives into the surrogate and the control will not flag it. `scripts/diagnose_feature_ic.py` is the worked example: its information coefficients are 1-day horizon, so its IC steps use `block_size=1`, and a test pins that value. At the 20-bar default those tests could not have failed.

The control costs `N_NULL_RUNS` full experiments, which is why the null runs use fewer seeds than the real run: the statistic is the best Sharpe the *search* finds, so the null need only represent the same procedure, not the same compute budget. Setting `N_NULL_RUNS = 0` in `scripts/run.py` skips it — and then the run must not be described as validated.

**The realized universe is recorded, and a partial fetch is an error.** `UNIVERSE_30` is an intention; two stages narrow it before any strategy is evaluated — a symbol can fail or return no rows, and `FeatureEngine` drops assets below `min_obs_fraction`. `fetch_ohlcv()` therefore raises `FetchError` by default when anything fails; a caller willing to proceed on fewer assets passes `allow_partial=True` with a `min_assets` floor, which is a declared and recorded choice rather than a log line. An empty universe always raises.

What survived both stages is written to `results/universe.json` and stamped onto every row as `n_assets` and `universe_fingerprint` (a short, order-independent id for the composition). Results computed on 26 assets are not comparable with results computed on 12, so **compare fingerprints, not asset counts.** This is the one bias channel neither DSR nor the null control can detect: every trial and every surrogate inherits whatever universe it was handed, so a universe that drifts between runs leaves no trace in the statistics.

**Reported metrics vs. ranking sentinels:** `evaluate()` returns `(-inf, -inf, -tree_size)` for an individual that fails the trade filter or produces NaN metrics, so NSGA-II can still rank it. That sentinel is not a performance measurement, and reporting code never writes it out as a Sharpe ratio: `results.csv` carries `NaN` plus an `oos_status` (`ok`, `below_min_trades`, `nan_metrics`) and the observed `oos_n_trades`. `aggregate_seeds()` excludes unmeasured seeds from its median, IQR and positive count, and reports `n_seeds_valid_oos` alongside `n_seeds` so the denominator is visible. **NaN means "not measured", never "bad result".**

The OOS trade threshold is scaled to the OOS window length (`oos_min_trades`, overridable). `min_trades = 50` is a trade-*rate* requirement written for a ~12-month training window; applied verbatim to a 3-month OOS window it is close to unreachable even for a strategy trading at exactly its in-sample frequency, which would mark every strategy invalid. The 50-trade filter itself is unchanged for evolution.

**OOS holdout:** The test split is defined before the first evolution run via `WalkForwardSplitter` and passed to `evaluate()` exactly once, for final reporting only. The evolution loop never sees OOS data. This is enforced structurally — `WalkForwardRunner` holds `test_fm` as a local variable and does not pass it to `run_evolution()`.

## Current result

Run end to end on Binance daily data (2024-01-01 → 2026-04-01, 21 assets,
**three** walk-forward windows with **disjoint 4-month OOS periods**, three
seeds, selection on a held-out validation slice) with a **99-run** null control
whose surrogate was verified faithful in all 8 fidelity windows.

**No evidence of skill, and the out-of-sample result is negative.**

| statistic | observed | null median | p |
|---|---|---|---|
| **MAX** best IS Sharpe | +3.066 | +2.770 | 0.340 |
| **MAX** best OOS Sharpe | +1.207 | +1.484 | 0.620 |
| **TYPICAL** IS Sharpe | +2.216 | +2.167 | 0.470 |
| **TYPICAL** OOS Sharpe | **−0.533** | +0.255 | **0.700** |

Per-window OOS medians: +1.207, −1.759, −0.533 — one of three positive. The
search performs *worse* than signal-free data on the statistic that matters.
`results/README.md` is the full record.

This is the first result the project has produced that **reproduces**. Every
earlier figure was one sample from a distribution: `gp.cxOnePoint` picked the
crossover type with `random.choice(list(common_types))` over type objects, which
hash by address, so ASLR reordered the list between processes and the same seed
gave different answers. The headline moved +0.974 → +0.087 on identical code and
data before the fix.

**`dsr` reaches 0.970 on this run while the null control returns p = 0.700.**
Both are correct and they answer different questions. DSR corrects for selection
*across trials* and is blind by construction to bias shared by *every* trial —
seven such biases have been found in this project and it saw none of them. Where
they disagree, believe the null control.

### Where the signal actually is

`scripts/diagnose_feature_ic.py` answers the question the GP cannot ask about
itself. Eleven of twelve features carry no timing information distinguishable
from signal-free data. The twelfth, `ret_1d` (1-day cross-sectional reversal),
does: timing IC −0.0290 against a null of +0.0058 ± 0.0104, p = 0.0040 over 250
surrogates, which clears Bonferroni.

**It does not convert into an edge.** Gross Sharpe is +0.285 against a null mean
of −0.309 — inside one standard deviation, p = 0.231. Daily rebalancing turns
the book over 1.34×, a 49% annual drag at 10 bps, taking net Sharpe to −2.18.
Real information, too small to trade at this breadth and cost.

An earlier version of this section reported that five features cleared a
Bonferroni threshold with stable split-half signs. That was measured on *raw*
cross-sectional IC tested against *zero*, and signal-free surrogates reproduce
92–124% of it: it described a static property of the assets, not predictability.
Split-half stability was not corroboration either — a static property is stable
across halves by construction.

**Honest caveat:** Positive OOS Sharpe is the goal. Results depend on data
availability, asset universe, and evolution configuration. VGP is a framework
for reproducible research — it does not guarantee profitable strategies. Three
windows over two years of one asset class is thin, even for a negative result,
and 21 crypto assets at 0.615 mean correlation give only 2.4 effective bets.

A result here is only as good as three numbers read together: the OOS Sharpe,
the conservative `dsr` against the full evaluation count, and the null control
p-value. A high Sharpe with a high `dsr` and a null p-value of 0.7 means the
pipeline found the same thing in noise.

---

## License

MIT. See [LICENSE](LICENSE).

Reference: Bailey, D. H., & Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management, 40(5).

Open to contributions — see [CONTRIBUTING.md](CONTRIBUTING.md).
