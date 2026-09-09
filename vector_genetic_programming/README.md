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
cd vector-genetic-programming
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run full test suite
python -m pytest tests/ -v

# Run a minimal evolution experiment (requires parquet data in data/cache/)
python -m vgp.analysis.runner
```

The test suite covers all five sub-modules (smoke, data pipeline, GP primitives, backtest evaluation, evolution engine, walk-forward validation, visualizations). All 57 tests should pass.

---

## Architecture

| Module | Purpose |
|--------|---------|
| `vgp/data` | `DataLoader` reads parquet files into `dict[str, pd.DataFrame]`; `FeatureEngine` produces `float32 [T×F×A]` feature matrix; `WalkForwardSplitter` enforces train/val/test ordering structurally |
| `vgp/gp` | `PrimitiveSetTyped` with `Vector`/`Scalar` type tokens; 14 typed primitives (arithmetic, rolling stats, conditional); `TreeEvaluator` applies structural `fshift(1)` to prevent lookahead |
| `vgp/evolution` | `EvolutionConfig`; NSGA-II loop via `varOr` + `selNSGA2`; spawn `Pool` with JIT warmup initializer; `ParetoFront` hall-of-fame; checkpoint/resume via `dill` |
| `vgp/backtest` | `EvalConfig`; `evaluate()` returns `(sharpe, total_return, -tree_size)` fitness tuple; transaction costs baked in; `< 50 trades` receives worst-possible fitness (rankable by NSGA-II); `evaluate_with_status()` additionally reports whether that tuple is a measurement or the ranking sentinel |
| `vgp/analysis` | `WalkForwardRunner` orchestrates multi-seed evolution; `compute_dsr()` / `attach_dsr()` (Bailey & Lopez de Prado 2014); `aggregate_seeds()` summarizes measured seeds only; `plot_pareto_front()`, `plot_equity_curves()`, `plot_tree_graph()` |

---

## Research Notes

**Fitness objectives (NSGA-II minimizes the negatives):**

- `Sharpe ratio` — annualized IS Sharpe with fees; primary quality signal
- `total_return` — cumulative return over the training window
- `-tree_size` — node count negated; penalizes overly complex trees

**Deflated Sharpe Ratio (DSR):** After every window and seed has run, `attach_dsr()` deflates each trial's **in-sample** Sharpe — the one exposed to selection bias — for the number of trials, the cross-sectional spread of their Sharpe ratios, and the skewness and excess kurtosis of the return distribution (Bailey & Lopez de Prado, 2014). DSR > 0.95 indicates significance at the 5% level after correcting for multiple testing.

DSR is a property of the whole trial set, not of one row: the hurdle `E[SR_max]` is proportional to `sigma_SR`, the standard deviation of the Sharpe ratios across trials, so it cannot be computed inside the per-seed loop. `run_window()` leaves `dsr` as NaN and `attach_dsr()` fills it in.

Two things DSR does not do. It does not correct for bias shared by every trial — a common lookahead, one training window reused across seeds, a survivor-biased universe — because such bias moves all trials together and leaves `sigma_SR` unchanged. And its correction is only as large as the trial count it is given: a handful of tightly-clustered trials produces a small hurdle and therefore a high DSR. Read `dsr` alongside `dsr_n_trials` and `dsr_trial_sr_std`, both recorded in `results.csv`; a high DSR over few, similar trials is weak evidence, not strong evidence.

**Reported metrics vs. ranking sentinels:** `evaluate()` returns `(-inf, -inf, -tree_size)` for an individual that fails the trade filter or produces NaN metrics, so NSGA-II can still rank it. That sentinel is not a performance measurement, and reporting code never writes it out as a Sharpe ratio: `results.csv` carries `NaN` plus an `oos_status` (`ok`, `below_min_trades`, `nan_metrics`) and the observed `oos_n_trades`. `aggregate_seeds()` excludes unmeasured seeds from its median, IQR and positive count, and reports `n_seeds_valid_oos` alongside `n_seeds` so the denominator is visible. **NaN means "not measured", never "bad result".**

The OOS trade threshold is scaled to the OOS window length (`oos_min_trades`, overridable). `min_trades = 50` is a trade-*rate* requirement written for a ~12-month training window; applied verbatim to a 3-month OOS window it is close to unreachable even for a strategy trading at exactly its in-sample frequency, which would mark every strategy invalid. The 50-trade filter itself is unchanged for evolution.

**OOS holdout:** The test split is defined before the first evolution run via `WalkForwardSplitter` and passed to `evaluate()` exactly once, for final reporting only. The evolution loop never sees OOS data. This is enforced structurally — `WalkForwardRunner` holds `test_fm` as a local variable and does not pass it to `run_evolution()`.

**Honest caveat:** Positive OOS Sharpe is the goal. Results depend on data availability, asset universe, and evolution configuration. VGP is a framework for reproducible research — it does not guarantee profitable strategies.

---

## License

MIT. See [LICENSE](LICENSE).

Reference: Bailey, D. H., & Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management, 40(5).

Open to contributions — see [CONTRIBUTING.md](CONTRIBUTING.md).
