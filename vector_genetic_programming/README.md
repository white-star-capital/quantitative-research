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
| `vgp/backtest` | `EvalConfig`; `evaluate()` returns `(sharpe, total_return, -tree_size)` fitness tuple; transaction costs baked in; `< 50 trades` receives worst-possible fitness (rankable by NSGA-II) |
| `vgp/analysis` | `WalkForwardRunner` orchestrates multi-seed evolution; `compute_dsr()` (Bailey & Lopez de Prado 2014); `plot_pareto_front()`, `plot_equity_curves()`, `plot_tree_graph()` |

---

## Research Notes

**Fitness objectives (NSGA-II minimizes the negatives):**

- `Sharpe ratio` — annualized IS Sharpe with fees; primary quality signal
- `total_return` — cumulative return over the training window
- `-tree_size` — node count negated; penalizes overly complex trees

**Deflated Sharpe Ratio (DSR):** After evolution, `compute_dsr()` adjusts the observed OOS Sharpe for the number of independent strategy trials tested, skewness, and excess kurtosis of the return distribution. DSR > 0.95 means the strategy is statistically significant at the 5% level after correcting for multiple testing (Bailey & Lopez de Prado, 2014). A raw OOS Sharpe without DSR correction is not reliable evidence of skill when many strategies were evaluated.

**OOS holdout:** The test split is defined before the first evolution run via `WalkForwardSplitter` and passed to `evaluate()` exactly once, for final reporting only. The evolution loop never sees OOS data. This is enforced structurally — `WalkForwardRunner` holds `test_fm` as a local variable and does not pass it to `run_evolution()`.

**Honest caveat:** Positive OOS Sharpe is the goal. Results depend on data availability, asset universe, and evolution configuration. VGP is a framework for reproducible research — it does not guarantee profitable strategies.

---

## License

MIT. See [LICENSE](LICENSE).

Reference: Bailey, D. H., & Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management, 40(5).

Open to contributions — see [CONTRIBUTING.md](CONTRIBUTING.md).
