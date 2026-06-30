# Features Research: VGP for Crypto Trading

**Domain:** Genetic Programming framework for financial strategy evolution
**Researched:** 2026-06-03
**Overall confidence:** HIGH (core GP-finance literature confirmed; crypto on-chain GP specific = MEDIUM)

---

## Table Stakes (Must Have)

Missing any of these makes the system not credible as research.

---

### Primitive Set

#### Terminals (inputs to GP trees)

| Terminal Class | Examples | Notes |
|---|---|---|
| Price/return vectors | `close`, `open`, `high`, `low`, `returns` | Rolling window; shape `(n_bars,)` per asset |
| Volume vectors | `volume`, `quote_volume` | Same shape; normalize or not — document the choice |
| Derived market vectors | `log_returns`, `rolling_volatility(w)` | Pre-computed before evolution; not evolved |
| On-chain vectors | `mvrv`, `nvt`, `exchange_inflow`, `exchange_outflow`, `sopr` | Core differentiator for crypto; include even if sparse |
| Ephemeral constants | Random float in `[-1, 1]` or `[1, 50]` for window sizes | DEAP `addEphemeralConstant`; must be typed |

Recommendation: Keep terminal count small at first — 8 to 12 vectors. Expand only after baseline is validated. More terminals = larger search space = more generations needed to find non-noise.

#### Vector Operators (operate on rolling-window vectors → produce vectors)

| Operator | Signature | Purpose |
|---|---|---|
| `add(v1, v2)` | `Vec, Vec → Vec` | Element-wise arithmetic |
| `sub(v1, v2)` | `Vec, Vec → Vec` | |
| `mul(v1, v2)` | `Vec, Vec → Vec` | |
| `div_safe(v1, v2)` | `Vec, Vec → Vec` | Protected: return 1.0 on division by zero |
| `neg(v)` | `Vec → Vec` | |
| `abs_val(v)` | `Vec → Vec` | |
| `rolling_mean(v, w)` | `Vec, Int → Vec` | Core for trend signals |
| `rolling_std(v, w)` | `Vec, Int → Vec` | Core for volatility signals |
| `rolling_max(v, w)` | `Vec, Int → Vec` | |
| `rolling_min(v, w)` | `Vec, Int → Vec` | |
| `zscore(v, w)` | `Vec, Int → Vec` | Normalize within window; reduces scale sensitivity |
| `crossover_above(v1, v2)` | `Vec, Vec → BoolVec` | Returns 1 where v1 crosses above v2 |
| `threshold_gt(v, c)` | `Vec, Float → BoolVec` | v > constant |
| `threshold_lt(v, c)` | `Vec, Float → BoolVec` | |

#### Signal Aggregation Operators (BoolVec → Signal)

These are the final tree output layer — they collapse a boolean vector into a tradeable signal.

| Operator | Signature | Output |
|---|---|---|
| `long_signal(bool_vec)` | `BoolVec → {-1, 0, 1}` | 1 where True |
| `long_short_signal(bool_vec)` | `BoolVec → {-1, 0, 1}` | +1 where True, -1 where False |

The tree **root must return a signal type** — this is the STGP type constraint that prevents nonsensical trees. The VGP paper (arxiv 2504.05418) empirically confirms that strongly-typed VGP is always among the best performers; untyped GP is always among the worst.

#### What to omit from primitives (anti-bloat)

- Trigonometric functions (sin, cos, tan): included in vanilla GP but add noise and computational cost without financial meaning. Omit unless there is a specific hypothesis they model.
- LOG, SQRT without protection: cause NaN propagation that silently corrupts fitness evaluation.
- Momentum indicators as primitives (RSI, MACD, Bollinger): better as terminals (pre-computed features), not as operators the GP recombines. Keeps tree semantics interpretable.

---

### Fitness and Evaluation

#### Primary fitness objectives (NSGA-II Pareto front)

| Objective | Direction | Justification |
|---|---|---|
| Annualized Sharpe ratio (OOS) | Maximize | Primary credibility metric; risk-adjusted |
| Maximum drawdown | Minimize | Prevents high-Sharpe strategies that blow up |
| Tree complexity (node count) | Minimize | Parsimony pressure baked into multi-objective search |

Three objectives is the right number for a solo research project. Four or more objectives make Pareto front visualization intractable and dilute selection pressure.

#### Secondary metrics (reported, not optimized)

Compute these after evolution on the OOS split — they are diagnostic, not fitness signals. Optimizing for them directly invites overfitting.

| Metric | Why Report |
|---|---|
| Annualized return | Raw performance context |
| Sortino ratio | Distinguishes downside from upside volatility |
| Calmar ratio | Return / max drawdown; regime-adjusted perspective |
| Win rate | Sanity check; strategies with > 80% win rate are almost always overfit |
| Number of trades | Below 50 trades, all statistics are statistically unreliable |
| Average holding period | Classifies strategy (scalping vs swing vs positional) |
| Profit factor | Gross profit / gross loss |
| Deflated Sharpe ratio (DSR) | Corrects for selection bias across multiple trials; required for credible multi-run reporting |

#### Fitness evaluation mechanics

- Evaluate on **training split only** during evolution. The OOS split is touched exactly once — after the final evolved population is selected.
- Minimum trade filter: reject individuals with fewer than 50 trades as statistically uninformative (GT-Score paper, 2026).
- vectorbt computes all metrics in batch across the population. The fitness function wraps `portfolio.stats()` and extracts the objective vector. This is the primary performance bottleneck — keep the evaluation call vectorized, never per-individual Python loops.

#### GT-Score as a fitness option

The GT-Score formula `(μ · ln(z) · r²) / σd` integrates performance, statistical significance, consistency, and downside risk multiplicatively. In walk-forward validation it improved generalization ratio by 98% over raw Sortino. Consider it as an alternative single-objective fitness for ablation experiments. It is not a replacement for NSGA-II — it is a single-objective alternative worth comparing.

---

### Anti-Overfitting

This is the most important section. Every mechanism here is table stakes. Omitting any one is enough to invalidate research claims.

#### Walk-forward validation structure

```
Full dataset (chronological)
├── Training window       60–70% of data
├── Validation window     15–20% (used during evolution for early stopping / hyperparameter choice)
└── OOS holdout           15–20% — touched ONCE after all experiments are done
```

Rules:
- OOS data is invisible during evolution. No hyperparameter selection on OOS. No iterating after seeing OOS results.
- Use rolling or expanding train windows for multi-period WF runs. A single in-sample/OOS split is acceptable for v1 but must be stated explicitly in reporting.
- Time-series splits must respect temporal order — shuffling is forbidden. vectorbt has purged k-fold support; use it for multi-split experiments.

#### Parsimony pressure

Use NSGA-II's third objective (tree node count) as the primary parsimony mechanism. This is cleaner and more principled than a scalar penalty coefficient.

Supplementary hard limits (enforce these as constraints, not objectives):
- Maximum tree depth: 10 to 13 nodes (VGP paper used 13; Kostadinov recommends starting at 7)
- Maximum tree size: 60 to 90 nodes (VGP paper used 90)

HARM-GP (DEAP built-in `gp.harm`) is an alternative bloat controller that dynamically shapes the size distribution. Use it if NSGA-II alone produces bloated individuals despite the complexity objective.

Double tournament selection (`selDoubleTournament`) is another option — runs a fitness tournament followed by a size tournament. Less principled than NSGA-II for multi-objective but simpler to implement.

#### Multiple independent runs

Run at least 10 independent runs per experiment with different seeds. Report distributions (median Sharpe, IQR), not the best single run. The VGP paper used 10 runs per experiment; this is the community standard.

Single-run results are anecdotes, not findings.

#### Deflated Sharpe Ratio and PBO

When reporting results across multiple runs or primitive set configurations:
- Compute the Deflated Sharpe Ratio (DSR) to correct for selection bias. The DSR penalizes the winner's Sharpe based on how many strategies were tried.
- Compute the Probability of Backtest Overfitting (PBO, Bailey et al.) if running combinatorial path sampling. `pypbo` is the reference Python implementation.

Both must be reported when the number of evaluated strategies exceeds ~20. At scale (hundreds of evolved trees), DSR is mandatory to make OOS Sharpe claims credible.

#### Randomized training windows per generation

The VGP paper used a randomized training window start (buffer of 100 days, random first day per generation) to prevent the GP from memorizing a specific period. This is a low-cost, high-value anti-overfitting technique. Implement this from the start.

#### Regime awareness

A strategy that works only in one market regime (e.g., 2020–2021 bull run) will show strong in-sample Sharpe and collapse OOS if the OOS period covers different conditions. Mitigations:
- Include at least one bear market and one sideways market in the training window.
- Report regime-split statistics (bull vs bear vs sideways Sharpe) as secondary outputs.
- Crypto datasets spanning 2018–2024 cover at least two full cycles. Use them.

---

### Reproducibility

These are table stakes for a research-grade solo project. Without them, no result is recoverable or trustworthy.

| Requirement | Implementation |
|---|---|
| Global random seeds | `random.seed(seed)` + `np.random.seed(seed)` before every run; pass seed through all DEAP operators |
| Seed-per-run logging | Every experiment file name and log entry includes the seed used |
| Config-driven experiments | All hyperparameters in a single YAML/TOML config file; no magic numbers in code |
| Full run reproducibility | Same config + same seed = bit-identical results. Verify this explicitly |
| Experiment logging | Log: seed, config hash, primitive set description, objective values per generation, final OOS stats |
| Checkpointing | Save population state every N generations. Required for long runs; allows resumption and post-hoc analysis |
| Dependency pinning | `requirements.txt` or `pyproject.toml` with exact versions. Pin DEAP==1.4.4 and vectorbt==1.0.0 hard — both had breaking changes |
| Data versioning | Parquet files checksummed (MD5 or SHA-256). Log checksum in each experiment. Changing data with same config must not silently produce different results |

DEAP does not have a global seed parameter. The workaround (confirmed in sklearn-genetic-opt docs) is to seed `random` and `numpy.random` before initializing the toolbox. Document this explicitly in a `set_random_state(seed)` utility.

---

## Differentiators (Research-Grade)

Features that separate serious VGP research from toy GP implementations.

---

### Strongly-Typed GP (STGP)

Use DEAP's `PrimitiveSetTyped` with at minimum three distinct types: `VecFloat` (rolling-window float array), `VecBool` (boolean signal vector), and `Int` (window size parameter). The type system prevents trees where, e.g., a window-size integer is added element-wise to a price vector.

The VGP paper (arxiv 2504.05418) demonstrated empirically that STGP is always among the best performers across all three test assets over seven years of data. Standard untyped GP is always among the worst. This is the single highest-confidence finding in recent VGP-for-finance literature.

DEAP known gotcha: `bool` and `int` can collide in STGP. Use distinct Python classes (e.g., `class WindowSize: pass`) rather than built-in types to avoid silent type confusion.

### Multi-Asset Evolution

Evolve strategies on multiple assets simultaneously. Options:
1. **Shared primitive trees per asset class**: Same tree evaluated on BTC, ETH, and altcoins; fitness is the portfolio-level Sharpe across assets. Produces portable strategies.
2. **Per-asset trees with portfolio aggregation**: Separate GP runs per asset; combine signals in a second-stage portfolio layer.

Option 1 is more defensible as research because it directly tests generalization across assets. Option 2 overfits per-asset unless explicitly controlled.

### Pareto Front Visualization and Analysis

For every multi-objective run, produce:
- Pareto front scatter plot: Sharpe vs. max drawdown vs. complexity (3D or projected 2D pairs)
- Knee point identification: the strategy on the Pareto front with the best tradeoff between all objectives
- Hypervolume metric per generation: tracks evolutionary progress without cherry-picking

The knee point strategy — not the maximum Sharpe strategy — should be the headline result. Maximum Sharpe on a Pareto front is typically the most overfit solution.

### Variable Frequency Analysis

After evolution, count how often each primitive and terminal appears in the top-N Pareto front trees. This is the GP equivalent of feature importance and reveals which signals the evolutionary process found genuinely useful. Report this as a ranked table per experiment.

High-frequency primitives across independent runs (different seeds) have higher confidence as real signals. Primitives that appear in only one run are likely noise artifacts.

### Program Behavior Clustering

Evolved trees with different structures can produce nearly identical signals on the training data (semantic equivalence). Cluster the top Pareto front individuals by their signal correlation before selecting diverse strategies for OOS evaluation. This prevents reporting multiple "independent" strategies that are actually the same bet.

Warm Start GP (arxiv 2412.00896) showed that traditional GP produces identical or near-identical factors across runs. Tracking and de-duplicating by signal correlation is essential for honest reporting.

### Regime-Conditioned Evaluation

Run OOS evaluation separately on:
- Bull market periods (BTC price > 200-day MA)
- Bear market periods (BTC price < 200-day MA, declining)
- Sideways/high-volatility periods

A strategy with positive Sharpe in all three regimes is far more credible than one with a high average Sharpe dominated by one regime.

### Tree Visualization

Export evolved trees as:
- Graphviz `.dot` files for publication figures
- Human-readable infix notation (e.g., `rolling_mean(close, 14) > zscore(volume, 30)`)

This is required for interpretability claims and for the community-facing GitHub repo. A strategy that cannot be read cannot be peer-reviewed.

### On-Chain Features as Terminals

MVRV, NVT, exchange net inflow/outflow, SOPR, active addresses: these are public blockchain metrics that have no direct equivalent in equity markets. Including them as terminals tests whether the GP can discover on-chain + price signal combinations that generalize — a genuinely novel research contribution.

Practical note: on-chain data is daily or lower frequency. If price data is hourly/15-minute, resample on-chain features to match or treat them as slowly-varying constants within intraday windows. Document the resampling method explicitly.

---

## Anti-Features (Avoid)

Things that seem helpful but systematically hurt research quality.

---

### Too Many Primitives

**What goes wrong:** A large primitive set expands the search space faster than any population can explore it. The GP finds spuriously complex trees that have strong in-sample fitness by chance, not signal.

**The evidence:** Warm Start GP showed that fewer than 3% of randomly generated alphas achieve IC > 0.03 with an unconstrained search space. Constraining the space raised effective alpha density to 13%+.

**The rule:** Start with 10–15 primitives maximum. Each new primitive must earn its place with a prior hypothesis. "Let the GP figure it out" is not a hypothesis.

### Single-Run Best-of Reporting

**What goes wrong:** Reporting the best single run out of 10 as if it represents typical performance. This is the most common misrepresentation in GP finance papers.

**Prevention:** Report median and IQR across all runs. Report DSR. Report how many of the 10 runs produced positive OOS Sharpe, not just the best one's Sharpe.

### In-Sample Fitness as the Headline Result

**What goes wrong:** High in-sample Sharpe is trivially achievable by bloated trees. It measures the GP's ability to memorize, not generalize.

**The rule:** In-sample results are only reported to show the training process worked. The paper's headline number is always OOS Sharpe. Always.

### Refitting to OOS

**What goes wrong:** Seeing OOS results, adjusting hyperparameters, rerunning. The OOS split ceases to be OOS.

**Prevention:** Lock the OOS split before any experiment. Treat it as if it did not exist until the final reporting step. Use the validation split for hyperparameter search.

### Evaluating Only Trade Frequency Without Statistical Significance

**What goes wrong:** A strategy making 5 trades over 3 years might show 3.0 Sharpe. This is statistically meaningless.

**The rule:** Minimum 50 completed trades for any reported result. Below that, set fitness to zero and exclude from reporting. This is a hard constraint, not a soft preference.

### Overly Complex Tree Depth Limits

**What goes wrong:** Setting depth limit to 20+ allows the GP to evolve strategies that are effectively opaque lookup tables over the training data. They have zero interpretability and near-certain OOS failure.

**The rule:** Maximum depth 10–13. If a good strategy cannot be expressed in 13 levels, the primitive set is wrong, not the depth limit.

### Per-Asset Fitness Without Cross-Asset Validation

**What goes wrong:** Evolving a strategy that fits BTC 2020–2022 perfectly by exploiting idiosyncratic market microstructure. The strategy fails on ETH and fails on any BTC data outside its training window.

**Prevention:** Multi-asset fitness (Sharpe averaged across BTC + ETH + at least one altcoin) is table stakes for a multi-asset system. A strategy that generalizes across assets is worth reporting. One that fits one asset is a data artifact.

### Missing Transaction Cost Model

**What goes wrong:** GP maximizes gross Sharpe, producing high-frequency strategies that appear profitable before costs but bleed out on fees and slippage.

**The rule:** Include a realistic transaction cost model in every fitness evaluation. For crypto: 0.05–0.10% per side for liquid pairs (BTC, ETH), 0.10–0.25% for smaller altcoins. This is not a nice-to-have — it determines whether evolved strategies are physically tradeable.

---

## Feature Dependencies

```
Strongly-Typed Primitive Set
    → required by → Multi-Asset Evolution (type constraints scale correctly)
    → required by → Tree Visualization (types make infix notation interpretable)

NSGA-II Multi-Objective Fitness
    → requires → Complexity objective (else NSGA-II reduces to single-objective)
    → produces → Pareto Front Visualization
    → enables → Knee Point Selection (principal output strategy)

Walk-Forward OOS Structure
    → required by → All fitness reporting
    → enables → Regime-Conditioned Evaluation

Multiple Independent Runs (min 10)
    → required by → Deflated Sharpe Ratio reporting
    → required by → Variable Frequency Analysis (stability across seeds)
    → required by → Program Behavior Clustering (de-duplicate across runs)

Transaction Cost Model
    → required by → Fitness Evaluation (integrated, not post-hoc)
    → required by → Credible OOS Sharpe (gross ≠ net)

Minimum Trade Count Filter (≥50 trades)
    → required by → All statistical reporting
    → implemented in → Fitness Evaluation (zero-fitness rejection)

Data Checksumming
    → required by → Reproducibility (config + seed alone insufficient if data changes)
```

---

## MVP Recommendation

For a research-credible v1, prioritize in this order:

**Must ship in v1:**
1. Strongly-typed primitive set with 10–12 terminals (price, volume, 2–3 on-chain), vector operators, and signal aggregation layer
2. NSGA-II with Sharpe + max drawdown + complexity as the three objectives
3. Walk-forward OOS split enforced structurally (not by convention)
4. Minimum 10 independent seeded runs per experiment
5. Transaction cost model (flat bps per trade)
6. Hard tree depth and size limits (depth ≤ 13, size ≤ 90)
7. Config YAML + seed logging for every run
8. Reporting: median OOS Sharpe ± IQR across runs, Pareto front plot, DSR

**Defer to v2:**
- GT-Score as alternative fitness (implement for ablation after baseline is validated)
- Program behavior clustering (useful but adds complexity before baseline exists)
- HARM-GP bloat control (NSGA-II complexity objective is sufficient for v1; add HARM-GP if bloat persists empirically)
- Regime-conditioned evaluation (run after achieving positive OOS Sharpe; diagnose regime dependence as a second-order question)
- Semantic / geometric crossover (high complexity, uncertain payoff; standard subtree crossover is well-established)

---

## Sources

| Source | Confidence | Key Contribution |
|---|---|---|
| [Evolving Financial Trading Strategies with VGP (arxiv 2504.05418)](https://arxiv.org/abs/2504.05418) | HIGH | Primitive sets, STGP superiority finding, depth/size limits, evaluation protocol |
| [DEAP GP Documentation](https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html) | HIGH | PrimitiveSetTyped, HARM-GP, selDoubleTournament, cxOnePoint |
| [The GT-Score (arxiv 2602.00080)](https://arxiv.org/abs/2602.00080) | HIGH | Composite fitness formula, 50-trade minimum, 98% generalization improvement |
| [Building Cross-Sectional Strategies via Geometric Semantic GP (GECCO 2025)](https://link.springer.com/chapter/10.1007/978-3-031-90062-4_2) | MEDIUM | Semantic GP for finance; better than standard GP for cross-sectional problems |
| [Multi-objective GP with NSGA-II for trading (Springer 2025)](https://link.springer.com/article/10.1007/s10462-025-11390-9) | MEDIUM | MOO3 framework, directional changes + modified Sharpe |
| [Backtest Overfitting comparison: CPCV vs WF (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110) | HIGH | CPCV > WF for overfitting prevention; PBO and DSR methodology |
| [Alpha Mining via Warm Start GP (arxiv 2412.00896)](https://arxiv.org/abs/2412.00896) | MEDIUM | Search space constraint findings, 3% vs 13% effective alpha density |
| [Kostadinov GP Trading Overview](https://fabian-kostadinov.github.io/2014/11/01/evolving-trading-strategies-with-genetic-programming-gp-parameters-and-operators/) | MEDIUM | Practical GP parameter recommendations; tournament selection |
| [GP, Validation Sets, and Parsimony Pressure (Springer)](https://link.springer.com/chapter/10.1007/11729976_10) | HIGH | Parsimony pressure theory and interaction with validation |
| [Interpretable Walk-Forward Framework (arxiv 2512.12924)](https://arxiv.org/abs/2512.12924) | MEDIUM | 34-period rolling WF, statistical transparency, regime dependence |
