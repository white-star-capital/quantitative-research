# Research Summary: VGP for Crypto Trading

**Project:** Vector Genetic Programming (VGP) for Evolving Crypto Trading Strategies
**Domain:** Genetic Programming + Quantitative Finance + Crypto
**Researched:** 2026-06-03
**Confidence:** HIGH

---

## Executive Summary

This project builds a research-grade Genetic Programming system that evolves multi-asset crypto trading strategies using vectorized backtesting. The canonical architecture — confirmed by arxiv:2504.05418 and DEAP's official documentation — is a strongly-typed GP system (DEAP 1.4.4 with `PrimitiveSetTyped`) paired with a vectorized backtester (vectorbt 1.0.0) and multi-objective NSGA-II selection optimizing Sharpe ratio, max drawdown, and tree complexity simultaneously. The key research insight driving this design is that passing full time-series arrays through numpy ufuncs lets GP trees evaluate across thousands of bars without Python loops — which is what makes population-scale evolution computationally feasible.

The recommended build sequence is strictly dependency-ordered: data pipeline first, then primitive registry and tree mechanics, then the TreeEvaluator (the highest-risk and highest-novelty component), then vectorbt integration, then the evolution loop, then parallelism. The TreeEvaluator is where DEAP's `compile()` output meets numpy's broadcasting over multi-asset time series — this integration has the most surface area for subtle bugs and must be validated in isolation before the evolution loop touches it. Parallelism must not be introduced until a single-threaded end-to-end run completes correctly.

The three highest-probability project killers are: (1) lookahead bias via signal timing off-by-one in vectorbt — GP will evolve trees that exploit it invisibly and the backtest will look excellent; (2) fitness landscape gaming through degenerate archetypes (buy-and-hold masquerading as alpha, few-trade statistical artifacts, complexity bombs) — defeated by a hard minimum 50-trade filter plus three-objective NSGA-II fitness; and (3) the numpy 2.3 / numba hard incompatibility — a fresh `pip install numpy` in 2026 installs 2.3 which breaks numba with no fallback, so pinning `numpy<2.3` is a day-one environment requirement.

---

## The Stack (What to Use)

The stack is fully determined by two core constraints: DEAP is the only mature Python GP library with native `PrimitiveSetTyped` and `selNSGA2`, and vectorbt is the only open-source backtester that vectorizes across an entire population simultaneously. Everything else follows from these two choices.

**Core technologies (pinned versions matter):**

| Technology | Version | Why |
|---|---|---|
| DEAP | ==1.4.4 | Only library with PrimitiveSetTyped, selNSGA2, and single-line parallel map. Latest stable (Apr 2026). |
| vectorbt | ==1.0.0 | Vectorized portfolio simulation across 100-500 individuals in ~70-100ms. 1.0.0 adds optional Rust engine. |
| numpy | >=2.0,<2.3 | Upper bound is mandatory — numba hard-requires <2.3 as of Jun 2026. |
| numba | >=0.61.2 | 0.61.2 adds explicit NumPy 2.2 support. Required for vectorbt hot paths. |
| pandas | >=3.0.0 | CoW semantics are default in 3.0. Use `.to_numpy()` not `.values`. No chained assignment. |
| pyarrow | >=18.0 | Default pandas 3.0 parquet engine; faster than fastparquet. |
| mlflow | >=2.14 | Local-first experiment tracking. Parent/child run structure maps to experiment → seed run. |
| dill | >=0.3.8 | Drop-in pickle replacement for DEAP checkpointing. Handles closures that stdlib pickle cannot. |
| multiprocessing | stdlib | Single-line parallel evaluation: `toolbox.register("map", pool.map)`. Sufficient for 100-500 individuals. |

**What not to use:** gplearn (sklearn-only symbolic regression, no STGP), PyGAD (GA not GP), backtrader (event-driven, cannot vectorize a population), polars (constant boundary conversion with vectorbt/numba), Ray (justified only at >10,000 individuals — defer to v2), SCOOP (cluster deployment overhead not warranted for single workstation).

**Critical version landmine:** `numpy<2.3` must be pinned from day one. NumPy 2.3 is the default install in fresh 2026 environments and breaks numba with an ImportError, no fallback.

---

## Table Stakes Features

Without these, the system is not credible as research. There is no v1 without all of them.

**Primitive Set (must have all three layers):**
- Price/volume/on-chain vector terminals: 8-12 total at v1 launch. Start small — more terminals expand the search space faster than any population can explore, increasing the proportion of overfitted results.
- Vector operators: element-wise arithmetic, rolling mean/std/max/min, zscore, protected division (return 1.0 on div-by-zero). All must accept and return numpy arrays, never Python scalars or pandas objects.
- Signal aggregation layer: crossover operators and threshold comparators producing `BoolVec`, with a typed root node that forces tree output to be a tradeable signal. This is the STGP type constraint that makes uninterpretable trees impossible to generate.

**Fitness and Evaluation:**
- NSGA-II with three objectives: Sharpe (maximize), max drawdown (minimize), tree node count (minimize). Three objectives is the right number — four or more dilutes selection pressure and makes Pareto fronts unanalyzable.
- Minimum 50-trade hard filter: set fitness to worst-possible (not zero, not NaN) for any individual with fewer than 50 completed trades. Below 50 trades, all statistics are statistically meaningless. This is a constraint, not a soft preference.
- Transaction costs baked into every fitness call: 0.2% per side default for crypto (conservative). Gross-of-cost Sharpe is not a reportable result.
- OOS data locked before any experiment begins. Training split only during evolution. OOS touched exactly once at the end.

**Anti-Overfitting (all required; missing any one invalidates results):**
- Walk-forward OOS split with temporal ordering enforced structurally (assert `test_start > train_end` in WalkForwardSplitter constructor)
- Minimum 10 independent runs per experiment; report median OOS Sharpe ± IQR, not the best single run. Single-run results are anecdotes.
- Randomized training window start per generation (100-day buffer, random first day) — low-cost, high-value anti-memorization technique from the VGP paper
- Deflated Sharpe Ratio (DSR) reported whenever N evaluated strategies > 20, which is always in GP. At 50,000 evaluated strategies (100 generations × 500 individuals), IS Sharpe needs to exceed ~3.0 before it is likely real.

**Reproducibility:**
- Both `random.seed()` and `np.random.seed()` set before every run (DEAP uses Python's `random` module independently from numpy)
- Config YAML with all hyperparameters; no magic numbers in code
- Parquet files checksummed (SHA-256); checksum logged with every experiment
- Checkpoint every N generations via dill (not stdlib pickle — dill handles closures)

**Defer to v2:**
- GT-Score as alternative fitness (run after baseline is validated; useful for ablation)
- Program behavior clustering / semantic de-duplication of Pareto front
- HARM-GP bloat control (NSGA-II complexity objective is sufficient for v1)
- Regime-conditioned evaluation split by bull/bear/sideways
- Ray / SCOOP distributed parallelism

---

## Critical Architecture Decisions

These four decisions constrain everything downstream. Getting them wrong requires rewriting core components.

### Decision 1: Strongly-Typed GP, shared tree across assets (non-negotiable)

Use `PrimitiveSetTyped` with distinct Python classes — not built-in types — for `Vector`, `Scalar`, and `WindowSize`. The VGP paper confirmed empirically that STGP is always among the best performers; untyped GP is always among the worst, across all three test assets and seven years of data. This is the single highest-confidence finding in recent VGP-for-finance literature.

Evaluate the same tree on multiple assets using different feature inputs, producing a `[T × A]` signal matrix. Multi-asset fitness (Sharpe averaged across BTC + ETH + at least one altcoin) directly tests generalization — a strategy that works across assets is a finding; one that works only on BTC is a data artifact.

### Decision 2: Fitness tuple format for NSGA-II

The fitness tuple must be `(sharpe_ratio, -max_drawdown, -tree_node_count)` — all signed so DEAP maximizes everything with `weights=(1.0, 1.0, 1.0)`. Tree node count as the third objective implements parsimony pressure without a calibrated penalty coefficient.

`creator.Individual` and `creator.FitnessMulti` must be defined at module level in a dedicated `gp_types.py` file. This is a hard constraint for `multiprocessing.Pool` pickling, not a style preference.

### Decision 3: Hard component boundary between GP engine and backtesting

`EvolutionLoop` must not import vectorbt. `BacktestRunner` must not import DEAP. The interface between them is a typed `FitnessVector = tuple[float, ...]`. This separation makes the backtesting engine replaceable without touching GP code.

The `TreeEvaluator → BacktestRunner` interface is the most important in the system: TreeEvaluator produces `entries: bool[T×A]`, `exits: bool[T×A]`, `prices: float32[T×A]`. BacktestRunner consumes this and returns the fitness tuple. Nothing crosses this boundary except numpy arrays and plain tuples.

### Decision 4: Feature arrays in shared memory, not pickled per evaluation

When parallelizing, place the `[T × F × A]` float32 feature matrix in `multiprocessing.shared_memory.SharedMemory` once before the Pool is created. Workers receive only the GP tree string (small) and the shared memory block name + shape + dtype (tiny metadata), then reconstruct a zero-copy numpy view. Pickling a large numpy array per individual per generation would dominate all other runtime costs.

Features must be computed once before evolution begins and held read-only. FeatureEngine must never run inside the evaluation loop.

---

## Build Order

This is the critical path. Each phase blocks the next. Do not deviate from this sequence.

```
Phase 1: Data Pipeline (no blockers)
    DataLoader: parquet → aligned pandas DataFrame (multi-asset OHLCV)
    FeatureEngine: indicators → float32 np.ndarray [T × F × A]
    WalkForwardSplitter: train/test views with temporal ordering assertion
    Environment validation: numba/numpy/vectorbt smoke test
    Backtesting harness smoke test: confirm signal shift (lookahead prevention)
    DELIVERABLE: validated feature matrix + confirmed numerics stack

Phase 2: Primitive Registry + Fitness Design (requires Phase 1 for data shapes)
    PrimitiveSetTyped: Vector/Scalar/WindowSize type tokens (Python classes, not builtins)
    Primitive functions: all module-level, all accept/return numpy arrays, no lambdas
    Protected operators: div_safe, log_safe, sqrt_safe (no NaN propagation)
    Fitness function: Sharpe + drawdown + node count with edge case handling
    Minimum trade filter: hard rejection at < 50 trades, fitness = worst-possible
    Transaction cost model: 0.2% per side baked into BacktestRunner
    Type audit: generate 1000 random trees, zero IndexErrors allowed
    DELIVERABLE: validated primitive set + fitness function

Phase 3: TreeEvaluator — the hardest component (requires Phases 1-2)
    compile() via gp.compile() → Callable
    execute(): pass full 2D [T × F] arrays, numpy broadcasting vectorizes across time
    SignalInterpreter: threshold-based float → bool entries/exits
    Signal shift enforcement: fshift(1) applied structurally at this layer
    DELIVERABLE: single-tree evaluation verified manually on known-correct signals

Phase 4: BacktestRunner + vectorbt Integration (requires Phase 3)
    vbt.Portfolio.from_signals() wiring with freq, fees, slippage
    FitnessVector extraction: (sharpe, -max_drawdown, -tree_size)
    Numba JIT warmup probe before any evolution starts
    Degenerate case tests: zero trades, all-True signal, NaN tree output
    DELIVERABLE: fitness function verified on synthetic individuals

Phase 5: EvolutionLoop — single-threaded (requires Phases 2-4)
    DEAP toolbox wiring: selNSGA2, cxOnePoint, mutUniform, eaMuPlusLambda
    Population initialization: ramped half-and-half
    Generation loop with logbook + HallOfFame
    Static depth limit: max_depth=8 (override DEAP's default of 17)
    DELIVERABLE: one complete generation end-to-end, single-threaded, results inspected

Phase 6: Parallelism (requires Phase 5 validated — do not skip this gate)
    Shared memory setup for feature matrix
    Pool wiring: toolbox.register("map", pool.map)
    JIT warmup probes in all workers via Pool initializer
    Pickle test: pickle.dumps(toolbox.evaluate) before Pool launch
    DELIVERABLE: parallel generation, performance benchmarked

Phase 7: Experiment Infrastructure (scaffold in Phase 1, complete here)
    Config YAML/dataclass: all hyperparameters, seeds, data dates
    MLflow tracking: parent experiment → nested seed runs, per-generation metrics
    Checkpointing: dill pickle of {population, generation, hof, logbook, rndstate}
    Multi-seed orchestration: 10 independent runs per experiment
    DELIVERABLE: reproducible, logged, resumable experiments

Phase 8: Walk-Forward Validation + OOS Reporting
    Multi-window walk-forward splits
    OOS evaluation: single pass on held-out data, after all hyperparameter decisions frozen
    Reporting: median OOS Sharpe ± IQR, DSR, Pareto front visualization
    Tree export: graphviz .dot, human-readable infix notation
    Variable frequency analysis: which primitives appear most across Pareto front + runs
    DELIVERABLE: credible, publishable research results
```

**Critical path:** Phase 1 → 2 → 3 → 4 → 5. Phases 6 and 7 can overlap once Phase 5 is validated. Phase 8 requires all preceding phases.

TreeEvaluator (Phase 3) is the highest-risk single component. It is where the research novelty lives and where integration bugs between DEAP's compile() and numpy's broadcasting are most likely to surface. Budget disproportionate testing time here.

---

## Top 3 Risks to Watch

### Risk 1: Lookahead bias in vectorbt signal timing

**Probability:** Near-certain if not explicitly prevented. **Impact:** Complete result invalidation — produces a convincing-looking lie.

GP will evolve trees that exploit close-price lookahead because natural selection is indifferent to whether an advantage is real or an artifact. The vectorbt documentation explicitly warns: "If you generated signals using close price, don't forget to shift your signals by one tick forward." Warning signs include OOS Sharpe > 2.0 on simple technical indicator combinations, and strategies that do not degrade when latency assumptions are increased.

**Mitigation:** Apply `signals.vbt.fshift(1)` as a structural invariant inside TreeEvaluator, not a configuration option. Write a CI-level test that compares a known strategy's result on shifted vs. unshifted signals and asserts they differ. This test gates Phase 4 completion.

### Risk 2: numpy 2.3 / numba hard incompatibility

**Probability:** Certain on any fresh Python environment created after June 2026. **Impact:** Complete environment failure before writing any project code.

`pip install numpy` installs 2.3. `pip install numba` in that environment raises `ImportError: Numba needs NumPy 2.2 or less.` This is confirmed by multiple downstream projects (xcdat, cuml). There is no workaround except pinning.

**Mitigation:** Pin `numpy>=2.0,<2.3` in `pyproject.toml` in the first commit. Run the numba/numpy/vectorbt smoke test as the first action in Phase 1. Use this environment also to explicitly set `vbt.settings.set_option("engine", "numba")` to prevent Rust engine auto-dispatch from introducing machine-dependent behavior.

### Risk 3: Degenerate evolution from naive fitness function

**Probability:** High — documented failure mode in GP finance literature across multiple independent sources. **Impact:** 100+ generation runs producing strategies that appear valid until OOS evaluation reveals them as artifacts.

The three degenerate archetypes are: buy-and-hold (almost always long, high Sharpe on bull datasets), few-trade artifacts (2-3 lucky trades show Sharpe 4+ on 8 trades), and complexity bombs (deep trees that perfectly memorize training noise). GP's crossover operator amplifies all three by preferentially sampling from the highest-fitness (most degenerate) individuals.

**Mitigation:** The minimum 50-trade hard filter plus three-objective NSGA-II fitness together defeat all three archetypes. These must be in place before the first evolution run. Adding them after observing degenerate populations invalidates earlier results. Additionally, track IS/OOS Sharpe ratio per generation as a live diagnostic — a ratio > 2x signals degenerate evolution in progress.

---

## Tensions and Open Questions

**Tension 1: numpy version ceiling vs. future compatibility**

The `numpy<2.3` pin is required today but will become wrong when numba adds 2.3 support. Resolution: track numba release notes and update the pin when support is confirmed. This is a maintenance task, not a design flaw.

**Tension 2: vectorbt 1.0.0 migration guide is sparse**

Core `from_signals` signature is confirmed stable from 0.x to 1.0.0, and the Rust engine is additive. However, no official migration guide was found for the 1.0.0 rewrite. The `from_signals` API smoke test in Phase 4 is the primary guard against undocumented breaking changes. Do not assume 0.x patterns carry forward without explicit testing against 1.0 docs.

**Tension 3: On-chain data frequency mismatch**

On-chain metrics (MVRV, NVT, SOPR, exchange flows) are daily or lower frequency. If price data is intraday, resampling must use `resample().last()` aligned to the last bar of each day to avoid multi-timeframe lookahead. Simplest v1 resolution: use daily OHLCV price data throughout, eliminating the mismatch entirely.

**Open question: on-chain data availability in parquet files**

Research assumes on-chain metrics are present in the parquet dataset. If they are absent or sparse, defer on-chain terminals and constrain the primitive set to price/volume for v1. Do not let data availability block the core GP system.

**Open question: optimal population size and generation count**

200 individuals / 100 generations is the literature starting point (VGP paper). These are hyperparameters that should be benchmarked on actual hardware before committing to a 10-seed experiment run. A single 10-generation test run is sufficient to estimate cost.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | DEAP 1.4.4 and vectorbt 1.0.0 confirmed on PyPI; numpy/numba compatibility verified against official release notes; pandas 3.0 CoW confirmed in official docs |
| Features | HIGH | Core findings (STGP superiority, NSGA-II objectives, 50-trade minimum, DSR reporting) backed by arxiv:2504.05418 and arxiv:2602.00080, both peer-reviewed academic sources |
| Architecture | HIGH | Component boundaries, interface contracts, and build order grounded in arxiv:2504.05418 + DEAP official docs + vectorbt docs |
| Pitfalls | HIGH | Lookahead, numba/numpy incompatibility, and degenerate evolution confirmed by multiple independent sources; not speculative |

**Overall confidence:** HIGH

### Gaps to Address

- **vectorbt 1.0.0 breaking changes:** No official migration guide. Verify all API calls against 1.0 docs during Phase 4 — do not assume 0.x knowledge.
- **On-chain data sourcing:** If absent from parquet files, defer on-chain terminals entirely. The core VGP system is valid without them.
- **Population/generation count tuning:** Benchmark on actual hardware before scheduling 10-seed experiment runs. Literature values are starting points, not prescriptions.

---

## Sources

### Primary (HIGH confidence)
- arxiv:2504.05418 — VGP architecture, STGP superiority, vectorized execution pattern, primitive set design, 10-run protocol
- DEAP ReadTheDocs (1.4.3/1.4.4) — PrimitiveSetTyped, selNSGA2, multiprocessing pattern, checkpointing, HARM-GP
- vectorbt.dev API docs (1.0.0) — from_signals signature, Portfolio.stats(), freq requirement, Rust engine
- numba release notes 0.61.0, 0.61.2 — NumPy 2.1/2.2 support explicitly stated
- pandas 3.0 what's new — CoW semantics, .to_numpy() requirement

### Secondary (MEDIUM confidence)
- arxiv:2602.00080 (GT-Score) — minimum 50-trade filter, composite fitness formula, 98% generalization improvement
- arxiv:2412.00896 (Warm Start GP) — primitive set size constraint, 3% vs 13% effective alpha density
- Kostadinov (2014/2015) — practical GP parameter recommendations, parsimony pressure
- Bailey et al. (PBO framework) — multiple testing correction, sqrt(log(N)) Sharpe threshold
- Coinbase Institutional Research — survivorship bias inflates crypto backtests 17-22% annually

### Tertiary
- vectorbt 1.0.0 vs 0.x delta — no official migration guide; core API inferred stable from PyPI release and community observation

---

*Research completed: 2026-06-03*
*Ready for roadmap: yes*
