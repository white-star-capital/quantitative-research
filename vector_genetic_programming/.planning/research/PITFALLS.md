# Pitfalls Research: VGP for Crypto Trading

**Domain:** Genetic Programming for quantitative crypto trading
**Researched:** 2026-06-03
**Overall confidence:** HIGH (multiple independent sources, academic literature, primary tool docs)

---

## Critical (Project-Killers)

These mistakes cause complete invalidation of results. A system that hits any of these produces a convincing-looking lie.

---

### 1. Lookahead Bias via Signal Timing Off-by-One

**What it is:** The single most common and insidious error in vectorized backtesting. A signal computed at bar `t` using `close[t]` is executed at bar `t` (the same close), meaning the backtest assumes you knew the closing price before it closed. In reality you can only act at bar `t+1`.

In vectorbt, `Portfolio.from_signals()` and `Portfolio.from_orders()` execute at the price you supply — if you pass the close series directly without shifting, you are filling at the price that generated the signal. The official docs explicitly warn: *"If you generated signals using close price, don't forget to shift your signals by one tick forward, e.g., `signals.vbt.fshift(1)`."*

For a VGP system this is especially dangerous because GP will evolve trees that exploit the bias — they won't "look" like lookahead, they'll just be very good at the "close of the current bar" which is impossible to trade.

**Warning signs:**
- OOS Sharpe > 2.0 on simple technical indicator combinations
- Strategy performance does not degrade when you increase latency assumptions
- Win rates above 60% on trend strategies with no drawdown control

**Prevention:**
- Always pass `open` of bar `t+1` as execution price, not `close[t]`
- Use `signals.vbt.fshift(1)` before passing any signal array to portfolio methods
- Write a canonical timing-correctness test: manually verify first 10 trades against raw data
- Use `from_signals(price=open.shift(-1))` pattern for next-bar execution

**Phase to address:** Phase 1 (backtesting harness construction). Build this into the evaluation layer before any evolution runs. A single automated sanity check — compare backtest on forward-shifted signals vs. unshifted — should be a CI gate.

---

### 2. GP Fitness Landscape Gaming (Degenerate Evolution)

**What it is:** GP will find the highest-fitness individuals, not the "best trading strategies." These are different objectives. The evolutionary search reliably discovers three degenerate archetypes:

1. **Buy-and-hold masquerading as a strategy:** Trees that are almost always long. On a bull crypto dataset this scores extremely high Sharpe with no alpha. Looks great in IS, collapses immediately OOS in sideways/bear conditions.

2. **Lucky few-trade artifacts:** A tree that happened to catch 2-3 extraordinary moves (COVID crash recovery, BTC halving) with perfect timing. Sharpe of 4+ on 8 trades. Not a strategy — a statistical artifact. GP assigns maximum fitness and reproduces it aggressively.

3. **Complexity bombs:** Deep trees that perfectly fit training noise. These show near-zero drawdown in IS because they find the exact sequence of false signals that happened to work on that specific data slice. Total collapse OOS.

Research (Kostadinov 2014, confirmed by multiple academic sources) shows that fitness-proportional selection combined with simple return-maximization amplifies all three patterns simultaneously, because the crossover operator preferentially samples from the fittest (most overfitted) individuals.

**Warning signs:**
- IS Sharpe >> OOS Sharpe (ratio > 2x)
- Evolved trees cluster at maximum depth limit
- Best-of-generation fitness jumps sharply in first 10-20 generations then plateaus
- Winning strategies have fewer than 20 trades in the training window

**Prevention:**
- Minimum trade count filter: reject any individual with < N trades (N = 30 minimum, ideally 50+) as fitness = -infinity
- Multi-objective fitness: simultaneously optimize Sharpe AND penalize drawdown AND require minimum trade frequency
- Separate fitness into: `raw_sharpe * trade_count_penalty * complexity_penalty`
- Use tournament selection instead of fitness-proportional to reduce dominance effects
- Early stopping: track validation Sharpe separately; stop if IS/OOS diverges beyond threshold

**Phase to address:** Phase 2 (fitness function design). This is the most important design decision in the entire project. Get it wrong and all subsequent evolution is garbage.

---

### 3. Tree Bloat Destroying Search Quality

**What it is:** GP trees grow unboundedly over generations. The crossover operator systematically favors larger individuals because larger trees have more crossover points, making them more likely to be selected as parents and to contribute large subtrees to offspring. Left unchecked, trees grow to the depth limit within 20-50 generations, consuming all computational budget on evaluating giant trees that are no more fit than small ones.

Bloated trees cause three compounding problems:
1. Each evaluation is slow (deep tree = many numpy operations in the VGP context)
2. Bloated trees are inherently overfit — they carry "introns" (neutral but space-filling subtrees) that memorize training data
3. The search space explodes exponentially with depth, destroying GP's ability to find meaningful improvements

DEAP's default `staticLimit` sets a maximum height of 17 — this prevents the worst blowup but does not prevent the gradual accumulation of complexity within that limit.

**Warning signs:**
- Average tree size grows monotonically over generations without plateau
- Mean tree size approaches the depth limit after generation 30
- Fitness improvement stops while tree complexity keeps growing
- Evaluation time per generation increases over time

**Prevention:**
- `staticLimit(key=operator.attrgetter("height"), max_value=8)` — use 8, not 17. Deep trees in financial GP are almost always overfit.
- Add explicit parsimony pressure to fitness: `fitness = sharpe - alpha * tree_size` where alpha is calibrated so a tree twice as complex needs ~0.2 additional Sharpe to survive
- Track and log tree size statistics every generation — make this a monitored metric, not an afterthought
- Consider lexicase selection or NSGA-II multi-objective: treat size as a second objective to minimize simultaneously

**Phase to address:** Phase 2 (GP infrastructure). Set these limits before first evolution run; retroactively adding them invalidates earlier results.

---

### 4. Data Leakage Through Feature Engineering

**What it is:** Any feature derived from data that wasn't available at bar `t` contaminates the primitive set. This includes:

- Rolling statistics computed on the full dataset before splitting (e.g., `zscore = (x - x.mean()) / x.std()` where mean/std use future data)
- On-chain metrics with reporting delays used as if they were real-time
- Cross-asset features that implicitly encode future information (e.g., a ratio that normalizes by end-of-period values)
- Any scaler (`MinMaxScaler`, `StandardScaler`) fit on the full dataset before train/test split

In a VGP context this is especially severe because primitives are reused across all individuals. If even one primitive in the primitive set has leakage, every evolved tree that uses it is contaminated. The contamination is invisible — the primitive looks like any other feature.

**Warning signs:**
- Primitives that require knowing the distribution of the full series (global min, global max, global mean)
- On-chain primitives sourced from endpoints without documented latency
- Feature normalization applied before train/test split

**Prevention:**
- All features must be computable as a pure rolling function of past data only: `rolling_mean(window=W)` not `global_mean()`
- Document the exact reporting latency for every on-chain metric — default to pessimistic assumption (if unknown, assume 24h delay)
- Apply all scalers inside the training fold only; pass scaler to test fold via `transform`, never `fit_transform` on combined data
- For VGP primitives: all primitives must be functions of `array[:t]` only — implement a temporal unit test that passes a truncated array and verifies identical result to the equivalent window on the full array

**Phase to address:** Phase 1 (data pipeline) and Phase 2 (primitive set construction). A review checkpoint before any evolution runs should audit every primitive.

---

### 5. Multiple Testing / Data Snooping Inflation

**What it is:** GP runs thousands to millions of fitness evaluations on the same training data. Each evaluation is an implicit hypothesis test. By generation 100 with a population of 500, you have evaluated 50,000+ distinct strategies on the same data. The probability that the best-observed strategy has a positive Sharpe purely by chance is very high — this is the multiple comparisons problem, and GP amplifies it maximally because it is explicitly designed to search the space exhaustively.

The Probability of Backtest Overfitting (PBO) framework (Bailey et al.) shows that for a strategy selected from N trials, the minimum required Sharpe ratio to claim statistical significance scales as `sqrt(log(N))`. At N=50,000 trials, your IS Sharpe needs to exceed ~3.0 before it's likely to be real.

**Warning signs:**
- Best IS Sharpe in final population >> 2.0 but OOS Sharpe < 0.5
- Strategies discovered "exactly" fit known historical events in the training window
- No holdout set was maintained through development — OOS data was peeked at for debugging

**Prevention:**
- Maintain a strict three-way split: train (GP evolution), validation (fitness selection / early stopping), test (never touched until final evaluation)
- Walk-forward OOS: report results across multiple non-overlapping test windows, not a single OOS period
- Require positive Sharpe on at least 3 independent OOS windows before claiming success
- For publication/community release credibility: use combinatorial purged cross-validation (CPCV) rather than simple train/test split

**Phase to address:** Phase 2 (validation framework). Design the split protocol before any evolution runs — retroactively setting aside OOS data after having seen it is not valid.

---

## Serious (Quality Killers)

These won't immediately kill the project but will produce results that don't hold up to scrutiny.

---

### 6. Premature Population Convergence

**What it is:** GP populations converge to a monoculture of similar trees early in evolution (often by generation 20-30). Once the population is homogeneous, crossover produces offspring nearly identical to parents, mutation is the only source of novelty, and evolution effectively stalls. The discovered optimum is a local optimum in the fitness landscape, not the global one.

In financial GP this is compounded because early high-fitness individuals (often buy-and-hold or lucky few-trade strategies — see Pitfall 2) dominate selection aggressively.

**Warning signs:**
- Fitness improvement plateau before generation 50
- Population diversity metric (average pairwise tree edit distance) drops below 10% of initial value
- Tournament selection winners are all identical or near-identical

**Prevention:**
- Implement behavioral diversity: track the signal correlation between individuals, penalize individuals whose signals are highly correlated with existing population members
- Island model / multi-deme evolution: maintain 4-8 subpopulations with occasional migration, preventing full convergence
- Frequency-dependent selection or niching: explicitly reward strategies that are different from what the population already has
- Increase mutation rate if diversity drops below threshold (adaptive mutation)

**Phase to address:** Phase 3 (GP algorithm tuning). Monitor diversity as a primary metric from the first run.

---

### 7. Fitness Function Naive Sharpe Problems

**What it is:** The standard Sharpe ratio has well-documented failure modes as a GP fitness function:

1. **Frequency-scale dependence:** Annualizing daily Sharpe via `sqrt(252)` and hourly Sharpe via `sqrt(8760)` produces incomparable numbers. A strategy that happens to trade at a high-volatility hour looks worse than an equivalent daily strategy. If your primitive set mixes timeframes, fitness comparisons are invalid.

2. **Undefined / degenerate Sharpe:** Zero-volatility return streams (e.g., a tree that never signals a trade) produce `NaN` or `inf` Sharpe. DEAP will not handle `NaN` fitness gracefully — it typically crashes or treats NaN as 0, both wrong.

3. **Short-sample Sharpe inflation:** With fewer than 30 trades, sample Sharpe has enormous confidence intervals. A GP tree with 5 winning trades and 2 losing trades can show Sharpe of 3+ purely from small-sample estimation error.

4. **Ignores tail risk:** A strategy that earns +1% daily for 99 days then loses -200% on day 100 has a high Sharpe ratio. Relevant for crypto where liquidation events and exchange failures are real.

**Warning signs:**
- Fitness function returns NaN or inf for any individual
- Best individuals have very few trades
- Evolved strategies show extreme drawdowns despite high Sharpe

**Prevention:**
- Replace raw Sharpe with: `fitness = sharpe * log(1 + n_trades) * (1 - max_drawdown_penalty)` or similar composite
- Explicitly handle edge cases: if std(returns) == 0 or n_trades < min_trades, fitness = -10.0 (worst possible, not NaN)
- Consider Calmar ratio or Sortino ratio as alternatives or components
- Cap individual trade return contributions to prevent single-event dominance

**Phase to address:** Phase 2 (fitness function). Test fitness function against degenerate inputs before any evolution.

---

### 8. Primitive Set Type Unsafety

**What it is:** DEAP's `PrimitiveSetTyped` enforces return types at the primitive level, but DEAP's tree generator will raise an `IndexError` (not a descriptive error) when the type system reaches an unsatisfiable state — for example, if a primitive requires a `BooleanArray` input but no primitive or terminal can produce `BooleanArray`. This failure mode silently truncates tree generation in some configurations.

For a VGP system mixing price arrays, boolean mask arrays, and scalar parameters, type mismatches are common. If using untyped `PrimitiveSet`, numpy dtype mismatches propagate silently — a tree that mixes float64 and int64 operations may produce incorrect results without raising an error.

**Warning signs:**
- `IndexError` during tree generation with no clear stack trace
- Trees that evaluate to all-True or all-False signals across all bars
- Silent NaN propagation through the primitive evaluation chain

**Prevention:**
- Use `PrimitiveSetTyped` from the start — do not use untyped `PrimitiveSet` for financial data with mixed types
- Before adding any primitive, verify both input and output types are satisfiable by existing terminals/primitives
- Wrap every primitive in a type-assertion test: call it on synthetic data of the expected dtype and verify output dtype
- Add a generation test: generate 1000 random trees and verify none cause IndexError before starting evolution

**Phase to address:** Phase 2 (primitive set design). Type audit before first evolution run.

---

### 9. On-Chain Data Stationarity Assumptions

**What it is:** Most on-chain metrics (NVT ratio, active addresses, SOPR, exchange netflows) are non-stationary — they have unit roots and trend over multi-year timescales. Using raw values as GP primitives means that trees evolved on 2020-2022 data will encounter totally different scale values in 2023-2024 OOS windows, producing signals with entirely different characteristics.

Additionally, metrics like exchange balance (in absolute BTC terms) have secular trends that make them appear predictive in-sample (correlation with price trend) but are actually spurious regressions.

**Warning signs:**
- Augmented Dickey-Fuller test rejects stationarity for primitive input series
- GP discovers trees that are linear in a specific on-chain metric
- OOS performance degrades sharply as the OOS window extends further from the IS period

**Prevention:**
- Always use rates of change, z-scores over rolling windows, or rank-based transforms for on-chain metrics — never raw values
- Rolling z-score: `(x - rolling_mean(x, W)) / rolling_std(x, W)` using W = 30-90 days
- Apply ADF stationarity test to every primitive input series before including in the primitive set
- Document the stationarity transform for every on-chain primitive in the codebase

**Phase to address:** Phase 1 (data pipeline) and Phase 2 (primitive set design).

---

### 10. DEAP Multiprocessing Pickle Failures

**What it is:** DEAP's parallel evaluation requires all objects to be pickleable. Lambda functions, closures over non-pickleable objects, and Python `functools.partial` objects (in Python 3) can fail silently or produce cryptic errors when used as fitness functions or operators with `multiprocessing.Pool`.

Specifically: if your fitness function closes over a pandas DataFrame or a numba-compiled function, it will fail pickling. DEAP's `toolbox.register` with lambda wrappers around your evaluation function is a common pattern that breaks under multiprocessing.

**Warning signs:**
- `_pickle.PicklingError` or `AttributeError: Can't pickle local object` when switching to parallel evaluation
- Evaluation works in single-process mode but hangs or crashes with `Pool`
- Inconsistent results between parallel and serial modes (indicating worker state isn't being shared correctly)

**Prevention:**
- Use module-level functions (not lambdas or closures) as the fitness evaluation callable
- Test pickle serialization explicitly: `import pickle; pickle.dumps(toolbox.evaluate)` before first parallel run
- Pass data to workers via constructor arguments or shared memory, not via closure capture
- Consider using `pathos.multiprocessing` instead of stdlib `multiprocessing` — it uses `dill` for serialization and handles more Python objects

**Phase to address:** Phase 3 (parallel infrastructure).

---

## Common but Recoverable

These are annoying and waste time but don't invalidate results if caught during development.

---

### 11. Transaction Cost Underestimation

**What it is:** Crypto transaction costs are higher than they appear. A strategy that looks profitable at 0.1% per side will often be unprofitable at realistic costs:
- Exchange fees: 0.05-0.10% taker per side (binance spot), 0.02-0.04% maker
- Slippage: 0.05-0.30% depending on order size and liquidity; expands sharply in low-liquidity alts
- Funding rates: perpetual futures cost 0.01-0.03% per 8 hours for long positions during bull markets

For high-frequency GP trees (ones that signal many trades), even 0.15% per side round-trip destroys profitability. A tree signaling 200 trades/year pays ~60% round-trip cost annually.

**Prevention:**
- Default to 0.2% per side as baseline (conservative) — lower this only with explicit justification
- Add slippage model: additional 0.1% for altcoins, 0.05% for BTC/ETH
- In vectorbt: `vbt.Portfolio.from_signals(fees=0.002, slippage=0.001)`
- Report net-of-cost Sharpe as the primary metric; report gross-of-cost as secondary

**Phase to address:** Phase 1 (backtesting harness).

---

### 12. Survivorship Bias in Multi-Asset Universe

**What it is:** If the multi-asset universe is defined as "top 50 crypto by current market cap," it implicitly includes only assets that survived and grew — excluding coins that failed, were delisted, or declined. Research shows survivorship bias alone inflates crypto backtested returns by 17-22% annually (Coinbase Institutional Research).

**Warning signs:**
- Universe defined by current-day rankings/filters rather than point-in-time rankings
- No delisted assets appear in historical data

**Prevention:**
- Define universe using point-in-time market cap data (at each historical date, which assets were in the top N)
- Or: restrict to assets that have been continuously traded for the full backtest period (conservative but clean)
- Document the exact universe construction methodology in any published results

**Phase to address:** Phase 1 (data pipeline).

---

### 13. Reproducibility Failures

**What it is:** GP runs are stochastic. Without explicit seed management, two runs of the same configuration produce different results. This is expected, but several subtle failure modes cause irreproducibility even with fixed seeds:

1. **DEAP's random module is separate from numpy's:** `random.seed(42)` and `np.random.seed(42)` must both be set; they are independent
2. **Multiprocessing breaks seed reproducibility:** Workers in a Pool do not inherit the parent's random state
3. **Numba JIT cache stale state:** If a numba-compiled primitive is modified, the compiled cache may not invalidate, causing the cached (old) version to run
4. **Floating point non-determinism:** Multi-threaded numpy operations (via BLAS) can produce slightly different results depending on thread scheduling

**Prevention:**
- Set both `random.seed(SEED)` and `np.random.seed(SEED)` at experiment start
- Log the full configuration (seed, population size, generations, primitive set hash) with every run
- For parallel evaluation: seed each worker individually using `worker_id * prime + base_seed`
- Clear numba cache with `find . -name "__pycache__" -exec rm -rf {} +` before runs where primitives changed
- Use `PYTHONHASHSEED=<N>` environment variable for deterministic Python hashing

**Phase to address:** Phase 3 (experiment infrastructure).

---

## vectorbt 1.0 Specific

---

### VBT-1: Rust Engine Auto-Dispatch Surprises

**What it is:** vectorbt 1.0 introduced an optional Rust backend (`pip install vectorbt[rust]`). When installed, functions automatically dispatch to the Rust engine based on a global engine setting. The dispatch behavior is `"auto"` by default, which means the same code can silently run different code paths on different machines depending on whether the Rust extension is installed. This breaks reproducibility and makes performance benchmarks machine-dependent.

**Prevention:**
- Explicitly set engine at startup: `vbt.settings.set_option("engine", "numba")` to force consistent behavior
- Document which engine was used in any published benchmark
- Test critical portfolio calculations on both engines and verify numerical equivalence before relying on `"auto"`

---

### VBT-2: Soft Dtype Casting Masks Real Bugs

**What it is:** vectorbt 1.0 introduced "soft dtype casting" — automatic handling of dtype mismatches (e.g., int → float) with configurable warnings instead of hard failures. This is convenient but dangerous: a GP tree that produces integer signals when float is expected will now silently succeed where it previously would have raised an error, masking a type bug in the primitive set.

**Prevention:**
- Enable strict dtype warnings during development: configure warnings to raise exceptions
- Add dtype assertion tests to every primitive: verify output dtype matches specification
- Review the vectorbt upgrade notes carefully if upgrading from 0.x

---

### VBT-3: pandas 2.0 Silent Dtype Changes

**What it is:** vectorbt 0.28.4 added pandas 2.0 compatibility. pandas 2.0 changed many default dtypes (Int64Index removed, datetime dtype resolution changes, int32/int64 inconsistencies). If running vectorbt with pandas 2.x, certain operations that previously returned int64 now return int32, and operations on DatetimeIndex attributes behave differently. These changes can cause silent numerical differences in portfolio calculations.

**Prevention:**
- Pin pandas version in requirements: `pandas>=2.0,<3.0` and test against the specific version
- Run the full backtesting test suite when upgrading pandas
- Check for any `.dt` accessor operations in signal processing code — datetime attribute dtypes changed

---

### VBT-4: Multi-Timeframe Lookahead in Resampling

**What it is:** When computing indicators on a higher timeframe (e.g., daily MACD signals used in an hourly strategy), the standard resampling approach of aggregating 24 hourly bars into one daily bar and using the result for the current period introduces lookahead — the daily bar's close isn't known until the last of the 24 hourly bars completes.

vectorbt does not automatically prevent this. The burden is entirely on the user to align higher-timeframe signals to the correct hourly bar (the last bar of the period, not the first).

**Prevention:**
- When resampling for multi-timeframe signals: align the daily signal to the last hourly bar of that day, not the first
- Use `resample().last()` to get the last bar's value, then forward-fill
- Write explicit unit tests for multi-timeframe alignment with known-correct expected values

---

## numba / NumPy 2.x Compatibility

---

### NUMBA-1: NumPy 2.3 Hard Incompatibility

**What it is:** As of June 2026, Numba hard-requires NumPy <= 2.2. NumPy 2.3 was released and is the default install in fresh environments (`pip install numpy`). Running `pip install vectorbt` in a fresh Python 3.12+ environment will install NumPy 2.3, then installing numba will either fail or raise `ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.3.`

This is an active issue (confirmed in multiple downstream projects including xcdat, cuml). It will likely be resolved in a future numba release, but cannot be assumed.

**Prevention:**
- Pin numpy in requirements: `numpy>=1.24,<2.3`
- Or use conda to manage the numba/numpy version pair — conda resolves these constraints automatically
- Verify the constraint at environment setup time: `python -c "import numba; import numpy; print(numpy.__version__)"`

---

### NUMBA-2: JIT Cache Stale After Primitive Changes

**What it is:** Numba caches compiled functions in `__pycache__`. The cache invalidation logic only checks the main jit function's source file — it does NOT detect changes in functions called by the main function from other modules. In a VGP system where primitives are defined in separate modules and called inside a numba-compiled evaluation loop, modifying a primitive will not invalidate the cache. The old compiled version will silently run.

**Prevention:**
- Set `NUMBA_CACHE_DIR` to a temporary directory and clear it at the start of each experiment run
- Or disable caching during development: `@jit(nopython=True, cache=False)`
- Enable caching only in production runs where primitives are stable
- Add a cache version hash: embed a hash of all primitive source files in a comment visible to numba, forcing recompilation when primitives change

---

### NUMBA-3: NEP-050 Type Semantics Not Implemented

**What it is:** NumPy 2.0 introduced NEP-050, a major change to how type promotion and scalar/array interactions work (e.g., `np.int8(1) + 1` now returns `int8` rather than `int64`). Numba's team has explicitly decided NOT to implement NEP-050 semantics, maintaining binary compatibility but behavioral incompatibility with native NumPy 2.x type promotion.

This means: code that relies on numpy scalar type promotion inside numba-compiled functions may produce different results than the same code run outside numba (in pure numpy). The difference is silent — no error, wrong dtype.

**Prevention:**
- Be explicit about dtypes in all numeric operations within numba-compiled code: `np.float64(x) + np.float64(y)` not `x + y`
- Add numerical equivalence tests that run the same computation in both numba-compiled and pure-numpy modes and compare outputs
- Do not rely on numpy scalar promotion behavior inside numba functions

---

### NUMBA-4: Global Variable Caching as Constants

**What it is:** Numba treats global variables as constants at compile time. If a numba-compiled function reads a global variable (e.g., a lookback window parameter), the compiled cache stores the value at compilation time. Changing the global variable at runtime does NOT change the compiled function's behavior. This is a subtle correctness bug in GP systems where evaluation parameters might be passed as globals.

**Warning signs:**
- Changing a parameter global doesn't change strategy behavior
- Performance is inexplicably identical across different parameter settings

**Prevention:**
- Pass all parameters as function arguments, never as globals, to numba-compiled functions
- Never store mutable state in module-level variables that numba functions read

---

## Phase-Specific Warning Matrix

| Phase | Topic | Likely Pitfall | Mitigation |
|-------|-------|----------------|------------|
| 1 (Data pipeline) | Signal timing | Lookahead via close-price execution | `fshift(1)` harness test before any strategy |
| 1 (Data pipeline) | Universe construction | Survivorship bias | Point-in-time universe definition |
| 1 (Data pipeline) | On-chain features | Non-stationarity | ADF test on every input series |
| 2 (Fitness function) | Fitness design | Degenerate evolution (buy-and-hold, few trades) | Minimum trade filter + multi-objective |
| 2 (Fitness function) | Sharpe computation | NaN/inf Sharpe on degenerate trees | Explicit edge case handling |
| 2 (Primitive set) | Type safety | DEAP IndexError, silent dtype errors | PrimitiveSetTyped + type tests |
| 2 (Primitive set) | Protected operators | Division by zero, NaN propagation | Protected div/log/sqrt primitives |
| 2 (Validation split) | Multiple testing | Data snooping inflation | Three-way split defined before first run |
| 3 (GP tuning) | Tree size | Bloat destroying search | Depth limit=8, parsimony pressure |
| 3 (GP tuning) | Population diversity | Premature convergence | Diversity metric monitoring + niching |
| 3 (Parallel infra) | Multiprocessing | Pickle serialization failures | Module-level eval function, dill test |
| 3 (Parallel infra) | Reproducibility | Seeds not set / multiprocessing breaks seeds | Both random seeds + worker seeding |
| All | numba cache | Stale JIT cache after primitive changes | Clear cache on primitive changes |
| All | numpy version | NumPy 2.3 breaks numba | Pin `numpy<2.3` |

---

## Sources

- Kostadinov, F. (2014). "Evolving Trading Strategies with Genetic Programming - Fitness Functions." http://fabian-kostadinov.github.io/2014/12/22/evolving-trading-strategies-with-genetic-programming-fitness-functions/
- Kostadinov, F. (2015). "Evolving Trading Strategies with Genetic Programming - Punishing Complexity." http://fabian-kostadinov.github.io/2015/01/14/evolving-trading-strategies-with-genetic-programming-punishing-complexity/
- DEAP Documentation: Genetic Programming. https://deap.readthedocs.io/en/master/api/gp.html
- DEAP Documentation: Strongly Typed GP. https://deap.readthedocs.io/en/master/examples/gp_spambase.html
- DEAP Documentation: Using Multiple Processors. https://deap.readthedocs.io/en/master/tutorials/basic/part4.html
- vectorbt Documentation: Portfolio base API. https://vectorbt.dev/api/portfolio/base/
- vectorbt GitHub Issue #101: Multiple timeframes and lookahead. https://github.com/polakowo/vectorbt/issues/101
- Numba Discussion: NumPy 2.x support community update. https://numba.discourse.group/t/numpy-2-x-support-community-update/2815
- Numba Discussion: Communicating NumPy 2.0 Changes. https://numba.discourse.group/t/communicating-numpy-2-0-changes-to-numba-users/2457
- Numba Cache Behaviour Discussion. https://numba.discourse.group/t/cache-behaviour/1520
- Rodrigues, N.M. et al. (2006). "A comparison of bloat control methods for genetic programming." Evolutionary Computation. https://dl.acm.org/doi/10.1162/evco.2006.14.3.309
- Luke, S. & Panait, L. "Bounding Bloat in Genetic Programming." https://arxiv.org/pdf/1806.02112
- Arxiv 1504.08168: "Model Selection and Overfitting in Genetic Programming: Empirical Study." https://arxiv.org/abs/1504.08168
- Springer: "Applications of genetic programming to finance and economics." https://link.springer.com/article/10.1007/s10710-019-09359-z
- Coinbase Institutional Research: survivorship bias inflates crypto backtests 17-22% annually. Via: https://midlandsinbusiness.com/backtesting-crypto-strategies-data-sources-bias-pitfalls-and-validation
- Bailey et al. Probability of Backtest Overfitting framework. Via: https://portfoliooptimizationbook.com/book/8.3-dangers-backtesting.html
- AlgoXpert Alpha Research Framework (IS/WFA/OOS Protocol). https://arxiv.org/pdf/2603.09219
- IEEE: "A Study of GP's Division Operators for Symbolic Regression." https://ieeexplore.ieee.org/document/4724988/
- xCDAT Issue #775: Numba needs NumPy 2.2 or less, Got NumPy 2.3. https://github.com/xCDAT/xcdat/issues/775
