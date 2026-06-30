"""NSGA-II evolution loop — DEAP toolbox, varOr-based generation loop.

ARCHITECTURE INVARIANT (D-15 / CLAUDE.md Architecture Invariants):
  This module must NOT import vectorbt at module level or inside any function.
  The interface is: numpy feature_matrix in -> (population, hof, logbook) out.
  Backtest evaluation is accessed only via functools.partial(evaluate, ...).

  _jit_warmup() is the ONLY function that imports vectorbt. It runs exclusively
  inside spawn worker processes via Pool(initializer=_jit_warmup). It must never
  be called from the main process.

IMPORTANT: eaMuPlusLambda is NOT used directly because it has no per-generation
callback hook required for checkpointing. The loop replicates eaMuPlusLambda's
body using algorithms.varOr() — identical logic, full checkpoint control.
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import multiprocessing
import operator
import random
from datetime import datetime

import numpy as np
from deap import algorithms, base, gp, tools
from tqdm import tqdm

from vgp.backtest.runner import EvalConfig, evaluate
from vgp.evolution.checkpoint import load_checkpoint, save_checkpoint
from vgp.evolution.config import EvolutionConfig
from vgp.evolution.tracker import NoOpTracker
from vgp.gp.gp_types import build_pset, creator  # noqa: F401 — side effect: registers creator.Individual

logger = logging.getLogger(__name__)

# TREE_HEIGHT_LIMIT is read from EvolutionConfig.tree_height_limit at toolbox build time.
# The module-level constant is NOT used — config is the single source of truth.


# ---------------------------------------------------------------------------
# Module-level JIT warmup function — MUST be module-level for spawn pickling
# CLAUDE.md constraint #8: runs in each spawn worker before any evaluation.
# ---------------------------------------------------------------------------

def _jit_warmup() -> None:
    """Trigger numba JIT compilation in spawn worker before evaluation begins.

    CLAUDE.md constraint #8: warmup MUST run in the Pool initializer, not in
    the main process. Each spawn worker gets a fresh Python interpreter with
    an uncompiled numba JIT cache. Running a minimal Portfolio.from_signals
    call here ensures compilation happens once per worker at startup, not on
    the first real evaluate() call (which would add 30-60s latency).

    vectorbt is imported INSIDE this function body (deferred import).
    It is NEVER imported at module level — D-15 architecture invariant.
    """
    import pandas as pd  # noqa: PLC0415 — intentionally deferred
    import vectorbt as vbt  # noqa: PLC0415 — intentionally deferred; D-15 compliant

    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    close = pd.DataFrame({"a": np.ones(10) * 100.0}, index=dates)
    entries = pd.DataFrame({"a": np.array([True, False] * 5)}, index=dates)
    exits = pd.DataFrame({"a": np.array([False, True] * 5)}, index=dates)
    vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        fees=0.001,
        freq="1D",
        init_cash=1000.0,
        group_by=True,
        cash_sharing=True,
    )


# ---------------------------------------------------------------------------
# Toolbox builder
# ---------------------------------------------------------------------------

def _build_toolbox(
    pset: gp.PrimitiveSetTyped,
    feature_matrix: np.ndarray,
    eval_config: EvalConfig,
    config: EvolutionConfig,
) -> base.Toolbox:
    """Build and return a DEAP Toolbox configured for NSGA-II GP evolution.

    Does NOT wire toolbox.map — that is done in run_evolution() after the
    Pool is created (or skipped for n_jobs=1).

    Parameters
    ----------
    pset : gp.PrimitiveSetTyped
        Primitive set from build_pset() (already includes gt/lt/if_then_else).
    feature_matrix : np.ndarray
        [T x F x A] float32 — captured in the functools.partial for evaluate.
    eval_config : EvalConfig
        Backtest configuration — captured in the functools.partial for evaluate.
    config : EvolutionConfig
        Evolution hyperparameters — tree_height_limit used for staticLimit.
    """
    toolbox = base.Toolbox()

    # Population initialization: ramped half-and-half, max initial depth 4
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation: functools.partial captures feature_matrix and eval_config.
    # DEAP calls toolbox.evaluate(ind) — partial supplies the remaining kwargs (D-04).
    # functools.partial with a numpy array and a dataclass is pickle-safe for spawn.
    toolbox.register(
        "evaluate",
        functools.partial(evaluate, feature_matrix=feature_matrix, config=eval_config),
    )

    # NSGA-II selection (EVO-01)
    toolbox.register("select", tools.selNSGA2)

    # Genetic operators (EVO-01)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

    # Depth limit on BOTH operators (EVO-03, D-16, CLAUDE.md #5).
    # staticLimit replaces oversized offspring with a copy of the parent.
    # Applying to only one operator allows the other to bypass the limit.
    toolbox.decorate(
        "mate",
        gp.staticLimit(
            key=operator.attrgetter("height"),
            max_value=config.tree_height_limit,
        ),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(
            key=operator.attrgetter("height"),
            max_value=config.tree_height_limit,
        ),
    )

    return toolbox


# ---------------------------------------------------------------------------
# Statistics builder
# ---------------------------------------------------------------------------

def _build_stats() -> tools.MultiStatistics:
    """Build MultiStatistics capturing Sharpe and tree size per generation (EVO-06).

    Returns a MultiStatistics with two chapters:
      'fitness': sharpe_max, sharpe_mean, sharpe_min
      'size': size_mean, size_max

    The Logbook record per generation will be:
      {'fitness': {'sharpe_max': float, ...}, 'size': {'size_mean': float, ...}}
    """
    # Fitness stats: extract first element of 3-tuple (Sharpe)
    fit_stats = tools.Statistics(key=operator.attrgetter("fitness.values"))
    fit_stats.register("sharpe_max", lambda vals: float(max(v[0] for v in vals)))
    fit_stats.register("sharpe_mean", lambda vals: float(np.mean([v[0] for v in vals])))
    fit_stats.register("sharpe_min", lambda vals: float(min(v[0] for v in vals)))

    # Tree size stats: len(ind) gives node count
    size_stats = tools.Statistics(key=len)
    size_stats.register("size_mean", lambda vals: float(np.mean(vals)))
    size_stats.register("size_max", lambda vals: float(max(vals)))

    return tools.MultiStatistics(fitness=fit_stats, size=size_stats)


# ---------------------------------------------------------------------------
# Logbook flattening for MLflow
# ---------------------------------------------------------------------------

def _flatten_record(record: dict) -> dict:
    """Flatten a MultiStatistics logbook record to a flat {chapter__key: value} dict.

    MultiStatistics record format: {'fitness': {'sharpe_max': 1.2, ...}, 'size': {...}}
    MLflow log_metrics requires a flat dict with string keys and numeric values.
    """
    flat: dict[str, float] = {}
    for key, val in record.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat[f"{key}__{subkey}"] = float(subval)
        else:
            flat[key] = float(val)
    return flat


# ---------------------------------------------------------------------------
# Main evolution function
# ---------------------------------------------------------------------------

def run_evolution(
    config: EvolutionConfig,
    feature_matrix: np.ndarray,
    eval_config: EvalConfig,
    tracker=None,
    resume_checkpoint: str | None = None,
    desc: str | None = None,
) -> tuple:
    """Run NSGA-II GP evolution and return (population, hof, logbook).

    Parameters
    ----------
    config : EvolutionConfig
        NSGA-II hyperparameters: pop_size, n_generations, cxpb, mutpb, n_jobs, etc.
    feature_matrix : np.ndarray
        Shape [T x F x A], dtype float32. Train-split feature matrix from Phase 2.
        T = timesteps, F = 12 features, A = assets.
    eval_config : EvalConfig
        Backtest configuration. eval_config.close_prices must be set.
    tracker : NoOpTracker | MLflowTracker | None
        Duck-typed experiment tracker (D-03). If None, uses NoOpTracker (no-op).
    resume_checkpoint : str | None
        Path to a checkpoint file to resume from. If None, starts fresh.
    desc : str | None
        Label for the tqdm generation progress bar. Defaults to "seed{seed}".

    Returns
    -------
    tuple[list, tools.ParetoFront, tools.Logbook]
        (final_population, hall_of_fame, logbook)

    Raises
    ------
    ValueError
        If feature_matrix is not 3-D or F != 12.
    """
    if tracker is None:
        tracker = NoOpTracker()

    # Validate inputs early (mirrors runner.py pattern)
    if feature_matrix.ndim != 3:
        raise ValueError(
            f"feature_matrix must be 3-D [T x F x A], got shape {feature_matrix.shape}"
        )
    T, F, A = feature_matrix.shape
    if F != 12:
        raise ValueError(
            f"Expected F=12 feature columns (FEATURE_NAMES), got {F}"
        )

    # Seed all RNGs before anything else (EXP-03 — reproducibility)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Build primitive set and toolbox
    pset = build_pset()
    toolbox = _build_toolbox(pset, feature_matrix, eval_config, config)
    mstats = _build_stats()

    # Hall-of-fame: ParetoFront tracks all non-dominated individuals (D-14, EVO-04)
    hof = tools.ParetoFront()
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + mstats.fields

    # Generate run_id from timestamp + seed
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{config.seed}"

    # Resume or initialize population
    start_gen = 1
    if resume_checkpoint is not None:
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)
        ckpt = load_checkpoint(resume_checkpoint)
        # D-09: restore BOTH RNG states before any operator calls
        random.setstate(ckpt["rng_state"])
        np.random.set_state(ckpt["np_rng_state"])
        population = ckpt["population"]
        hof = ckpt["halloffame"]
        logbook = ckpt["logbook"]
        start_gen = ckpt["generation"] + 1
        logger.info(
            "Resumed from generation %d — continuing from gen %d",
            ckpt["generation"],
            start_gen,
        )
    else:
        population = toolbox.population(n=config.pop_size)

    # Set up parallel evaluation (EVO-07)
    pool = None
    if config.n_jobs > 1:
        # macOS Python 3.12 defaults to spawn; be explicit for cross-platform safety
        ctx = multiprocessing.get_context("spawn")
        # _jit_warmup runs once per worker at Pool creation, compiling numba JIT (CLAUDE.md #8)
        pool = ctx.Pool(processes=config.n_jobs, initializer=_jit_warmup)
        toolbox.register("map", pool.map)
        logger.info(
            "Parallel evaluation: %d workers (spawn context, JIT warmup active)",
            config.n_jobs,
        )
    else:
        # n_jobs=1: single-threaded debugging mode — use built-in map
        toolbox.register("map", map)
        logger.info("Single-threaded evaluation (n_jobs=1)")

    # Log hyperparameters to tracker (EXP-01)
    tracker.start_run(run_name=run_id)
    tracker.log_params(dataclasses.asdict(config))

    try:
        # Evaluate initial population (generation 0)
        if resume_checkpoint is None:
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            hof.update(population)
            record = mstats.compile(population)
            logbook.record(gen=0, nevals=len(invalid_ind), **record)
            tracker.log_metrics(_flatten_record(record), step=0)
            logger.info(
                "Gen 0 | nevals=%d | sharpe_max=%.4f",
                len(invalid_ind),
                record.get("fitness", {}).get("sharpe_max", float("nan")),
            )

        # Main evolution loop — varOr replicates eaMuPlusLambda internals with checkpoint hook
        pbar_desc = desc if desc is not None else f"seed{config.seed}"
        with tqdm(
            range(start_gen, config.n_generations + 1),
            desc=pbar_desc,
            unit="gen",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for gen in pbar:
                # varOr: each offspring is CX OR mutation (not both) of a random parent
                # Clones individuals and deletes fitness.values on modified ones
                offspring = algorithms.varOr(
                    population, toolbox, config.pop_size, config.cxpb, config.mutpb
                )

                # Evaluate only individuals with invalidated fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Update ParetoFront with new offspring (EVO-04)
                hof.update(offspring)

                # NSGA-II selection: mu individuals from combined parent + offspring pool
                population[:] = toolbox.select(population + offspring, config.pop_size)

                # Statistics and logging (EVO-06, EXP-02)
                record = mstats.compile(population)
                logbook.record(gen=gen, nevals=len(invalid_ind), **record)
                tracker.log_metrics(_flatten_record(record), step=gen)

                sharpe_max = record.get("fitness", {}).get("sharpe_max", float("nan"))
                size_mean  = record.get("size", {}).get("size_mean", float("nan"))
                pbar.set_postfix(SR=f"{sharpe_max:+.3f}", nodes=f"{size_mean:.1f}")
                logger.debug(
                    "Gen %d/%d | nevals=%d | sharpe_max=%.4f | size_mean=%.1f",
                    gen, config.n_generations, len(invalid_ind), sharpe_max, size_mean,
                )

                # Checkpoint every checkpoint_freq generations (EVO-05, D-08)
                if gen % config.checkpoint_freq == 0:
                    ckpt_path = f"{config.checkpoint_dir}/{run_id}/gen_{gen:04d}.pkl"
                    save_checkpoint(
                        ckpt_path,
                        population=population,
                        halloffame=hof,
                        logbook=logbook,
                        generation=gen,
                        seed=config.seed,
                    )
                    logger.debug("Checkpoint saved: %s", ckpt_path)

    finally:
        if pool is not None:
            pool.close()
            pool.join()
        tracker.end_run()

    return population, hof, logbook
