"""
Visualization tests — VAL-05, VAL-06, VAL-07.

Tests verify that plot functions create non-empty PNG files.
Visual correctness is a human review item (checkpoint in Plan 03).

All tests use synthetic data — no parquet files, no actual evolution.
matplotlib.use('Agg') must be called before any pyplot import.
"""
from __future__ import annotations

import os
import tempfile

import matplotlib
matplotlib.use('Agg')  # headless safety — must precede any pyplot import

import numpy as np
import pandas as pd
import pytest
from deap import base, gp, tools
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_T = 100    # small T for fast test runtime
_F = 12
_A = 2

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_dir():
    """Temporary directory for PNG output; cleaned up after module tests."""
    d = tempfile.mkdtemp(prefix="vgp_test_plots_")
    yield d
    # No cleanup — let OS handle tmp on CI


@pytest.fixture(scope="module")
def real_individual():
    """A real DEAP creator.Individual for tree visualization tests."""
    from vgp.gp.gp_types import build_pset, creator
    pset = build_pset()
    tb = base.Toolbox()
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
    ind = tb.individual()
    ind.fitness.values = (0.5, 0.1, -float(len(ind)))
    return ind


@pytest.fixture(scope="module")
def synthetic_feature_matrix():
    """Random [T x F x A] float32."""
    rng = np.random.default_rng(99)
    return rng.standard_normal((_T, _F, _A)).astype(np.float32)


@pytest.fixture(scope="module")
def synthetic_eval_config(synthetic_feature_matrix):
    """EvalConfig with synthetic close prices covering _T timesteps."""
    from vgp.backtest.runner import EvalConfig
    dates = pd.date_range("2024-01-01", periods=_T, freq="D")
    close = pd.DataFrame(
        np.ones((_T, _A)) * 100.0,
        index=dates,
        columns=[f"asset_{a}" for a in range(_A)],
    )
    return EvalConfig(
        fee_bps=10.0,
        min_trades=1,  # low threshold so synthetic signals pass
        freq="1D",
        init_cash=10_000.0,
        close_prices=close,
    )

# ---------------------------------------------------------------------------
# VAL-05: Pareto front plot
# ---------------------------------------------------------------------------

def _make_synthetic_hof(n: int = 5) -> list:
    """List of mock individuals with .fitness.values for plot_pareto_front."""
    rng = np.random.default_rng(0)
    hof = []
    for _ in range(n):
        ind = MagicMock()
        ind.fitness.values = (
            float(rng.uniform(-0.5, 2.0)),   # sharpe
            float(rng.uniform(-0.2, 1.5)),   # total_return
            float(-rng.integers(5, 30)),      # -tree_size (stored negative)
        )
        hof.append(ind)
    return hof


def test_plot_pareto_front_creates_png(tmp_dir):
    """VAL-05: plot_pareto_front() creates a non-empty PNG file."""
    from vgp.analysis.plots import plot_pareto_front
    out = os.path.join(tmp_dir, "pareto_front.png")
    hof = _make_synthetic_hof(5)
    plot_pareto_front(hof, output_path=out)
    assert os.path.exists(out), "PNG file not created"
    assert os.path.getsize(out) > 1000, f"PNG file too small: {os.path.getsize(out)} bytes"


def test_plot_pareto_front_empty_hof_raises(tmp_dir):
    """VAL-05: empty hof raises ValueError."""
    from vgp.analysis.plots import plot_pareto_front
    out = os.path.join(tmp_dir, "pareto_empty.png")
    with pytest.raises(ValueError, match="empty"):
        plot_pareto_front([], output_path=out)

# ---------------------------------------------------------------------------
# VAL-06: Equity curve plot
# ---------------------------------------------------------------------------

def test_plot_equity_curves_creates_png(tmp_dir, real_individual, synthetic_feature_matrix, synthetic_eval_config):
    """VAL-06: plot_equity_curves() creates a non-empty PNG file with IS/OOS boundary."""
    from vgp.analysis.plots import plot_equity_curves
    out = os.path.join(tmp_dir, "equity_curves.png")
    # train_end_date within the _T=100-day range (day 60 of 100)
    train_end_date = "2024-03-01"
    plot_equity_curves(
        individuals=[real_individual],
        feature_matrix=synthetic_feature_matrix,
        eval_config=synthetic_eval_config,
        train_end_date=train_end_date,
        output_path=out,
    )
    assert os.path.exists(out), "PNG file not created"
    assert os.path.getsize(out) > 1000, f"PNG file too small: {os.path.getsize(out)} bytes"


def test_plot_equity_curves_empty_individuals(tmp_dir, synthetic_feature_matrix, synthetic_eval_config):
    """VAL-06: empty individuals list logs warning and returns without error."""
    from vgp.analysis.plots import plot_equity_curves
    out = os.path.join(tmp_dir, "equity_empty.png")
    # Should not raise — just logs warning
    plot_equity_curves(
        individuals=[],
        feature_matrix=synthetic_feature_matrix,
        eval_config=synthetic_eval_config,
        train_end_date="2024-03-01",
        output_path=out,
    )
    # File should NOT be created (function returns early)
    assert not os.path.exists(out)

# ---------------------------------------------------------------------------
# VAL-07: GP tree graph export
# ---------------------------------------------------------------------------

def test_plot_tree_graph_creates_png(tmp_dir, real_individual):
    """VAL-07: plot_tree_graph() creates a non-empty PNG file with readable node labels."""
    from vgp.analysis.plots import plot_tree_graph
    out = os.path.join(tmp_dir, "tree_graph.png")
    plot_tree_graph(real_individual, output_path=out, title="Test Tree")
    assert os.path.exists(out), "PNG file not created"
    assert os.path.getsize(out) > 1000, f"PNG file too small: {os.path.getsize(out)} bytes"
