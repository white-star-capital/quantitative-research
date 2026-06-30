"""Publication-quality visualizations for VGP results.

Three functions:
  plot_pareto_front  — 3D scatter of HOF fitness values (VAL-05)
  plot_equity_curves — IS + OOS equity curves overlaid (VAL-06)
  plot_tree_graph    — GP tree structure via NetworkX (VAL-07)

matplotlib.use('Agg') is set at module level for headless / CI safety.
graphviz binary is NOT required — nx.bfs_layout() is used for tree layout.
evaluate() does NOT return a Portfolio object; plot_equity_curves calls
vbt.Portfolio.from_signals() directly with the same params as evaluate().
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use('Agg')  # Must be called before any pyplot import — headless safety
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — required for projection='3d'

import networkx as nx
import numpy as np
import pandas as pd

from vgp.backtest.runner import EvalConfig
from vgp.gp.gp_types import build_pset
from vgp.gp.tree_evaluator import TreeEvaluator

logger = logging.getLogger(__name__)


def plot_pareto_front(
    hof,
    output_path: str = "results/pareto_front.png",
) -> None:
    """Export Pareto front scatter plot (Sharpe vs return vs tree size).

    Parameters
    ----------
    hof : tools.ParetoFront
        Hall-of-fame from run_evolution(). Each element is creator.Individual
        with .fitness.values = (sharpe, total_return, -tree_size).
    output_path : str
        Destination PNG file path. Parent directory must exist.

    Raises
    ------
    ValueError
        If hof is empty.
    """
    if not hof:
        raise ValueError("hof is empty — nothing to plot")

    sharpes = [ind.fitness.values[0] for ind in hof]
    returns = [ind.fitness.values[1] for ind in hof]
    sizes = [-ind.fitness.values[2] for ind in hof]  # stored as negative; negate for node count

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(sharpes, returns, sizes, c=sharpes, cmap='viridis', s=60, alpha=0.8)
    ax.set_xlabel('Sharpe Ratio', labelpad=10)
    ax.set_ylabel('Total Return', labelpad=10)
    ax.set_zlabel('Tree Size (nodes)', labelpad=10)
    ax.set_title('NSGA-II Pareto Front — Top Generation', fontsize=13)
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio', shrink=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Pareto front plot saved to %s (%d individuals)", output_path, len(hof))


def plot_equity_curves(
    individuals: list,
    feature_matrix: np.ndarray,
    eval_config: EvalConfig,
    train_end_date: str,
    output_path: str = "results/equity_curves.png",
) -> None:
    """Plot IS + OOS equity curves for top individuals.

    evaluate() discards the Portfolio object, so this function calls
    vbt.Portfolio.from_signals() directly using the same parameters.
    This is intentional duplication for the visualization layer.

    Parameters
    ----------
    individuals : list
        Top individuals from hof (use hof[:min(3, len(hof))]).
    feature_matrix : np.ndarray
        Full data [T x F x A] float32 — covers IS + OOS period for visualization.
    eval_config : EvalConfig
        Must have close_prices set to the FULL data DataFrame (IS + OOS combined).
    train_end_date : str
        ISO date string e.g. "2024-12-31". Vertical line drawn here.
    output_path : str
        Destination PNG file path.
    """
    if not individuals:
        logger.warning("plot_equity_curves: no individuals provided — skipping")
        return

    import vectorbt as vbt  # noqa: PLC0415 — deferred (avoid loading vbt on vgp.analysis import)

    T, F, A = feature_matrix.shape
    pset = build_pset()
    evaluator = TreeEvaluator(pset)
    fee_per_side = (eval_config.fee_bps / 2.0) / 10_000.0

    train_end_ts = pd.Timestamp(train_end_date)

    fig, axes = plt.subplots(len(individuals), 1, figsize=(14, 4 * len(individuals)))
    if len(individuals) == 1:
        axes = [axes]

    for rank, (ind, ax) in enumerate(zip(individuals, axes)):
        # Re-run TreeEvaluator to get signals for the full data range
        signals = np.zeros((T, A), dtype=np.float32)
        for a in range(A):
            signals[:, a] = evaluator.execute(ind, feature_matrix[:, :, a])

        long_entries = signals > 0
        short_entries = signals < 0
        long_exits = signals <= 0
        short_exits = signals >= 0

        # Direct vbt call — same params as evaluate() in runner.py
        pf = vbt.Portfolio.from_signals(
            close=eval_config.close_prices,
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            size=1.0 / A,
            size_type="percent",
            upon_opposite_entry="close",
            fees=fee_per_side,
            freq=eval_config.freq,
            init_cash=eval_config.init_cash,
            group_by=True,
            cash_sharing=True,
        )

        nav = pf.value()       # pd.Series with DatetimeIndex
        ax.plot(nav.index, nav.to_numpy(), linewidth=1.2, label=f'HOF rank {rank}')
        ax.axvline(x=train_end_ts, color='red', linestyle='--', linewidth=1.5,
                   label='IS / OOS boundary')
        ax.axvspan(nav.index[0], train_end_ts, alpha=0.05, color='blue', label='IS period')
        ax.set_ylabel('Portfolio NAV')
        ax.set_title(
            f'Rank {rank} — height={ind.height} nodes={len(ind)} '
            f'SR={ind.fitness.values[0]:.3f}',
            fontsize=10,
        )
        ax.legend(loc='upper left', fontsize=8)

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Equity curve plot saved to %s (%d individuals)", output_path, len(individuals))


def plot_tree_graph(
    individual,
    output_path: str = "results/tree_graph.png",
    title: str = "GP Tree Structure",
) -> None:
    """Export GP tree as a human-readable NetworkX graph image.

    Uses deap.gp.graph() to get nodes/edges/labels; nx.bfs_layout(G, start=0)
    for top-down hierarchical layout. No graphviz binary required.

    Parameters
    ----------
    individual : creator.Individual
        DEAP PrimitiveTree. Terminal names come from pset.renameArguments()
        (e.g. 'ret_1d', 'vol_5d') which was called in build_pset().
    output_path : str
        Destination PNG file path.
    title : str
        Figure title prefix.
    """
    from deap import gp as deap_gp  # noqa: PLC0415 — deferred (avoid loading deap on vgp.analysis import)

    nodes, edges, labels = deap_gp.graph(individual)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos = nx.bfs_layout(G, start=0)

    fig, ax = plt.subplots(figsize=(max(10, len(nodes) * 0.8), 8))
    nx.draw(
        G,
        pos=pos,
        labels=labels,
        ax=ax,
        node_color='lightblue',
        node_size=1800,
        font_size=9,
        font_weight='bold',
        arrows=True,
        arrowsize=15,
        edge_color='gray',
    )
    ax.set_title(
        f'{title} (height={individual.height}, nodes={len(individual)})',
        fontsize=12,
        pad=20,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(
        "Tree graph saved to %s (height=%d, nodes=%d)",
        output_path, individual.height, len(individual),
    )
