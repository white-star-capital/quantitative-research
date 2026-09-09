"""Analysis: walk-forward runner, DSR reporting, visualizations."""
from vgp.analysis.dsr import (
    IS_RETURNS_KEY,
    aggregate_seeds,
    attach_dsr,
    compute_dsr,
    save_results_csv,
)
from vgp.analysis.null_control import (
    NullControlResult,
    best_sharpes,
    block_bootstrap_ohlcv,
    empirical_p_value,
    run_null_control,
)
from vgp.analysis.runner import WalkForwardRunner, WindowSpec, generate_windows
from vgp.analysis.plots import plot_equity_curves, plot_pareto_front, plot_tree_graph

__all__ = [
    "WalkForwardRunner",
    "WindowSpec",
    "generate_windows",
    "compute_dsr",
    "attach_dsr",
    "aggregate_seeds",
    "IS_RETURNS_KEY",
    "save_results_csv",
    "run_null_control",
    "NullControlResult",
    "block_bootstrap_ohlcv",
    "empirical_p_value",
    "best_sharpes",
    "plot_pareto_front",
    "plot_equity_curves",
    "plot_tree_graph",
]
