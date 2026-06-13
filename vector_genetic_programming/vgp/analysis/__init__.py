"""Analysis: walk-forward runner, DSR reporting, visualizations."""
from vgp.analysis.dsr import aggregate_seeds, compute_dsr, save_results_csv
from vgp.analysis.runner import WalkForwardRunner, WindowSpec, generate_windows
from vgp.analysis.plots import plot_equity_curves, plot_pareto_front, plot_tree_graph

__all__ = [
    "WalkForwardRunner",
    "WindowSpec",
    "generate_windows",
    "compute_dsr",
    "aggregate_seeds",
    "save_results_csv",
    "plot_pareto_front",
    "plot_equity_curves",
    "plot_tree_graph",
]
