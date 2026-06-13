"""Analysis: walk-forward runner, DSR reporting, visualizations."""
from vgp.analysis.dsr import aggregate_seeds, compute_dsr, save_results_csv
from vgp.analysis.runner import WalkForwardRunner, WindowSpec, generate_windows

__all__ = [
    "WalkForwardRunner",
    "WindowSpec",
    "generate_windows",
    "compute_dsr",
    "aggregate_seeds",
    "save_results_csv",
]
