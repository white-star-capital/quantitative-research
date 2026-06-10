"""Duck-typed experiment tracker — NoOpTracker (default) and MLflowTracker (optional).

CONTEXT.md D-03: The tracking layer must be fully decoupled from the evolution loop.
The evolution loop receives a tracker object and calls tracker.log_params(...),
tracker.log_metrics(...) without knowing which implementation it holds.

CONTEXT.md D-01/D-02: MLflow is an optional extra (mlflow requires pandas<3 which
conflicts with the core pandas>=3.0.0 requirement). When mlflow is NOT installed,
NoOpTracker provides silent no-ops so evolution proceeds without error or warning.
"""
from __future__ import annotations


class NoOpTracker:
    """Silent no-op tracker. Used when mlflow is not installed (D-02).

    All methods are intentional no-ops — do not add logging or side effects here.
    """

    def start_run(self, run_name: str = "") -> None:
        pass

    def log_params(self, params: dict) -> None:
        pass

    def log_metrics(self, metrics: dict, step: int = 0) -> None:
        pass

    def end_run(self) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass


class MLflowTracker:
    """Real MLflow tracker. Only usable when `pip install vgp[tracking]` is installed.

    All mlflow imports are deferred inside __init__ to fail gracefully if not installed.
    If mlflow is not available, use make_tracker(use_mlflow=False) instead.
    """

    def __init__(self, experiment_name: str = "vgp") -> None:
        try:
            import mlflow  # deferred — only available with [tracking] extra
        except ImportError as e:
            raise ImportError(
                "mlflow is not installed. Install with: pip install vgp[tracking]\n"
                "Note: mlflow requires pandas<3, which conflicts with the core vgp "
                "environment. Use a separate tracking environment or NoOpTracker."
            ) from e
        self._mlflow = mlflow
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str = "") -> None:
        self._mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int = 0) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def end_run(self) -> None:
        self._mlflow.end_run()

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)


def make_tracker(use_mlflow: bool = False, experiment_name: str = "vgp") -> NoOpTracker:
    """Factory: return MLflowTracker if use_mlflow=True, else NoOpTracker (D-03).

    When use_mlflow=True and mlflow is not installed, raises ImportError with
    a clear install instruction.
    """
    if use_mlflow:
        return MLflowTracker(experiment_name)
    return NoOpTracker()
