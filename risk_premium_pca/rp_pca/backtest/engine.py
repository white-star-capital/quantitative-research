"""
Walk-forward out-of-sample backtest engine.

The article's OOS procedure:
    1. Estimate covariance with an EWMA or rolling window of length `cov_window`.
    2. Estimate means with a shorter EWMA or rolling window of length `mean_window`.
    3. Fit RP-PCA (or PCA) on the training window.
    4. Construct tangency and min-variance portfolios from the top-K factors.
    5. Hold the portfolio for `rebalance_days` days (walk forward one step).
    6. Repeat until the end of the sample.

Key insight from the article: separating cov/mean estimation windows with
EWMA allows the mean estimate to react faster (shorter half-life) while the
covariance estimate stays stable (longer half-life).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..models.rp_pca import RPPCA
from ..models.covariance import (
    sample_cov, ewma_cov,
    sample_mean, ewma_mean,
    ledoit_wolf_cov,
)
from ..portfolio.construction import PortfolioConstructor
from ..portfolio.metrics import compute_metrics_table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResults:
    """Container for all walk-forward backtest outputs."""

    # Per-day return series for each strategy
    return_series: dict[str, pd.Series] = field(default_factory=dict)

    # Rebalance dates and associated portfolio weights
    rebalance_log: list[dict] = field(default_factory=list)

    # Metrics table (computed on demand)
    _metrics: Optional[pd.DataFrame] = field(default=None, repr=False)

    def metrics(
        self,
        risk_free_rate: float = 0.05,
        trading_days: int = 252,
    ) -> pd.DataFrame:
        if self._metrics is None:
            self._metrics = compute_metrics_table(
                {k: v.values for k, v in self.return_series.items()},
                risk_free_rate=risk_free_rate,
                trading_days=trading_days,
            )
        return self._metrics

    def cumulative(self) -> pd.DataFrame:
        """Wealth index (normalised to 1 at start) for all strategies."""
        frames = {}
        for name, r in self.return_series.items():
            frames[name] = np.exp(r.cumsum())
        return pd.DataFrame(frames)

    def returns_df(self) -> pd.DataFrame:
        """Daily return series as a single DataFrame (date index, one column per strategy)."""
        return pd.DataFrame(self.return_series)

    def rebalance_log_df(self) -> pd.DataFrame:
        """Flatten rebalance_log to a DataFrame for CSV export."""
        rows = []
        for entry in self.rebalance_log:
            row = {
                "rebalance_date": entry["rebalance_date"],
                "n_hold_days": entry["n_hold_days"],
                "gamma_used": entry["gamma_used"],
            }
            for i, w in enumerate(entry.get("rp_pca_tan_weights", [])):
                row[f"tan_w{i}"] = w
            for i, w in enumerate(entry.get("rp_pca_mv_weights", [])):
                row[f"mv_w{i}"] = w
            rows.append(row)
        return pd.DataFrame(rows)

    def signals_df(self, strategy: str) -> pd.DataFrame:
        """
        Return a signals DataFrame: one row per rebalance date, one column per asset.

        Values are the **target asset-space portfolio weights** at each rebalance,
        i.e.  w_asset = L @ w_factor  (N,).  Downstream code can use this as the
        target portfolio for execution; trades = w_asset(t) − w_asset(t-1).

        Parameters
        ----------
        strategy : str
            One of "RP-PCA Tangency", "RP-PCA Min-Var", "PCA Tangency", "PCA Min-Var".

        Returns
        -------
        pd.DataFrame  shape (rebalances, 1 + N): [rebalance_date, asset1, asset2, …]
        Empty DataFrame if asset weights were not recorded.
        """
        key_map = {
            "RP-PCA Tangency": "asset_weights_rp_tan",
            "RP-PCA Min-Var":  "asset_weights_rp_mv",
            "PCA Tangency":    "asset_weights_pca_tan",
            "PCA Min-Var":     "asset_weights_pca_mv",
            "alpha_concentrated_RP-PCA-Tangency": "asset_weights_alpha_conc_rp_tan",
            "alpha_concentrated_RP-PCA-Min-Var":  "asset_weights_alpha_conc_rp_mv",
            "alpha_concentrated_PCA-Tangency":    "asset_weights_alpha_conc_pca_tan",
            "alpha_concentrated_PCA-Min-Var":     "asset_weights_alpha_conc_pca_mv",
        }
        if strategy not in key_map:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {list(key_map.keys())}"
            )
        weight_key = key_map[strategy]

        if not self.rebalance_log:
            return pd.DataFrame()

        first = self.rebalance_log[0]
        if weight_key not in first:
            return pd.DataFrame()

        asset_names: list[str] = first.get("asset_names", [
            f"asset_{i}" for i in range(len(first[weight_key]))
        ])

        rows = []
        for entry in self.rebalance_log:
            row = {"rebalance_date": entry["rebalance_date"]}
            weights = entry.get(weight_key, [])
            for name, w in zip(asset_names, weights):
                row[name] = w
            rows.append(row)

        return pd.DataFrame(rows)

    def save_to_csv(
        self,
        output_dir: Path,
        prefix: str = "backtest",
        risk_free_rate: float = 0.05,
        trading_days: int = 252,
    ) -> list[Path]:
        """
        Save all backtest outputs to CSV for further analysis.

        Writes: returns, cumulative returns, metrics, rebalance log.
        Returns the list of written file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        returns_path = output_dir / f"{prefix}_returns.csv"
        self.returns_df().to_csv(returns_path)
        written.append(returns_path)

        cumulative_path = output_dir / f"{prefix}_cumulative.csv"
        self.cumulative().to_csv(cumulative_path)
        written.append(cumulative_path)

        metrics_path = output_dir / f"{prefix}_metrics.csv"
        self.metrics(risk_free_rate=risk_free_rate, trading_days=trading_days).to_csv(
            metrics_path
        )
        written.append(metrics_path)

        if self.rebalance_log:
            rebalance_path = output_dir / f"{prefix}_rebalance_log.csv"
            self.rebalance_log_df().to_csv(rebalance_path, index=False)
            written.append(rebalance_path)

        return written


# ---------------------------------------------------------------------------
# Main backtest engine
# ---------------------------------------------------------------------------

class WalkForwardBacktest:
    """
    Walk-forward out-of-sample backtest for PCA / RP-PCA portfolios.

    Parameters
    ----------
    returns : DataFrame of shape (T, N)
        Full return matrix (dates × assets).
    cov_window : int
        Rolling window length for covariance estimation (trading days).
    mean_window : int
        Rolling window length for mean estimation (trading days).
    rebalance_days : int
        Number of days between portfolio rebalances (walk-forward step).
    n_components : int
        Number of PCA / RP-PCA components.
    gamma : float or None
        RP-PCA penalty.  None → auto (T of training window).
    use_ewma : bool
        If True, use EWMA estimators for cov and mean.
    ewma_cov_halflife : int
        EWMA half-life for covariance (used when use_ewma=True).
    ewma_mean_halflife : int
        EWMA half-life for mean (used when use_ewma=True).
    cov_method : str
        "sample" | "ewma" | "ledoit_wolf"
    risk_free_rate : float
        Annualised risk-free rate.
    target_gross_leverage : float
        Target gross exposure as a fraction of NAV (default 1.0 = 1x, i.e.
        the sum of absolute asset weights equals 1).  The factor-space
        weights are scaled by ``target / sum(|w_asset|)`` at every rebalance
        so that both the stored asset weights and the holding-period returns
        are consistent with the chosen leverage level.
    concentrated_top_n : int or None
        If set, compute alpha_concentrated_* strategies: keep only top N assets
        by |weight| per rebalance, zero the rest, renormalise. None or 0 = skip.
    buy_hold_benchmark : str or None
        If set and the column exists in ``returns``, track a buy-and-hold of that
        asset (series key equals the column name, e.g. ``\"BTC\"``). If the column
        is missing, the series is omitted (with a warning), **except** for
        ``buy_hold_benchmark=\"TAO\"`` when ``\"TAO\"`` is not a column: then a
        **synthetic** flat benchmark is used (zero log returns), appropriate for
        TAO-denominated subnet universes where TAO is the numeraire. None = no buy-and-hold.
    vw_benchmark_mcaps : DataFrame or None
        If provided, compute a market-cap-weighted benchmark (``"Value-Weight"``).
        Must have the same DatetimeIndex and column names as ``returns``.  Market
        caps are lagged by one day to avoid look-ahead bias.
    selected_factors : list of int or None
        If set, only use these factor indices (0-based) for portfolio construction
        at each rebalance.  For example, ``[0, 2, 4]`` uses PC1, PC3, PC5 only.
        The model still fits all ``n_components`` factors but the portfolio
        optimiser sees only the selected subset.  None = use all factors (default).
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        cov_window: int = 252,
        mean_window: int = 63,
        rebalance_days: int = 21,
        n_components: int = 5,
        gamma: Optional[float] = None,
        use_ewma: bool = True,
        ewma_cov_halflife: int = 60,
        ewma_mean_halflife: int = 21,
        cov_method: str = "ewma",
        risk_free_rate: float = 0.05,
        trading_days: int = 252,
        target_gross_leverage: float = 1.0,
        concentrated_top_n: Optional[int] = None,
        buy_hold_benchmark: Optional[str] = "BTC",
        vw_benchmark_mcaps: Optional[pd.DataFrame] = None,
        selected_factors: Optional[list[int]] = None,
    ) -> None:
        self.returns = returns
        self.cov_window = cov_window
        self.mean_window = mean_window
        self.rebalance_days = rebalance_days
        self.n_components = n_components
        self.gamma = gamma
        self.use_ewma = use_ewma
        self.ewma_cov_halflife = ewma_cov_halflife
        self.ewma_mean_halflife = ewma_mean_halflife
        self.cov_method = cov_method
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.target_gross_leverage = target_gross_leverage
        self.concentrated_top_n = concentrated_top_n
        self.buy_hold_benchmark = buy_hold_benchmark
        self.vw_benchmark_mcaps = vw_benchmark_mcaps
        self.selected_factors = selected_factors

        self._dates = returns.index
        self._T = len(returns)
        self._X = returns.values  # (T, N) numpy array

    def _synthetic_tao_buy_hold(self) -> bool:
        """Flat TAO numeraire: zero log returns when there is no TAO price column."""
        b = self.buy_hold_benchmark
        return b == "TAO" and "TAO" not in self.returns.columns

    # ------------------------------------------------------------------
    # Run the backtest
    # ------------------------------------------------------------------

    def run(
        self,
        include_pca: bool = True,
        include_benchmarks: bool = True,
    ) -> BacktestResults:
        """
        Execute the walk-forward backtest.

        Parameters
        ----------
        include_pca : bool
            Also run standard uncentered PCA (γ=1) for comparison.
        include_benchmarks : bool
            Also compute equal-weighted and optional buy-and-hold benchmark
            (see ``buy_hold_benchmark``).

        Returns
        -------
        BacktestResults
        """
        if self._T < 3:
            logger.warning("Not enough rows for backtest (need >=3, got %d)", self._T)
            return BacktestResults(return_series={}, rebalance_log=[])

        # Adaptive windows: when the sample is short, use as much history as exists
        # rather than requiring the full configured window length.
        max_train_len = max(2, self._T - 2)
        eff_cov_window = min(self.cov_window, max_train_len)
        eff_mean_window = min(self.mean_window, max_train_len)
        start_idx = max(eff_cov_window, eff_mean_window)
        if (eff_cov_window, eff_mean_window) != (self.cov_window, self.mean_window):
            logger.info(
                "Adaptive windows enabled: cov %d→%d, mean %d→%d (T=%d)",
                self.cov_window,
                eff_cov_window,
                self.mean_window,
                eff_mean_window,
                self._T,
            )

        bench = self.buy_hold_benchmark
        if (
            include_benchmarks
            and bench is not None
            and bench not in self.returns.columns
            and not self._synthetic_tao_buy_hold()
        ):
            logger.warning(
                "buy_hold_benchmark=%r not in return columns; omitting buy-and-hold series",
                bench,
            )

        # Pre-build rebalance schedule
        rebalance_indices = list(
            range(start_idx, self._T - 1, self.rebalance_days)
        )

        # Accumulate daily returns per strategy
        acc: dict[str, list[float]] = {
            "RP-PCA Tangency": [],
            "RP-PCA Min-Var": [],
        }
        if include_pca:
            acc["PCA Tangency"] = []
            acc["PCA Min-Var"] = []
        if self.concentrated_top_n and self.concentrated_top_n > 0:
            acc["alpha_concentrated_RP-PCA-Tangency"] = []
            acc["alpha_concentrated_RP-PCA-Min-Var"] = []
            if include_pca:
                acc["alpha_concentrated_PCA-Tangency"] = []
                acc["alpha_concentrated_PCA-Min-Var"] = []
        if include_benchmarks:
            acc["Equal-Weight"] = []
        vw_active = include_benchmarks and self.vw_benchmark_mcaps is not None
        if vw_active:
            acc["Value-Weight"] = []
            # Pre-align market caps to return columns and index
            _mcap_cols = self.returns.columns.intersection(
                self.vw_benchmark_mcaps.columns
            )
            self._vw_mcap = self.vw_benchmark_mcaps[_mcap_cols].reindex(
                self.returns.index
            ).ffill().values  # (T, N_common)
            self._vw_col_idx = [
                self.returns.columns.get_loc(c) for c in _mcap_cols
            ]
        bench_active = (
            include_benchmarks
            and self.buy_hold_benchmark is not None
            and (
                self.buy_hold_benchmark in self.returns.columns
                or self._synthetic_tao_buy_hold()
            )
        )
        if bench_active:
            acc[self.buy_hold_benchmark] = []

        date_acc: list[pd.Timestamp] = []
        rebalance_log: list[dict] = []

        for i, t in enumerate(
            tqdm(rebalance_indices, desc="Walk-forward backtest")
        ):
            # Determine end of holding period (next rebalance or end of sample)
            next_t = (
                rebalance_indices[i + 1]
                if i + 1 < len(rebalance_indices)
                else self._T
            )
            hold_slice = slice(t, next_t)

            # ── Cov window slice ──────────────────────────────────────
            cov_slice = self._X[max(0, t - eff_cov_window): t]
            mean_slice = self._X[max(0, t - eff_mean_window): t]

            # ── Estimate moments ──────────────────────────────────────
            Sigma, mu = self._estimate_moments(cov_slice, mean_slice)

            # ── Fit RP-PCA ────────────────────────────────────────────
            # We pass the full cov window data for the factor return projection
            rp = RPPCA(n_components=self.n_components, gamma=self.gamma)
            rp.fit(cov_slice, cov_matrix=Sigma, mean_vector=mu)

            # ── Construct portfolios (fit on train, predict on hold) ───
            # Optionally subset to selected factors
            rp_L = rp.loadings_
            if self.selected_factors is not None:
                rp_L = rp_L[:, self.selected_factors]

            train_F = cov_slice @ rp_L
            hold_X = self._X[hold_slice]
            hold_F = hold_X @ rp_L

            pc = PortfolioConstructor(
                loadings=rp_L,
                factor_returns=train_F,
                risk_free_rate=self.risk_free_rate,
                trading_days=self.trading_days,
            )

            # RP-PCA tangency weights (fitted on train)
            w_tan = pc.tangency_weights()
            w_mv = pc.min_var_weights()

            # Asset-space weights  w_asset = L @ w  (N,)
            # Scale to target_gross_leverage so that sum(|w_asset|) == target.
            # The same scale is applied to the holding-period returns, keeping
            # w_asset and returns perfectly consistent.
            asset_names = self.returns.columns.tolist()
            w_asset_rp_tan = pc.to_asset_weights(w_tan)
            w_asset_rp_mv  = pc.to_asset_weights(w_mv)

            gross_tan = float(np.abs(w_asset_rp_tan).sum())
            gross_mv  = float(np.abs(w_asset_rp_mv).sum())
            scale_tan = self.target_gross_leverage / gross_tan if gross_tan > 1e-10 else 1.0
            scale_mv  = self.target_gross_leverage / gross_mv  if gross_mv  > 1e-10 else 1.0

            w_asset_rp_tan = w_asset_rp_tan * scale_tan
            w_asset_rp_mv  = w_asset_rp_mv  * scale_mv

            # Append holding-period returns (scaled identically to asset weights)
            rp_tan_ret = hold_F @ w_tan * scale_tan
            rp_mv_ret  = hold_F @ w_mv  * scale_mv
            acc["RP-PCA Tangency"].extend(rp_tan_ret.tolist())
            acc["RP-PCA Min-Var"].extend(rp_mv_ret.tolist())

            # ── Standard PCA (γ = 1) comparison ──────────────────────
            log_pca_tan: list[float] = []
            log_pca_mv: list[float] = []
            if include_pca:
                pca = RPPCA(n_components=self.n_components, gamma=1.0)
                pca.fit(cov_slice, cov_matrix=Sigma, mean_vector=mu)
                pca_L = pca.loadings_
                if self.selected_factors is not None:
                    pca_L = pca_L[:, self.selected_factors]
                train_Fp = cov_slice @ pca_L
                hold_Fp = hold_X @ pca_L
                pcp = PortfolioConstructor(
                    loadings=pca_L,
                    factor_returns=train_Fp,
                    risk_free_rate=self.risk_free_rate,
                    trading_days=self.trading_days,
                )
                pca_w_tan = pcp.tangency_weights()
                pca_w_mv  = pcp.min_var_weights()

                pca_asset_tan = pcp.to_asset_weights(pca_w_tan)
                pca_asset_mv  = pcp.to_asset_weights(pca_w_mv)

                gross_pca_tan = float(np.abs(pca_asset_tan).sum())
                gross_pca_mv  = float(np.abs(pca_asset_mv).sum())
                scale_pca_tan = self.target_gross_leverage / gross_pca_tan if gross_pca_tan > 1e-10 else 1.0
                scale_pca_mv  = self.target_gross_leverage / gross_pca_mv  if gross_pca_mv  > 1e-10 else 1.0

                pca_asset_tan = pca_asset_tan * scale_pca_tan
                pca_asset_mv  = pca_asset_mv  * scale_pca_mv

                acc["PCA Tangency"].extend((hold_Fp @ pca_w_tan * scale_pca_tan).tolist())
                acc["PCA Min-Var"].extend((hold_Fp @ pca_w_mv  * scale_pca_mv).tolist())
                log_pca_tan = pca_asset_tan.tolist()
                log_pca_mv  = pca_asset_mv.tolist()

            # ── Alpha concentrated (top N assets by |weight|) ───────────
            log_alpha_conc: dict[str, list[float]] = {}
            if self.concentrated_top_n and self.concentrated_top_n > 0:
                w_conc_rp_tan = self._concentrate_weights(
                    np.asarray(w_asset_rp_tan), self.concentrated_top_n
                )
                w_conc_rp_mv = self._concentrate_weights(
                    np.asarray(w_asset_rp_mv), self.concentrated_top_n
                )
                acc["alpha_concentrated_RP-PCA-Tangency"].extend(
                    (hold_X @ w_conc_rp_tan).tolist()
                )
                acc["alpha_concentrated_RP-PCA-Min-Var"].extend(
                    (hold_X @ w_conc_rp_mv).tolist()
                )
                log_alpha_conc["asset_weights_alpha_conc_rp_tan"] = w_conc_rp_tan.tolist()
                log_alpha_conc["asset_weights_alpha_conc_rp_mv"] = w_conc_rp_mv.tolist()
                if include_pca:
                    w_conc_pca_tan = self._concentrate_weights(
                        np.asarray(pca_asset_tan), self.concentrated_top_n
                    )
                    w_conc_pca_mv = self._concentrate_weights(
                        np.asarray(pca_asset_mv), self.concentrated_top_n
                    )
                    acc["alpha_concentrated_PCA-Tangency"].extend(
                        (hold_X @ w_conc_pca_tan).tolist()
                    )
                    acc["alpha_concentrated_PCA-Min-Var"].extend(
                        (hold_X @ w_conc_pca_mv).tolist()
                    )
                    log_alpha_conc["asset_weights_alpha_conc_pca_tan"] = w_conc_pca_tan.tolist()
                    log_alpha_conc["asset_weights_alpha_conc_pca_mv"] = w_conc_pca_mv.tolist()

            # ── Benchmarks ────────────────────────────────────────────
            if include_benchmarks:
                ew_ret = hold_X.mean(axis=1)
                acc["Equal-Weight"].extend(ew_ret.tolist())

            if vw_active:
                # Value-weighted returns using lagged market caps
                mcap_hold = self._vw_mcap[hold_slice]  # (n_hold, N_common)
                hold_X_common = self._X[hold_slice][:, self._vw_col_idx]
                # Lag mcap by 1 day; for the first day of each hold period,
                # use the market cap from the day before hold_slice starts.
                if t > 0:
                    prev_mcap = self._vw_mcap[t - 1: t]  # (1, N_common)
                    lagged_mcap = np.vstack([prev_mcap, mcap_hold[:-1]])
                else:
                    lagged_mcap = mcap_hold  # no prior data available
                row_sums = np.nansum(lagged_mcap, axis=1, keepdims=True)
                row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
                weights = np.nan_to_num(lagged_mcap / row_sums)
                vw_ret = (hold_X_common * weights).sum(axis=1)
                acc["Value-Weight"].extend(vw_ret.tolist())

            if bench_active:
                bcol = self.buy_hold_benchmark
                assert bcol is not None
                n_hold = next_t - t
                if bcol in self.returns.columns:
                    bh_idx = self.returns.columns.get_loc(bcol)
                    acc[bcol].extend(hold_X[:, bh_idx].tolist())
                else:
                    acc[bcol].extend([0.0] * n_hold)

            # ── Log ───────────────────────────────────────────────────
            date_acc.extend(self._dates[hold_slice].tolist())
            log_entry: dict = {
                "rebalance_date":       self._dates[t],
                "n_hold_days":          next_t - t,
                "rp_pca_tan_weights":   w_tan.tolist(),
                "rp_pca_mv_weights":    w_mv.tolist(),
                "gamma_used":           rp.gamma_used_,
                "asset_names":          asset_names,
                "asset_weights_rp_tan": w_asset_rp_tan.tolist(),
                "asset_weights_rp_mv":  w_asset_rp_mv.tolist(),
            }
            if include_pca:
                log_entry["asset_weights_pca_tan"] = log_pca_tan
                log_entry["asset_weights_pca_mv"]  = log_pca_mv
            log_entry.update(log_alpha_conc)
            rebalance_log.append(log_entry)

        # ── Build return series keyed by date ─────────────────────────
        # Trim all to the same length (in case of rounding)
        n = min(len(v) for v in acc.values())
        dates_trim = date_acc[:n]

        return_series = {
            name: pd.Series(vals[:n], index=dates_trim, name=name)
            for name, vals in acc.items()
        }

        return BacktestResults(
            return_series=return_series,
            rebalance_log=rebalance_log,
        )

    # ------------------------------------------------------------------
    # Grid search over estimation windows
    # ------------------------------------------------------------------

    def grid_search(
        self,
        cov_windows: list[int],
        mean_windows: list[int],
        metric: str = "Ann. Sharpe",
    ) -> pd.DataFrame:
        """
        Evaluate all (cov_window, mean_window) combinations OOS.

        Returns a pivot table with cov_window as rows, mean_window as cols.
        """
        records = []
        total = len(cov_windows) * len(mean_windows)
        logger.info("Grid search: %d combinations", total)

        for cw in cov_windows:
            for mw in mean_windows:
                if mw > cw:
                    continue  # mean window can't exceed cov window
                self.cov_window = cw
                self.mean_window = mw
                results = self.run(include_pca=False, include_benchmarks=False)
                m = results.metrics(self.risk_free_rate, self.trading_days)
                records.append({
                    "cov_window": cw,
                    "mean_window": mw,
                    "RP-PCA Tangency Sharpe": m.loc["RP-PCA Tangency", metric]
                    if "RP-PCA Tangency" in m.index else np.nan,
                    "RP-PCA Min-Var Sharpe": m.loc["RP-PCA Min-Var", metric]
                    if "RP-PCA Min-Var" in m.index else np.nan,
                })

        df = pd.DataFrame(records)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_moments(
        self, cov_data: np.ndarray, mean_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate Σ and μ according to the configured method."""
        if self.cov_method == "ewma":
            Sigma = ewma_cov(cov_data, halflife=self.ewma_cov_halflife)
            mu = ewma_mean(mean_data, halflife=self.ewma_mean_halflife)
        elif self.cov_method == "ledoit_wolf":
            Sigma = ledoit_wolf_cov(cov_data)
            mu = sample_mean(mean_data)
        else:  # "sample"
            Sigma = sample_cov(cov_data)
            mu = sample_mean(mean_data)
        return Sigma, mu

    def _concentrate_weights(self, w_asset: np.ndarray, top_n: int) -> np.ndarray:
        """
        Keep only top_n assets by |weight|, zero the rest, renormalise to target_gross_leverage.
        """
        if top_n <= 0 or w_asset.size == 0:
            return w_asset.copy()
        k = min(top_n, w_asset.size)
        idx = np.argsort(np.abs(w_asset))[::-1][:k]
        w_out = np.zeros_like(w_asset)
        w_out[idx] = w_asset[idx]
        gross = float(np.abs(w_out).sum())
        if gross > 1e-10:
            w_out = w_out * (self.target_gross_leverage / gross)
        return w_out
