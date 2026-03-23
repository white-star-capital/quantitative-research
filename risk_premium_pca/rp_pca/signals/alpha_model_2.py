"""
Alpha Model 2: Directional long/short signal engine.

Uses RP-PCA as a signal engine (factor momentum, factor reversal, residual momentum,
risk adjustment) to build a composite SCORE and long/short portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import Config
from ..models.rp_pca import RPPCA
from ..models.covariance import (
    sample_cov,
    sample_mean,
    ewma_cov,
    ewma_mean,
    ledoit_wolf_cov,
)
from ..portfolio.metrics import compute_metrics_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _z_score_to_0_100(z: np.ndarray) -> np.ndarray:
    """Map z-scores (over assets) to 0-100 scale. Uses percentile rank."""
    if z.size == 0:
        return z.copy()
    ranks = np.argsort(np.argsort(z))
    out = (ranks / (len(ranks) - 1)) * 100.0 if len(ranks) > 1 else np.full_like(z, 50.0)
    return out.astype(np.float64)


def _estimate_moments(
    cov_data: np.ndarray,
    mean_data: np.ndarray,
    cov_method: str,
    ewma_cov_halflife: int,
    ewma_mean_halflife: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate Sigma and mu using the same logic as WalkForwardBacktest."""
    if cov_method == "ewma":
        Sigma = ewma_cov(cov_data, halflife=ewma_cov_halflife)
        mu = ewma_mean(mean_data, halflife=ewma_mean_halflife)
    elif cov_method == "ledoit_wolf":
        Sigma = ledoit_wolf_cov(cov_data)
        mu = sample_mean(mean_data)
    else:
        Sigma = sample_cov(cov_data)
        mu = sample_mean(mean_data)
    return Sigma, mu


# ---------------------------------------------------------------------------
# Signal components (pure functions given L and data slices)
# ---------------------------------------------------------------------------

def signal_factor_momentum(
    X_lookback: np.ndarray,
    L: np.ndarray,
    lookback_days: int,
) -> np.ndarray:
    """
    Signal A: Pure factor momentum.
    E[R] = L @ lambda_hat, lambda_hat = mean(F)/std(F) (Sharpe-scaled).
    Returns component score 0-100 per asset.
    """
    tau = min(lookback_days, X_lookback.shape[0])
    X_tau = X_lookback[-tau:] if tau < X_lookback.shape[0] else X_lookback
    F = X_tau @ L  # (T, K)
    mean_f = np.mean(F, axis=0)
    std_f = np.std(F, axis=0, ddof=1)
    std_f = np.where(std_f > 1e-10, std_f, 1.0)
    lambda_hat = mean_f / std_f
    E_R = L @ lambda_hat  # (N,)
    z = (E_R - np.mean(E_R)) / (np.std(E_R, ddof=1) + 1e-10)
    return _z_score_to_0_100(z)


def signal_factor_reversal(
    X_lookback: np.ndarray,
    X_short: np.ndarray,
    L: np.ndarray,
    reversal_z_threshold: float,
) -> np.ndarray:
    """
    Signal B: Factor reversal (contrarian on stretched factors).
    Returns component score 0-100 per asset (high = oversold, good for long).
    """
    F_lookback = X_lookback @ L  # (T, K)
    std_f = np.std(F_lookback, axis=0, ddof=1)
    std_f = np.where(std_f > 1e-10, std_f, 1.0)

    if X_short.size == 0:
        return np.full(L.shape[0], 50.0)

    F_short = X_short @ L  # (T_short, K)
    F_recent = np.mean(F_short, axis=0)
    stretched = np.abs(F_recent) > (reversal_z_threshold * std_f)
    if not np.any(stretched):
        return np.full(L.shape[0], 50.0)

    lambda_rev = np.zeros(L.shape[1])
    lambda_rev[stretched] = F_recent[stretched]
    E_R_rev = -L @ lambda_rev
    z = (E_R_rev - np.mean(E_R_rev)) / (np.std(E_R_rev, ddof=1) + 1e-10)
    return _z_score_to_0_100(z)


def signal_residual_momentum(
    X_lookback: np.ndarray,
    L: np.ndarray,
    residual_days: int,
) -> np.ndarray:
    """
    Signal C: Residual momentum (idiosyncratic).
    X_hat = X @ L @ L'; eps = X - X_hat; mom_eps over last residual_days.
    Returns component score 0-100 per asset.
    """
    tau = min(residual_days, X_lookback.shape[0])
    X_tau = X_lookback[-tau:] if tau < X_lookback.shape[0] else X_lookback
    X_hat = X_tau @ L @ L.T
    eps = X_tau - X_hat
    mean_eps = np.mean(eps, axis=0)
    std_eps = np.std(eps, axis=0, ddof=1)
    std_eps = np.where(std_eps > 1e-10, std_eps, 1.0)
    mom_eps = mean_eps / std_eps
    z = (mom_eps - np.mean(mom_eps)) / (np.std(mom_eps, ddof=1) + 1e-10)
    return _z_score_to_0_100(z)


def signal_risk_adjustment(X_lookback: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Risk adjustment: R² from regression of each asset on factors.
    High R² = low idiosyncratic risk = high score (0-100).
    """
    F = X_lookback @ L
    N = X_lookback.shape[1]
    r_sq = np.zeros(N)
    for i in range(N):
        y = X_lookback[:, i]
        X_f = F
        if X_f.size > 0 and np.std(y) > 1e-10:
            beta = np.linalg.lstsq(X_f, y, rcond=None)[0]
            y_hat = X_f @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_sq[i] = 1.0 - (ss_res / (ss_tot + 1e-10))
        else:
            r_sq[i] = 0.0
    r_sq = np.clip(r_sq, 0.0, 1.0)
    return r_sq * 100.0


def composite_score(
    score_a: np.ndarray,
    score_b: np.ndarray,
    score_c: np.ndarray,
    score_risk: np.ndarray,
    w_a: float,
    w_b: float,
    w_c: float,
    w_r: float,
) -> np.ndarray:
    """Weighted composite SCORE, clipped to 0-100."""
    composite = w_a * score_a + w_b * score_b + w_c * score_c + w_r * score_risk
    return np.clip(composite, 0.0, 100.0)


def build_long_short_weights(
    score: np.ndarray,
    long_threshold: float,
    short_threshold: float,
    gross_leverage: float,
    net_cap: float,
    max_long_pct: float,
    max_short_pct: float,
) -> np.ndarray:
    """
    Position sizing: raw_w_i ∝ (SCORE_i - mean(SCORE)).
    Normalize to gross_leverage; apply single-name caps then net exposure cap.
    """
    demeaned = score - np.mean(score)
    denom = np.sum(np.abs(demeaned))
    if denom < 1e-10:
        return np.zeros_like(score)
    w = demeaned / denom * gross_leverage
    w = np.clip(w, -max_short_pct, max_long_pct)
    gross = np.sum(np.abs(w))
    if gross > 1e-10:
        w = w * (gross_leverage / gross)
    net = np.sum(w)
    if abs(net) > net_cap:
        excess = np.sign(net) * (abs(net) - net_cap)
        w = w - excess / len(w)
        w = np.clip(w, -max_short_pct, max_long_pct)
    return w


# ---------------------------------------------------------------------------
# Results and backtest runner
# ---------------------------------------------------------------------------

@dataclass
class AlphaModel2Results:
    """Container for Alpha Model 2 backtest outputs."""

    return_series: pd.Series = field(default_factory=pd.Series)
    rebalance_log: list[dict] = field(default_factory=list)

    def metrics(
        self,
        risk_free_rate: float = 0.05,
        trading_days: int = 252,
    ) -> pd.DataFrame:
        if self.return_series.empty:
            return pd.DataFrame()
        return compute_metrics_table(
            {"Alpha Model 2": self.return_series.values},
            risk_free_rate=risk_free_rate,
            trading_days=trading_days,
        )

    def cumulative(self) -> pd.Series:
        """Wealth index (normalised to 1 at start)."""
        if self.return_series.empty:
            return pd.Series(dtype=float)
        return np.exp(self.return_series.cumsum())


def run_alpha_model_2_backtest(
    returns: pd.DataFrame,
    config: Config,
) -> AlphaModel2Results:
    """
    Walk-forward backtest for Alpha Model 2 using base RP-PCA estimation.

    Reuses cov_window, mean_window, rebalance_days, n_components, gamma,
    cov_method from config. At each rebalance computes signals A/B/C + risk,
    composite SCORE, and long/short weights.
    """
    cfg = config.alpha_model_2
    bc = config.backtest
    mc = config.model

    X = returns.values
    dates = returns.index
    asset_names = returns.columns.tolist()
    T, N = X.shape

    if T < 3:
        return AlphaModel2Results(
            return_series=pd.Series(dtype=float),
            rebalance_log=[],
        )

    max_train_len = max(2, T - 2)
    eff_cov_window = min(bc.cov_window, max_train_len)
    eff_mean_window = min(bc.mean_window, max_train_len)
    start_idx = max(eff_cov_window, eff_mean_window)
    if start_idx >= T - 1:
        return AlphaModel2Results(
            return_series=pd.Series(dtype=float),
            rebalance_log=[],
        )

    rebalance_indices = list(range(start_idx, T - 1, bc.rebalance_days))
    return_acc: list[float] = []
    date_acc: list[pd.Timestamp] = []
    rebalance_log: list[dict] = []

    for i, t in enumerate(tqdm(rebalance_indices, desc="Alpha Model 2 backtest")):
        next_t = (
            rebalance_indices[i + 1]
            if i + 1 < len(rebalance_indices)
            else T
        )
        hold_slice = slice(t, next_t)
        hold_X = X[hold_slice]

        cov_slice = X[max(0, t - eff_cov_window) : t]
        mean_slice = X[max(0, t - eff_mean_window) : t]

        Sigma, mu = _estimate_moments(
            cov_slice,
            mean_slice,
            mc.cov_method,
            mc.ewma_cov_halflife,
            mc.ewma_mean_halflife,
        )

        rp = RPPCA(n_components=mc.n_components, gamma=mc.gamma)
        rp.fit(cov_slice, cov_matrix=Sigma, mean_vector=mu)
        L = rp.loadings_

        lookback_days = min(cfg.factor_premia_lookback, cov_slice.shape[0])
        reversal_days = min(cfg.reversal_short_days, cov_slice.shape[0])
        residual_days = min(cfg.residual_momentum_days, cov_slice.shape[0])

        score_a = signal_factor_momentum(
            cov_slice, L, lookback_days
        )
        X_short = cov_slice[-reversal_days:] if reversal_days > 0 else np.empty((0, N))
        score_b = signal_factor_reversal(
            cov_slice,
            X_short,
            L,
            cfg.reversal_factor_z_threshold,
        )
        score_c = signal_residual_momentum(
            cov_slice, L, residual_days
        )
        score_risk = signal_risk_adjustment(cov_slice, L)

        score = composite_score(
            score_a,
            score_b,
            score_c,
            score_risk,
            cfg.weight_factor_momentum,
            cfg.weight_factor_reversal,
            cfg.weight_residual_momentum,
            cfg.weight_risk_adjustment,
        )

        w = build_long_short_weights(
            score,
            cfg.score_long_threshold,
            cfg.score_short_threshold,
            cfg.gross_leverage,
            cfg.net_exposure_cap,
            cfg.max_single_long_pct,
            cfg.max_single_short_pct,
        )

        hold_ret = hold_X @ w
        return_acc.extend(hold_ret.tolist())
        date_acc.extend(dates[hold_slice].tolist())

        log_entry: dict = {
            "rebalance_date": dates[t],
            "asset_names": asset_names,
            "weights": w.tolist(),
            "score": score.tolist(),
            "score_factor_momentum": score_a.tolist(),
            "score_factor_reversal": score_b.tolist(),
            "score_residual_momentum": score_c.tolist(),
            "score_risk": score_risk.tolist(),
        }
        rebalance_log.append(log_entry)

    n = len(return_acc)
    dates_trim = date_acc[:n]
    return_series = pd.Series(return_acc, index=dates_trim, name="Alpha Model 2")

    return AlphaModel2Results(
        return_series=return_series,
        rebalance_log=rebalance_log,
    )
