"""
Fama-MacBeth (1973) two-pass cross-sectional asset pricing test.

Validates whether RP-PCA (or PCA) factors are *priced* risk factors —
i.e. whether exposure to these factors earns a statistically significant
risk premium in the cross-section of asset returns.

Procedure
---------
**Pass 1 (time-series):** For each asset *i*, regress its return on the
K factor returns over the first ``split_frac`` of the sample to obtain
betas β_i  (N × K).

**Pass 2 (cross-sectional):** For each date *t* in the second half,
regress the N asset returns on their betas:
    r_{i,t} = λ_0,t + β_i' λ_t + η_{i,t}

The average λ̄ across time periods is the estimated risk premium for
each factor.  T-statistics are computed with the optional Shanken (1992)
errors-in-variables correction.

Reference
---------
Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium:
Empirical tests. *Journal of Political Economy*, 81(3), 607–636.

Shanken, J. (1992). On the estimation of beta-pricing models.
*Review of Financial Studies*, 5(1), 1–33.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FamaMacBethResult:
    """Results from the two-pass Fama-MacBeth procedure."""

    # Factor risk premia (average lambdas, excluding intercept)
    risk_premia: np.ndarray          # (K,)
    risk_premia_tstat: np.ndarray    # (K,) Shanken-corrected t-stats

    # Intercept (pricing error test: should be zero if model prices correctly)
    intercept: float
    intercept_tstat: float

    # Cross-sectional R² time series
    cross_sectional_r2: np.ndarray   # (T2,)
    mean_r2: float

    # Raw outputs
    lambdas: np.ndarray              # (T2, K+1) — col 0 = intercept
    betas: np.ndarray                # (N, K)

    # Metadata
    model_name: str
    n_factors: int
    n_assets: int
    n_periods_pass2: int

    def summary_df(self) -> pd.DataFrame:
        """Summary table: one row per factor with lambda, t-stat, significance."""
        rows = [
            {
                "Factor": "Intercept",
                "Risk Premium": round(float(self.intercept), 6),
                "t-stat": round(float(self.intercept_tstat), 3),
                "Significant (5%)": abs(self.intercept_tstat) > 1.96,
            }
        ]
        for k in range(self.n_factors):
            rows.append({
                "Factor": f"F{k+1}",
                "Risk Premium": round(float(self.risk_premia[k]), 6),
                "t-stat": round(float(self.risk_premia_tstat[k]), 3),
                "Significant (5%)": abs(self.risk_premia_tstat[k]) > 1.96,
            })
        df = pd.DataFrame(rows).set_index("Factor")
        df.attrs["mean_r2"] = round(self.mean_r2, 4)
        df.attrs["n_periods"] = self.n_periods_pass2
        df.attrs["n_assets"] = self.n_assets
        return df

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"Fama-MacBeth Results: {self.model_name}",
            f"  N assets: {self.n_assets}, K factors: {self.n_factors}, "
            f"T2 periods: {self.n_periods_pass2}",
            f"  Mean cross-sectional R²: {self.mean_r2:.4f}",
            f"  Intercept: {self.intercept:.6f}  (t={self.intercept_tstat:.3f})",
        ]
        for k in range(self.n_factors):
            sig = "*" if abs(self.risk_premia_tstat[k]) > 1.96 else ""
            lines.append(
                f"  F{k+1}: λ={self.risk_premia[k]:.6f}  "
                f"(t={self.risk_premia_tstat[k]:.3f}){sig}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def fama_macbeth(
    returns: np.ndarray,
    factors: np.ndarray,
    model_name: str = "RP-PCA",
    shanken_correction: bool = True,
    split_frac: float = 0.5,
) -> FamaMacBethResult:
    """
    Two-pass Fama-MacBeth (1973) cross-sectional asset pricing test.

    Parameters
    ----------
    returns : ndarray (T, N)
        Asset return panel.
    factors : ndarray (T, K)
        Factor return panel (e.g. ``X @ loadings`` from RP-PCA).
    model_name : str
        Label for display.
    shanken_correction : bool
        Apply Shanken (1992) EIV correction to t-stats.
    split_frac : float
        Fraction of sample for pass 1 (time-series betas).
        Remaining used for pass 2 (cross-sectional regressions).
        If 0, use full sample for both (in-sample, less conservative).

    Returns
    -------
    FamaMacBethResult
    """
    T, N = returns.shape
    K = factors.shape[1]
    assert factors.shape[0] == T, "returns and factors must have same T"

    # Split point
    if split_frac > 0:
        T1 = max(K + 2, int(T * split_frac))
    else:
        T1 = T  # use full sample for both passes

    T2 = T - T1
    if T2 < 10:
        logger.warning(
            "Fama-MacBeth pass 2 has only %d periods (need ≥10 for reliable inference). "
            "Consider reducing split_frac.",
            T2,
        )

    # ── Pass 1: Time-series regressions ───────────────────────────────
    # For each asset i, regress r_{i,t} on F_t over t=0..T1-1
    # r_i = alpha_i + F @ beta_i + eps_i
    F1 = factors[:T1]  # (T1, K)
    R1 = returns[:T1]  # (T1, N)

    # Add intercept column
    X1 = np.column_stack([np.ones(T1), F1])  # (T1, K+1)
    # OLS: beta_hat = (X'X)^{-1} X'R  → shape (K+1, N)
    try:
        beta_hat = np.linalg.lstsq(X1, R1, rcond=None)[0]  # (K+1, N)
    except np.linalg.LinAlgError:
        logger.error("Singular matrix in pass 1 OLS")
        return _empty_result(model_name, N, K)

    betas = beta_hat[1:].T  # (N, K) — factor betas only (no intercept)

    # ── Pass 2: Cross-sectional regressions ───────────────────────────
    # For each t in [T1, T), regress r_{i,t} on beta_i
    if split_frac > 0:
        R2 = returns[T1:]  # (T2, N)
    else:
        R2 = returns  # full sample
        T2 = T

    # Add intercept to betas
    B = np.column_stack([np.ones(N), betas])  # (N, K+1)

    lambdas = np.empty((T2, K + 1))
    r2_series = np.empty(T2)

    try:
        BtB_inv = np.linalg.inv(B.T @ B)
    except np.linalg.LinAlgError:
        logger.error("Singular beta matrix in pass 2")
        return _empty_result(model_name, N, K)

    for t in range(T2):
        r_t = R2[t]  # (N,)
        # OLS: lambda_t = (B'B)^{-1} B' r_t
        lam_t = BtB_inv @ (B.T @ r_t)  # (K+1,)
        lambdas[t] = lam_t

        # Cross-sectional R²
        r_hat = B @ lam_t
        ss_res = np.sum((r_t - r_hat) ** 2)
        ss_tot = np.sum((r_t - r_t.mean()) ** 2)
        r2_series[t] = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    # ── Risk premia and t-statistics ──────────────────────────────────
    lambda_bar = lambdas.mean(axis=0)  # (K+1,)
    lambda_var = lambdas.var(axis=0, ddof=1) / T2  # (K+1,)

    if shanken_correction and K > 0:
        # Shanken (1992): Var_corrected = Var_FM * (1 + c)
        # where c = lambda_f' Sigma_F^{-1} lambda_f
        lambda_f = lambda_bar[1:]  # factor risk premia only
        if split_frac > 0:
            Sigma_F = np.cov(factors[:T1].T)
        else:
            Sigma_F = np.cov(factors.T)

        if Sigma_F.ndim == 0:
            Sigma_F = np.array([[Sigma_F]])

        try:
            Sigma_F_inv = np.linalg.inv(Sigma_F)
            c = float(lambda_f @ Sigma_F_inv @ lambda_f)
        except np.linalg.LinAlgError:
            c = 0.0
            logger.warning("Could not invert Sigma_F for Shanken correction")

        lambda_var_corrected = lambda_var * (1.0 + c)
    else:
        lambda_var_corrected = lambda_var

    # t-stats
    tstats = np.where(
        lambda_var_corrected > 1e-15,
        lambda_bar / np.sqrt(lambda_var_corrected),
        0.0,
    )

    return FamaMacBethResult(
        risk_premia=lambda_bar[1:],
        risk_premia_tstat=tstats[1:],
        intercept=float(lambda_bar[0]),
        intercept_tstat=float(tstats[0]),
        cross_sectional_r2=r2_series,
        mean_r2=float(np.mean(r2_series)),
        lambdas=lambdas,
        betas=betas,
        model_name=model_name,
        n_factors=K,
        n_assets=N,
        n_periods_pass2=T2,
    )


def compare_fama_macbeth(
    returns: np.ndarray,
    rppca_factors: np.ndarray,
    pca_factors: np.ndarray,
    **kwargs,
) -> tuple[FamaMacBethResult, FamaMacBethResult]:
    """Run Fama-MacBeth tests for both RP-PCA and PCA, return both results."""
    rp_result = fama_macbeth(
        returns, rppca_factors, model_name="RP-PCA", **kwargs
    )
    pca_result = fama_macbeth(
        returns, pca_factors, model_name="PCA", **kwargs
    )
    return rp_result, pca_result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _empty_result(model_name: str, N: int, K: int) -> FamaMacBethResult:
    """Return an empty result when estimation fails."""
    return FamaMacBethResult(
        risk_premia=np.zeros(K),
        risk_premia_tstat=np.zeros(K),
        intercept=0.0,
        intercept_tstat=0.0,
        cross_sectional_r2=np.array([]),
        mean_r2=0.0,
        lambdas=np.empty((0, K + 1)),
        betas=np.zeros((N, K)),
        model_name=model_name,
        n_factors=K,
        n_assets=N,
        n_periods_pass2=0,
    )
