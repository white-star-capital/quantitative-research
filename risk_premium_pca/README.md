# Risk-Premium PCA 

Python implementation of **Risk-Premium PCA (RP-PCA)** applied to a portfolio of 15 tokens, based on the research paper by Kiefer & Nowotny (2025).

**Core reference:** Lettau, M. & Pelger, M. (2020). *Estimating latent asset-pricing factors.* Journal of Finance.

## Architecture

```
rp_pca/
├── config.py                   # All hyperparameters in one place
├── app.py                      # Streamlit interactive dashboard
├── data/
│   ├── universe.py             # 30-coin universe definition
│   ├── fetcher.py              # Binance OHLCV downloader (cached)
│   ├── glosten_harris.py       # Spread estimation + mid-price recovery
│   └── processor.py            # Return computation + winsorisation
├── models/
│   ├── pca.py                  # Standard uncentered PCA (γ=1)
│   ├── rp_pca.py               # RP-PCA — the core algorithm
│   └── covariance.py           # Sample / EWMA / Ledoit-Wolf estimators
├── portfolio/
│   ├── construction.py         # Tangency + min-var from factor loadings
│   └── metrics.py              # Sharpe, vol, max DD, Sortino, Calmar
├── backtest/
│   ├── engine.py               # Walk-forward OOS backtest
│   └── regimes.py              # 6 crypto market regime definitions
├── robustness/
│   └── bootstrap.py            # Circular block bootstrap (Politis-Romano 1994)
├── analysis/
│   └── plots.py                # Plotly charts (frontier, cumulative, loadings…)
└── scripts/
    └── run_pipeline.py         # Headless CLI runner
```

---

## Quick Start

### 1. Install dependencies

```bash
cd rp_pca
pip install -r rp_pca/requirements.txt
```

### 2. Interactive Dashboard

```bash
streamlit run rp_pca/app.py
```

Navigate to `http://localhost:8501`.

---

## Factor Determination Process

Factors are determined through **PCA/RP-PCA decomposition** on return data. Here's the flow:

### 1. **Data Loading** → `rp_pca/data/`
   - **`fetcher.py`** or **`tao_subnet_loader.py`**: Downloads/loads raw prices
   - **`processor.py`**: Converts prices → **log returns** (daily percentage changes)
   - Returns DataFrame: shape `(T, N)` where T=days, N=assets

### 2. **Covariance/Mean Estimation** → `rp_pca/models/covariance.py`
   - **`sample_cov(returns)`**: Standard sample covariance matrix Σ (N × N)
   - **`sample_mean(returns)`**: Mean return vector μ (N,)
   - Also supports EWMA and Ledoit-Wolf shrinkage
   - **Key insight**: Both Σ and μ are estimated from rolling windows in walk-forward backtests

### 3. **RP-PCA Decomposition** → `rp_pca/models/rp_pca.py` (the MAGIC)
   This is where factors are created:

   ```python
   class RPPCA:
       def fit(self, returns, cov_matrix=Σ, mean_vector=μ):
           # Step 1: Form composite matrix
           M = Σ + γ · μ · μ'    # Lettau & Pelger (2020) formula
           
           # Step 2: Eigendecomposition (symmetric → eigh)
           eigenvalues, eigenvectors = np.linalg.eigh(M)
           
           # Step 3: Extract top-K eigenvectors as factor loadings
           self.loadings_ = eigenvectors[:, :K]  # (N, K) matrix
           
           # Step 4: Project returns onto loadings = FACTORS
           self.factors_ = returns @ self.loadings_  # (T, K) matrix
   ```

   **What happens:**
   - **γ (gamma)** controls the "RP-PCA flavor":
     - γ = 0 → standard centered PCA (ignores mean returns)
     - γ = 1 → uncentered PCA (second moment matrix)
     - γ > 1 → **Risk-Premium PCA** (favors high-mean factors)
   - **Eigenvalues** = importance of each factor (higher = more variance/return)
   - **Eigenvectors** = **factor loadings** L (how each asset loads onto each factor)
   - **Factors F = X · L** = the actual factor returns (time series of each factor)

### 4. **Portfolio Construction from Factors** → `rp_pca/portfolio/construction.py`
   ```python
   class PortfolioConstructor:
       def __init__(self, loadings, factor_returns):
           # loadings = L (N, K)
           # factor_returns = F (T, K)
           
           # Compute factor-space mean/cov
           μ_F = F.mean(axis=0)      # (K,) mean return per factor
           Σ_F = cov(F)              # (K, K) covariance of factors
           
           # Tangency: solve max Sharpe ratio in factor space
           w_F = Σ_F^{-1} · (μ_F - r_f) / denominator
           
           # Map back to asset space
           w_asset = L @ w_F         # (N,) = asset weights
   ```

### 5. **In-Sample Tab Visualization** → `rp_pca/app.py` & `rp_pca/analysis/plots.py`
   Displays factors via:
   - **`plot_explained_variance()`**: Bar chart of eigenvalues
   - **`plot_factor_loadings()`**: Heatmap of L (asset × factor)
   - **`plot_factor_sharpe()`**: Each factor's Sharpe, volatility, return

### 6. **Walk-Forward Backtest** → `rp_pca/backtest/engine.py`
   At each rebalance date `t`:
   1. Estimate Σ, μ from rolling window
   2. **Fit RP-PCA** → get new factors (Σ changes over time)
   3. Construct portfolio from factors
   4. Hold, then rebalance

---

## The "Best Features"

Now adds a **6th step** after factor determination:

```python
# In tab_best_features():

# Step 1: Get factors from In-Sample fit
factors_df = rppca.factor_sharpe()  # Ann. Sharpe, variance explained per factor

# Step 2: Score factors
scored = _score_factors(factors_df, sharpe_weight=0.6, variance_weight=0.4)
# score = 0.6 · |Sharpe|_norm + 0.4 · VarExpl%_norm

# Step 3: Select best factors
selected = [0, 1, 2]  # Keep PC1, PC2, PC3; drop PC4, PC5

# Step 4: Run backtest with ONLY selected factors
bt = WalkForwardBacktest(..., selected_factors=[0, 1, 2])
# Inside the backtest, when fitting RP-PCA:
#   rp_L = rp.loadings_[:, [0, 1, 2]]  # Subset to 3 factors only
#   portfolio construction uses only 3 factors instead of 5
```

---

| Component | What it does |
|-----------|-------------|
| **Data** | Raw prices → log returns (T, N) |
| **Covariance/Mean** | Σ (N,N) and μ (N,) estimation |
| **RPPCA.fit()** | **M = Σ + γμμ' → eigendecomposition → loadings L and factors F** ← **THE MAGIC** |
| **Portfolio** | L and F → tangency/min-var weights |
| **Backtest** | Rolling RPPCA at each date, optionally subset to best factors |

**The key insight**: Factors are **eigenvectors of the composite matrix M** (Σ + γμμ'). Higher eigenvalues = more important factors. The new Best Features tab just lets you **prune the less important ones** before backtesting.

---

## References

- Lettau, M., & Pelger, M. (2020). Estimating latent asset-pricing factors. *Journal of Finance*, 75(2), 919–969.
- Barillas, F., & Shanken, J. (2018). Comparing asset pricing models. *Journal of Finance*, 73(2), 715–754.
- Glosten, L. R., & Harris, L. E. (1988). Estimating the components of the bid/ask spread. *Journal of Financial Economics*, 21(1), 123–142.
- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *JASA*, 89(428), 1303–1313.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *JMVA*, 88(2), 365–411.

---

*Nothing in this implementation is investment advice.*
