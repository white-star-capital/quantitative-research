# Technical Specification: How Principal Components Are Determined

This document describes precisely how the codebase computes principal components (PCs) for both standard (uncentered) PCA and Risk-Premium PCA (RP-PCA). All formulas and code references are exact.

---

## 1. Scope and Conventions

- **Input**: The system never uses raw price levels for the decomposition. All PCs are determined from **returns** (and their estimated second moments).
- **Notation**:
  - \( T \): number of time observations (e.g. trading days).
  - \( N \): number of assets.
  - \( X \in \mathbb{R}^{T \times N} \): return matrix (rows = dates, columns = assets). In code this is `returns.values` or the array passed to `RPPCA.fit()` / `UncenteredPCA.fit()`.
  - \( \Sigma \in \mathbb{R}^{N \times N} \): estimated covariance matrix of returns (demeaned).
  - \( \mu \in \mathbb{R}^N \): estimated mean return vector.
  - \( K \): number of principal components (`n_components`).
  - \( L \in \mathbb{R}^{N \times K} \): loading matrix (eigenvectors); columns are PC directions.
  - \( F = X L \in \mathbb{R}^{T \times K} \): factor return matrix.

---

## 2. From Prices to Returns

Principal components are **not** determined by market price. They are determined by the **return matrix** \( X \), which is produced as follows.

### 2.1 Data source

- Raw data: OHLCV (e.g. daily close) per asset. Fetched via `BinanceFetcher` and exposed as a DataFrame of **closing prices** \( P \in \mathbb{R}^{T_0 \times N} \) (index = dates, columns = tickers).
- Location: `rp_pca/data/fetcher.py`; pipeline entry points: `rp_pca/scripts/run_pipeline.py`, `rp_pca/app.py`.

### 2.2 Return construction

`ReturnProcessor.fit_transform(prices)` in `rp_pca/data/processor.py`:

1. **Forward-fill** short gaps (default: max 3 consecutive days).
2. **Drop assets** with fewer than `min_obs_fraction × T` non-null observations (default 80%).
3. **Returns**:
   - If `use_log_returns=True` (default):
     \[
     r_{t,n} = \ln\left(\frac{P_{t,n}}{P_{t-1,n}}\right).
     \]
   - Else (simple returns): \( r_{t,n} = (P_{t,n} - P_{t-1,n}) / P_{t-1,n} \).
   - Implementation: `np.log(prices / prices.shift(1)).iloc[1:]` or `prices.pct_change().iloc[1:]`.
4. **Winsorise** each asset’s return series at configurable percentiles (default 1st and 99th); see `_winsorise_df`.
5. **Align** to a common date index and drop all-NaN rows.

Output: \( X \in \mathbb{R}^{T \times N} \) with \( T = \text{len(dates)} - 1 \). This \( X \) is the only data used for moment estimation and PCA/RP-PCA.

---

## 3. Moment Estimation (Σ and μ)

The composite matrix that defines the principal components depends on \( \Sigma \) and \( \mu \). The codebase supports **separate** estimation for covariance and mean (e.g. different windows or estimators in the backtest).

### 3.1 Covariance \( \Sigma \)

- **Sample** (`sample_cov` in `rp_pca/models/covariance.py`):
  \[
  \Sigma = \frac{1}{T-1}\, (X - \bar{X})^\top (X - \bar{X}), \qquad \bar{X} = \frac{1}{T}\sum_{t=1}^T X_{t,\cdot}.
  \]
  Implemented as `np.cov(returns, rowvar=False)` (variables = columns, \( \Rightarrow (N,N) \)).

- **EWMA** (`ewma_cov`): Exponentially weighted covariance with half-life \( h \). Let \( \lambda = 0.5^{1/h} \), \( w_t = \lambda^{T-1-t} / \sum_s \lambda^{T-1-s} \). Weighted mean \( \tilde{\mu} = \sum_t w_t X_{t,\cdot} \), demeaned \( \tilde{X}_t = X_t - \tilde{\mu} \), then
  \[
  \Sigma = \sum_{t=1}^T w_t\, \tilde{X}_t^\top \tilde{X}_t.
  \]
  Config: `ewma_cov_halflife` (default 60 days).

- **Ledoit–Wolf** (`ledoit_wolf_cov`): `sklearn.covariance.LedoitWolf().fit(returns).covariance_`; mean is still from the chosen mean estimator (e.g. sample).

### 3.2 Mean \( \mu \)

- **Sample** (`sample_mean`): \( \mu = \frac{1}{T}\sum_{t=1}^T X_{t,\cdot} \) — `returns.mean(axis=0)`.
- **EWMA** (`ewma_mean`): \( \mu = \sum_t w_t X_{t,\cdot} \) with \( w_t \propto \lambda^{T-1-t} \), \( \lambda = 0.5^{1/\text{halflife}} \). Config: `ewma_mean_halflife` (default 21 days).

### 3.3 Joint estimator and backtest windows

- **Factory**: `get_cov_estimator(method, **kwargs)` returns a callable that, given \( X \), returns `(Sigma, mu)` using the chosen covariance and mean estimators (`rp_pca/models/covariance.py`).
- **Walk-forward backtest** (`rp_pca/backtest/engine.py`): At each rebalance time \( t \),
  - `cov_slice` = \( X[\max(0, t - \text{cov\_window}) : t] \)
  - `mean_slice` = \( X[\max(0, t - \text{mean\_window}) : t] \)
  - \( \Sigma, \mu = \_estimate\_moments(\text{cov\_slice}, \text{mean\_slice}) \).

So \( \Sigma \) and \( \mu \) can use **different windows** (e.g. longer for covariance, shorter for mean). PCA/RP-PCA are then fit with this precomputed \( (\Sigma, \mu) \).

---

## 4. Composite Matrix and Principal Component Definition

Principal components are the **eigenvectors** of a symmetric \( N \times N \) matrix \( M \), ordered by decreasing eigenvalue. The definition of \( M \) differs between standard uncentered PCA and RP-PCA.

### 4.1 Risk-Premium PCA (RP-PCA) — primary path

**Reference**: Lettau & Pelger (2020); implementation: `rp_pca/models/rp_pca.py`, class `RPPCA`.

- **Composite matrix**
  \[
  M = \Sigma + \gamma\, \mu \mu^\top.
  \]
  Here \( \gamma \in (0,+\infty) \) is the penalty parameter. If `gamma=None`, the code sets \( \gamma = T \) at fit time so the mean term is on a comparable scale to the covariance (Lettau–Pelger convention).

- **Special cases**:
  - \( \gamma = 0 \): \( M = \Sigma \) — centered PCA (only variance).
  - \( \gamma = 1 \): \( M = \Sigma + \mu\mu^\top \) — uncentered (second-moment) PCA.
  - \( \gamma > 1 \) (e.g. \( \gamma = T \)): RP-PCA — emphasises directions that explain both variance and mean return.

**Code** (`RPPCA.fit()`):

```text
Sigma = cov_matrix if cov_matrix is not None else sample_cov(returns)
mu    = mean_vector if mean_vector is not None else sample_mean(returns)
M     = Sigma + gamma * np.outer(mu, mu)
eigenvalues, eigenvectors = np.linalg.eigh(M)
# Sort descending by eigenvalue; take first n_components eigenvectors as loadings
```

- **Eigendecomposition**: `np.linalg.eigh(M)` (symmetric); eigenvalues and eigenvectors are then sorted in **descending** order; the first \( K \) eigenvectors form the columns of \( L \).
- **Loadings**: `loadings_` = \( L \in \mathbb{R}^{N \times K} \).
- **Factor returns**: \( F = X L \), i.e. `factors_ = returns @ loadings_` (and out-of-sample: `transform(returns_new)` = `returns_new @ loadings_`).

So **principal components are determined by the spectrum of \( M = \Sigma + \gamma \mu\mu^\top \)** where \( \Sigma \) and \( \mu \) come from the **return** matrix \( X \) (and optionally from different windows/estimators in the backtest).

### 4.2 Standard uncentered PCA (alternative implementation)

**Implementation**: `rp_pca/models/pca.py`, class `UncenteredPCA`.

- **Second-moment matrix** (no separate \( \Sigma, \mu \) API):
  \[
  M = \frac{1}{T}\, X^\top X = \Sigma_{\text{sample}} + \mu \mu^\top,
  \]
  where \( \Sigma_{\text{sample}} \) and \( \mu \) are the sample covariance and mean of \( X \). So this is the \( \gamma=1 \) case applied to the same \( X \).

- **Code**: `M = (returns.T @ returns) / T`; then `eigh(M)`; sort descending; first \( K \) eigenvectors = loadings; `factors_ = returns @ loadings_`.

In the **backtest and app**, “PCA” is implemented as **RPPCA with γ=1** and the **same** precomputed \( (\Sigma, \mu) \) as RP-PCA, so both methods see the same moments and only differ in \( \gamma \):

```text
pca = RPPCA(n_components=K, gamma=1.0)
pca.fit(cov_slice, cov_matrix=Sigma, mean_vector=mu)
```

So in practice, **both PCA and RP-PCA principal components are determined by the same \( \Sigma \) and \( \mu \)**; the only difference is \( \gamma \) in \( M = \Sigma + \gamma \mu\mu^\top \).

---

## 5. Explained Variance (RP-PCA)

Explained variance is reported **relative to \( \Sigma \)** (not \( M \)), to match the article and to reflect variance explained in the demeaned returns:

\[
\text{var}_k = \frac{\ell_k^\top \Sigma \ell_k}{\operatorname{tr}(\Sigma)}, \qquad
\text{cumulative}_k = \sum_{j=1}^k \text{var}_j.
\]

Code: `rp_pca/models/rp_pca.py`, after computing loadings, `total_var = np.trace(Sigma)` and `factor_var[k] = loadings_[:, k] @ Sigma @ loadings_[:, k]`.

---

## 6. End-to-End Flow Summary

| Stage | What determines PCs | Code / config |
|-------|----------------------|----------------|
| 1 | **Returns** \( X \) from prices (log or simple, winsorised) | `ReturnProcessor.fit_transform(prices)` — `processor.py` |
| 2 | **Σ** from \( X \) (sample / EWMA / Ledoit–Wolf), possibly on `cov_slice` | `sample_cov` / `ewma_cov` / `ledoit_wolf_cov` — `covariance.py`; backtest: `_estimate_moments(cov_slice, mean_slice)` |
| 3 | **μ** from \( X \) (sample / EWMA), possibly on `mean_slice` | `sample_mean` / `ewma_mean` — `covariance.py` |
| 4 | **M** = Σ + γ μμ′ | `RPPCA.fit(..., cov_matrix=Sigma, mean_vector=mu)` — `rp_pca.py` |
| 5 | **Loadings** = first \( K \) eigenvectors of \( M \) (descending eigenvalue) | `np.linalg.eigh(M)` then sort and slice — `rp_pca.py` |
| 6 | **Factor returns** \( F = X L \) | `returns @ loadings_`; OOS: `hold_X @ loadings_` |

Principal components are therefore **fully determined by the return matrix** \( X \) and the chosen moment estimators and windows: they are **not** determined by price levels, only by the **distribution of returns** (and the parameter \( \gamma \) for RP-PCA).

---

## 7. References

- Lettau, M., & Pelger, M. (2020). Estimating latent asset-pricing factors. *Journal of Finance*, 75(2), 919–969.
- Kiefer & Nowotny (2025) — application to crypto (see project README).
- `rp_pca/models/rp_pca.py` — `RPPCA.fit()`, `transform()`.
- `rp_pca/models/pca.py` — `UncenteredPCA.fit()`.
- `rp_pca/models/covariance.py` — all \( \Sigma \) and \( \mu \) estimators.
- `rp_pca/backtest/engine.py` — `_estimate_moments`, use of `cov_slice` / `mean_slice`, and `RPPCA.fit(cov_slice, cov_matrix=Sigma, mean_vector=mu)`.
