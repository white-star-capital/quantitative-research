# quantitative-research

## June 4, 2026
### Risk-Premium PCA for TAO Subnet Portfolios

This research piece applies **Risk-Premium PCA (RP-PCA)** to Bittensor subnet returns. The approach decomposes the composite matrix **M = Σ + γμμ'** (Lettau & Pelger, 2020) into latent factors that capture both covariance structure and mean returns, then constructs tangency portfolios in factor space and validates them with walk-forward out-of-sample backtests across crypto market regimes.

| Resource | Link |
|----------|------|
| **Published article** | [Polaris by White Star Digital](https://whitestardigitalresearch.substack.com/p/polaris-by-white-star-digital-1) — White Star Digital Research (Substack) |
| **Project code** | [`risk_premium_pca/`](risk_premium_pca/) — Python RP-PCA implementation, TAO subnet data loader, walk-forward backtest engine, and Streamlit dashboard |
| **Documentation** | [`risk_premium_pca/README.md`](risk_premium_pca/README.md) |



## July 2, 2026
### Jump-Diffusion Pricing for Prediction Market Headlines

This research piece applies a **jump-diffusion probability model** to geopolitical Polymarket events, blending news-driven LLM analysis with regime-aware Monte Carlo pricing. The engine simulates probability paths with gradual drift, Poisson news jumps, and decay toward resolution, then outputs fair probability, edge, Kelly sizing, and trade signals for binary and multi-outcome markets.

| Resource | Link |
|----------|------|
| **Published article** | *TBD — link forthcoming* |
| **Project code** | [`pricing-headlines-jump-diffusion-model/`](pricing-headlines-jump-diffusion-model/) — Hourly Polymarket discovery agent, Ollama + Brave news pipeline, jump-diffusion pricing engine, and Streamlit dashboard |
| **Documentation** | [`pricing-headlines-jump-diffusion-model/README.md`](pricing-headlines-jump-diffusion-model/README.md) |
