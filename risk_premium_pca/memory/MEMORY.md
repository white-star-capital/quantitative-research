# Project Memory

## RP-PCA Implementation
- Location: `rp_pca/` in the worktree root
- 24 Python files, fully tested end-to-end
- Use pyenv Python: `/Users/ale/.pyenv/shims/python3` (has pandas, numpy, scikit-learn)
- System Python `/usr/bin/python3` (3.9.6) does NOT have pandas

## Key Architecture
- `rp_pca/config.py` — all hyperparameters (Config dataclass)
- `rp_pca/models/rp_pca.py` — core RPPCA class; gamma=None → auto T
- `rp_pca/backtest/engine.py` — WalkForwardBacktest with separate cov/mean windows
- `rp_pca/robustness/bootstrap.py` — CircularBlockBootstrap (Politis-Romano 1994)
- `rp_pca/app.py` — Streamlit dashboard (5 tabs)
- `rp_pca/scripts/run_pipeline.py` — headless CLI runner

## RP-PCA Math
M = Σ + γ·μμ'
- γ=0: centered PCA; γ=1: uncentered PCA; γ=T (auto): RP-PCA
- Modular: separate EWMA windows for Σ (halflife=60d) and μ (halflife=21d)
- Top-K=5 factors → tangency + min-var portfolios

## Running
```bash
streamlit run rp_pca/app.py       # dashboard
python3 rp_pca/scripts/run_pipeline.py  # headless
```

## Data
- Binance public REST API (no key): daily OHLCV cached as Parquet
- 30 USDT pairs (BTC, ETH, BNB, ... see data/universe.py)
- Jan 2021 – Dec 2025

## Repo Context
- Multiple unrelated strategy projects in subdirectories
- Root requirements.txt has numpy/pandas/scikit-learn/plotly
