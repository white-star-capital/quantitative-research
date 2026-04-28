"""
Risk-Premium PCA — Interactive Streamlit Dashboard

Run with:
    cd rp_pca
    streamlit run app.py

Tabs
----
1. Data          — download prices, inspect return distributions
2. In-Sample     — PCA vs RP-PCA decomposition, loadings, Sharpe per factor
3. Portfolio     — tangency + min-var portfolios, efficient frontier
4. OOS Backtest  — walk-forward performance table + cumulative returns + regime analysis
5. Best Configs  — full parameter sweep, top-5 configurations
6. Best Features — factor selection, scoring, and selected-factor backtest
7. Fama-MacBeth  — cross-sectional asset pricing test (priced risk factor validation)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup (allow running from within the rp_pca directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from rp_pca.config import (
    Config,
    RESULTS_DIR,
    ROOT_DIR,
    TAO_SUBNET_EXPORT_END,
    TAO_SUBNET_EXPORT_START,
    TAO_SUBNET_WIDE_PARQUET,
)
from rp_pca.data.universe import UNIVERSE_30
from rp_pca.data.fetcher import BinanceFetcher
from rp_pca.data.processor import ReturnProcessor, equal_weighted_returns
from rp_pca.data.tao_subnet_loader import load_tao_subnet_prices, load_tao_subnet_market_caps
from rp_pca.models.rp_pca import RPPCA
from rp_pca.models.covariance import get_cov_estimator
from rp_pca.portfolio.construction import PortfolioConstructor
from rp_pca.portfolio.metrics import compute_metrics_table
from rp_pca.backtest.engine import WalkForwardBacktest
from rp_pca.backtest.sweep import (
    run_parameter_sweep,
    build_sweep_dataframe,
    top_n_configs,
    metric_col,
    STRATEGIES_TO_CAPTURE,
    METRICS_TO_CAPTURE,
    PARAM_COLS,
)
from rp_pca.analysis.plots import (
    plot_efficient_frontier,
    plot_cumulative_returns,
    plot_explained_variance,
    plot_factor_loadings,
    plot_factor_sharpe,
    plot_rolling_sharpe,
    plot_correlation_heatmap,
    plot_regime_comparison,
    plot_fama_macbeth_comparison,
    plot_cross_sectional_r2,
)
from rp_pca.robustness.regimes import classify_regimes, compute_regime_metrics, plot_regime_timeline
from rp_pca.robustness.fama_macbeth import fama_macbeth, compare_fama_macbeth

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Plotly viewer options: pass via ``config=`` only. Do not pass ``width=`` to
# ``st.plotly_chart`` (Streamlit 1.50 treats unknown kwargs as deprecated).
# Use ``use_container_width=True`` for full-width charts.
_ST_PLOTLY_CONFIG: dict = {"displayModeBar": True}


def _metrics_performance_styler(df: pd.DataFrame):
    """Performance table styler; omit Sharpe gradient if that column is all-NaN."""
    pct_dd = [c for c in df.columns if "%" in c or "DD" in c]
    sty = df.style.format("{:.3f}", subset=["Ann. Sharpe", "Sortino"]).format(
        "{:.2f}", subset=pct_dd
    )
    if "Ann. Sharpe" in df.columns and df["Ann. Sharpe"].notna().any():
        sty = sty.background_gradient(subset=["Ann. Sharpe"], cmap="RdYlGn")
    return sty


def _tao_dir_fingerprint(root: Path) -> str:
    root = root.expanduser().resolve()
    pq = root / TAO_SUBNET_WIDE_PARQUET
    if pq.is_file():
        return f"pq:{pq.stat().st_mtime_ns}"
    paths = list(root.glob("sn*_tao_daily_candles.csv"))
    if not paths:
        return "empty"
    return f"csv:{len(paths)}:{max(p.stat().st_mtime_ns for p in paths)}"


@st.cache_data(show_spinner=False)
def _cached_load_tao_subnet_prices(
    dir_str: str, start: str, end: str, fp: str
) -> pd.DataFrame:
    _ = fp  # fingerprint only invalidates cache when files change
    return load_tao_subnet_prices(
        Path(dir_str),
        start_date=start,
        end_date=end,
    )


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RP-PCA Crypto Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "prices": None,
        "returns": None,
        "pca_model": None,
        "rppca_model": None,
        "backtest_results": None,
        "grid_sweep_results": None,
        "signal_backtest_results": None,
        "signal_strategy": None,
        "tao_market_caps": None,
        "config": Config(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------
_BINANCE_DEFAULT_START = "2021-01-01"
_BINANCE_DEFAULT_END   = "2025-12-31"


def render_sidebar() -> Config:
    st.sidebar.title("⚙️ Settings")
    cfg = st.session_state.config

    # ── Auto-populate dates when data source changes ───────────────────
    # Swap to the canonical date range for the active source whenever the
    # dates are still at the *other* source's defaults (i.e. not manually edited).
    if cfg.data.data_source == "tao_subnets":
        if cfg.data.start_date == _BINANCE_DEFAULT_START:
            cfg.data.start_date = TAO_SUBNET_EXPORT_START
        if cfg.data.end_date == _BINANCE_DEFAULT_END:
            cfg.data.end_date = TAO_SUBNET_EXPORT_END
    else:  # binance
        if cfg.data.start_date == TAO_SUBNET_EXPORT_START:
            cfg.data.start_date = _BINANCE_DEFAULT_START
        if cfg.data.end_date == TAO_SUBNET_EXPORT_END:
            cfg.data.end_date = _BINANCE_DEFAULT_END

    st.sidebar.subheader("Data")
    cfg.data.start_date = st.sidebar.text_input("Start date", cfg.data.start_date)
    cfg.data.end_date   = st.sidebar.text_input("End date",   cfg.data.end_date)

    st.sidebar.subheader("Model")
    cfg.model.n_components = st.sidebar.slider(
        "N components (K)", 1, 10, cfg.model.n_components
    )
    gamma_choice = st.sidebar.selectbox(
        "γ (RP-PCA penalty)",
        ["auto (= T)", "0 (centered PCA)", "1 (uncentered PCA)",
         "10", "50", "100", "200", "500"],
        index=0,
    )
    if gamma_choice == "auto (= T)":
        cfg.model.gamma = None
    elif gamma_choice == "0 (centered PCA)":
        cfg.model.gamma = 0.0
    elif gamma_choice == "1 (uncentered PCA)":
        cfg.model.gamma = 1.0
    else:
        cfg.model.gamma = float(gamma_choice)

    cfg.model.cov_method = st.sidebar.selectbox(
        "Covariance estimator",
        ["ewma", "sample", "ledoit_wolf"],
        index=0,
    )

    st.sidebar.subheader("Backtest")
    cfg.backtest.cov_window = st.sidebar.slider(
        "Cov window (days)", 63, 504, cfg.backtest.cov_window, step=21
    )
    cfg.backtest.mean_window = st.sidebar.slider(
        "Mean window (days)", 21, 252, cfg.backtest.mean_window, step=21
    )
    cfg.backtest.rebalance_days = st.sidebar.slider(
        "Rebalance frequency (days)", 5, 63, cfg.backtest.rebalance_days
    )

    st.sidebar.subheader("Risk-free rate")
    cfg.portfolio.risk_free_rate = st.sidebar.slider(
        "Annual risk-free rate (%)", 0.0, 10.0,
        cfg.portfolio.risk_free_rate * 100, step=0.25
    ) / 100.0

    st.session_state.config = cfg
    return cfg


# ---------------------------------------------------------------------------
# TAO path discovery helpers
# ---------------------------------------------------------------------------

def _tao_dir_status(path: Path) -> tuple[bool, str]:
    """
    Return (has_data: bool, status_message: str) for a candidate directory.
    """
    p = path.expanduser().resolve()
    if not p.exists():
        return False, f"Directory not found: `{p}`"
    pq = p / TAO_SUBNET_WIDE_PARQUET
    if pq.is_file():
        size_mb = pq.stat().st_size / 1_048_576
        return True, f"✅ Wide parquet found (`{pq.name}`, {size_mb:.1f} MB)"
    csvs = list(p.glob("sn*_tao_daily_candles.csv"))
    if csvs:
        return True, f"✅ {len(csvs)} subnet CSV files found"
    return False, f"⚠️ Directory exists but contains no TAO data files"


def _discover_tao_candidates() -> list[tuple[Path, str]]:
    """
    Return a list of (path, status_message) for all plausible TAO data locations.
    Checked in priority order: env-var → package-relative → CWD → home.
    """
    import os as _os

    candidates: list[Path] = []
    env = _os.environ.get("TAO_SUBNET_DIR", "").strip()
    if env:
        candidates.append(Path(env).expanduser().resolve())

    # Package-relative (the canonical location)
    candidates.append((ROOT_DIR / "data" / "cache" / "tao_subnets").resolve())
    # CWD-relative (useful when running streamlit from a different working dir)
    candidates.append((Path.cwd() / "tao_subnets").resolve())
    candidates.append((Path.cwd() / "data" / "tao_subnets").resolve())
    # Home-directory convenience location
    candidates.append(Path.home() / "tao_subnets")

    # Deduplicate while preserving order
    seen: set[Path] = set()
    results: list[tuple[Path, str]] = []
    for p in candidates:
        rp = p.expanduser().resolve()
        if rp in seen:
            continue
        seen.add(rp)
        ok, msg = _tao_dir_status(p)
        results.append((p, msg))

    return results


def _render_tao_path_selector(cfg: Config) -> None:
    """
    Render the TAO subnet directory selector with smart discovery,
    live status feedback, and an env-var hint.
    """
    import os as _os

    # ── Path input ────────────────────────────────────────────────────
    tao_dir_str = st.text_input(
        "Subnet data directory",
        value=str(cfg.data.tao_subnet_csv_dir),
        key="tao_subnet_csv_dir_input",
        help=(
            "Folder containing `sn*_tao_daily_candles.csv` files "
            "and/or `tao_subnets_wide.parquet`.  "
            "Set the **TAO_SUBNET_DIR** environment variable to make this "
            "persistent across sessions without editing the path here."
        ),
    )
    cfg.data.tao_subnet_csv_dir = Path(tao_dir_str).expanduser()

    # ── Live status ───────────────────────────────────────────────────
    ok, status_msg = _tao_dir_status(cfg.data.tao_subnet_csv_dir)
    if ok:
        st.success(status_msg)
    else:
        st.error(status_msg)

        # ── Auto-discovery ────────────────────────────────────────────
        candidates = _discover_tao_candidates()
        good = [(p, m) for p, m in candidates if m.startswith("✅")]
        if good:
            st.info(
                f"Found {len(good)} alternate location(s) with TAO data — "
                "click **Use** to switch."
            )
            for p, msg in good:
                col_a, col_b = st.columns([5, 1])
                col_a.markdown(f"`{p}`  \n{msg}")
                if col_b.button("Use", key=f"use_tao_dir_{hash(str(p))}"):
                    cfg.data.tao_subnet_csv_dir = p
                    st.session_state["tao_subnet_csv_dir_input"] = str(p)
                    st.rerun()
        else:
            # No data found anywhere — show setup instructions
            with st.expander("📋 Setup instructions — how to get TAO subnet data", expanded=True):
                pkg_dir = (ROOT_DIR / "data" / "cache" / "tao_subnets").resolve()
                env_hint = _os.environ.get("TAO_SUBNET_DIR", "")
                st.markdown(f"""
**Option A — Co-locate data with the codebase (recommended)**

Run the fetch script to download subnet candle data directly into the default directory:
```bash
# From the project root
python rp_pca/scripts/get_tao_stats_all_subnets.py
```
Files will be written to:
```
{pkg_dir}
```

**Option B — Use an existing data directory**

If you already have `sn*_tao_daily_candles.csv` files elsewhere, paste the path
into the field above.

**Option C — Set an environment variable (persistent)**

Add to your shell profile (`.bashrc`, `.zshrc`, etc.) or `.env` file:
```bash
export TAO_SUBNET_DIR="/path/to/your/tao_subnets"
```
The app will auto-detect this on startup and pre-fill the path above.

**What the directory should contain:**
- `sn1_tao_daily_candles.csv`, `sn2_tao_daily_candles.csv`, … (one per subnet)
- Optionally: `tao_subnets_wide.parquet` (pre-built wide matrix, faster loads)
                """)

    # ── Date-range hint ───────────────────────────────────────────────
    if ok:
        st.caption(
            f"Default Taostats export window: **{TAO_SUBNET_EXPORT_START}** → "
            f"**{TAO_SUBNET_EXPORT_END}**  "
            "(`get_tao_stats_all_subnets.py`). Match sidebar Start / End to your cache.  "
            "Set **TAO_SUBNET_DIR** env var to make this path persist across sessions."
        )


# ---------------------------------------------------------------------------
# Tab 1: Data
# ---------------------------------------------------------------------------
def tab_data(cfg: Config):
    st.header("📥 Data")

    src_ix = 0 if cfg.data.data_source == "binance" else 1
    data_source_choice = st.radio(
        "Price source",
        ["Binance (30 USDT pairs)", "TAO subnets (Taostats CSV)"],
        index=src_ix,
        horizontal=True,
        key="data_source_radio",
    )
    cfg.data.data_source = "binance" if data_source_choice.startswith("Binance") else "tao_subnets"

    if cfg.data.data_source == "tao_subnets":
        _render_tao_path_selector(cfg)

    col1, col2 = st.columns([3, 1])
    with col1:
        if cfg.data.data_source == "binance":
            st.write(
                f"Universe: **{len(UNIVERSE_30)} cryptocurrencies** vs USDT on Binance  "
                f"| Period: **{cfg.data.start_date}** → **{cfg.data.end_date}**"
            )
        else:
            st.write(
                "Universe: **TAO-denominated subnet tokens** (columns `SN{netuid}`) — "
                f"returns are vs holding TAO | Period: **{cfg.data.start_date}** → **{cfg.data.end_date}**"
            )

    def _apply_prices(prices: pd.DataFrame, source_name: str) -> None:
        if len(prices) == 0:
            st.error(f"No data loaded ({source_name}).")
            return
        # Use relaxed min_obs for TAO subnets (different birth dates)
        min_obs = (
            cfg.data.tao_min_obs_fraction
            if cfg.data.data_source == "tao_subnets"
            else cfg.data.min_obs_fraction
        )
        processor = ReturnProcessor(
            winsorize_lower=cfg.data.winsorize_lower,
            winsorize_upper=cfg.data.winsorize_upper,
            min_obs_fraction=min_obs,
        )
        returns = processor.fit_transform(prices)
        st.session_state.prices = prices
        st.session_state.returns = returns
        st.session_state.pca_model = None
        st.session_state.rppca_model = None
        st.session_state.backtest_results = None
        if len(returns) == 0:
            st.error("No valid returns after processing. Check data quality and configuration.")
            return
        st.success(f"Loaded {len(returns)} days × {len(returns.columns)} assets ({source_name})")

    if cfg.data.data_source == "tao_subnets":
        tdir = cfg.data.tao_subnet_csv_dir.expanduser().resolve()
        has_tao_cache = (tdir / TAO_SUBNET_WIDE_PARQUET).is_file() or any(
            tdir.glob("sn*_tao_daily_candles.csv")
        )
        if (
            has_tao_cache
            and st.session_state.prices is None
            and not st.session_state.get("_tao_autoload_attempted", False)
        ):
            st.session_state._tao_autoload_attempted = True
            try:
                fp = _tao_dir_fingerprint(tdir)
                prices = _cached_load_tao_subnet_prices(
                    str(tdir), cfg.data.start_date, cfg.data.end_date, fp
                )
                cfg.backtest.buy_hold_benchmark = "TAO"
                if len(prices) > 0:
                    _apply_prices(prices, "TAO subnets (auto-loaded)")
                    try:
                        st.session_state.tao_market_caps = load_tao_subnet_market_caps(
                            tdir, start_date=cfg.data.start_date, end_date=cfg.data.end_date,
                        )
                    except Exception:
                        st.session_state.tao_market_caps = None
            except (FileNotFoundError, ValueError, OSError) as e:
                st.warning(f"TAO subnet auto-load skipped: {e}")

    with col2:
        load_binance = cfg.data.data_source == "binance" and st.button(
            "🔄 Download / Refresh Data", type="primary"
        )
        load_tao = cfg.data.data_source == "tao_subnets" and st.button(
            "📂 Load subnet CSVs", type="primary"
        )

    if load_binance:
        with st.spinner("Fetching OHLCV from Binance…"):
            fetcher = BinanceFetcher(cache_dir=cfg.data.cache_dir)
            prices = fetcher.fetch_all(
                start_date=cfg.data.start_date,
                end_date=cfg.data.end_date,
            )
            cfg.backtest.buy_hold_benchmark = "BTC"
            _apply_prices(prices, "Binance")

    if load_tao:
        with st.spinner("Loading subnet candles…"):
            try:
                tdir = cfg.data.tao_subnet_csv_dir.expanduser().resolve()
                fp = _tao_dir_fingerprint(tdir)
                prices = _cached_load_tao_subnet_prices(
                    str(tdir), cfg.data.start_date, cfg.data.end_date, fp
                )
            except (FileNotFoundError, ValueError, OSError) as e:
                st.error(str(e))
                prices = pd.DataFrame()
            cfg.backtest.buy_hold_benchmark = "TAO"
            if len(prices) > 0:
                _apply_prices(prices, "TAO subnets")
                try:
                    tdir2 = cfg.data.tao_subnet_csv_dir.expanduser().resolve()
                    st.session_state.tao_market_caps = load_tao_subnet_market_caps(
                        tdir2, start_date=cfg.data.start_date, end_date=cfg.data.end_date,
                    )
                except Exception:
                    st.session_state.tao_market_caps = None

    if st.session_state.returns is not None:
        returns: pd.DataFrame = st.session_state.returns

        if len(returns) == 0:
            st.warning("No data available. The returns dataframe is empty.")
            return

        st.subheader("Return matrix preview")
        st.dataframe(returns.tail(10).style.format("{:.4f}"), use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Trading days", len(returns))
        col_b.metric("Assets", len(returns.columns))
        col_c.metric("Start / End",
                     f"{returns.index[0].date()} → {returns.index[-1].date()}")

        st.subheader("Return distribution (annualised)")
        ann_ret = (returns.mean() * 252 * 100).sort_values(ascending=False)
        ann_vol = (returns.std() * np.sqrt(252) * 100)

        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=ann_ret.index.tolist(),
            y=ann_ret.values,
            marker_color=["#00C49F" if v > 0 else "#FF4444" for v in ann_ret.values],
        ))
        fig.update_layout(
            title="Annualised Mean Return per Asset (%)",
            xaxis_title="Asset",
            yaxis_title="Return (%)",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True, config=_ST_PLOTLY_CONFIG)

        st.plotly_chart(
            plot_correlation_heatmap(returns),
            use_container_width=True,
            config=_ST_PLOTLY_CONFIG,
        )
    else:
        if cfg.data.data_source == "binance":
            st.info("Click **Download / Refresh Data** to load prices from Binance.")
        else:
            st.info(
                "Run `get_tao_stats_all_subnets.py` into this directory (or add CSVs / `tao_subnets_wide.parquet`). "
                "Data auto-loads once per session when present; use **Load subnet CSVs** to refresh."
            )


# ---------------------------------------------------------------------------
# Tab 2: In-Sample Analysis
# ---------------------------------------------------------------------------
def tab_insample(cfg: Config):
    st.header("🔬 In-Sample Analysis")

    if st.session_state.returns is None:
        st.warning("Load data first (Data tab).")
        return

    returns = st.session_state.returns
    X = returns.values

    if st.button("▶ Run PCA + RP-PCA", type="primary"):
        with st.spinner("Fitting models…"):
            estimator = get_cov_estimator(
                cfg.model.cov_method,
                ewma_cov_halflife=cfg.model.ewma_cov_halflife,
                ewma_mean_halflife=cfg.model.ewma_mean_halflife,
            )
            Sigma, mu = estimator(X)

            pca = RPPCA(n_components=cfg.model.n_components, gamma=1.0)
            pca.fit(X, cov_matrix=Sigma, mean_vector=mu)

            rppca = RPPCA(n_components=cfg.model.n_components, gamma=cfg.model.gamma)
            rppca.fit(X, cov_matrix=Sigma, mean_vector=mu)

            st.session_state.pca_model = pca
            st.session_state.rppca_model = rppca
        st.success(f"Fitted PCA (γ=1) and RP-PCA (γ={rppca.gamma_used_:.1f})")

    pca: RPPCA = st.session_state.pca_model
    rppca: RPPCA = st.session_state.rppca_model

    if pca is None:
        return

    # ── Explained variance ───────────────────────────────────────────
    st.subheader("Explained Variance")
    col_pca, col_rp = st.columns(2)
    with col_pca:
        st.write("**Standard PCA (γ=1)**")
        st.dataframe(pca.variance_table().style.format("{:.2f}"), use_container_width=True)
    with col_rp:
        st.write(f"**RP-PCA (γ={rppca.gamma_used_:.1f})**")
        st.dataframe(rppca.variance_table().style.format("{:.2f}"), use_container_width=True)

    st.plotly_chart(
        plot_explained_variance(
            {"PCA (γ=1)": pca.variance_table(), f"RP-PCA (γ={rppca.gamma_used_:.0f})": rppca.variance_table()}
        ),
        use_container_width=True,
        config=_ST_PLOTLY_CONFIG,
    )

    # ── Factor Sharpe ─────────────────────────────────────────────────
    st.subheader("Factor-Level Sharpe Ratios")
    pca_fs = pca.factor_sharpe()
    rp_fs = rppca.factor_sharpe()

    col1, col2 = st.columns(2)
    with col1:
        st.write("**PCA Factors**")
        st.dataframe(pca_fs.style.format("{:.3f}"), use_container_width=True)
    with col2:
        st.write("**RP-PCA Factors**")
        st.dataframe(rp_fs.style.format("{:.3f}"), use_container_width=True)

    st.plotly_chart(
        plot_factor_sharpe(
            {"PCA": pca_fs, "RP-PCA": rp_fs}
        ),
        use_container_width=True,
        config=_ST_PLOTLY_CONFIG,
    )

    # ── Factor Loadings ───────────────────────────────────────────────
    st.subheader("Factor Loadings")
    tab_l1, tab_l2 = st.tabs(["PCA loadings", "RP-PCA loadings"])
    with tab_l1:
        st.plotly_chart(
            plot_factor_loadings(pca.loadings_, returns.columns.tolist(), "PCA"),
            use_container_width=True,
            config=_ST_PLOTLY_CONFIG,
        )
    with tab_l2:
        st.plotly_chart(
            plot_factor_loadings(rppca.loadings_, returns.columns.tolist(), "RP-PCA"),
            use_container_width=True,
            config=_ST_PLOTLY_CONFIG,
        )


# ---------------------------------------------------------------------------
# Tab 3: Portfolio
# ---------------------------------------------------------------------------
def tab_portfolio(cfg: Config):
    st.header("💼 Portfolio Construction")

    if st.session_state.pca_model is None:
        st.warning("Run In-Sample Analysis first.")
        return

    returns = st.session_state.returns
    X = returns.values
    pca: RPPCA = st.session_state.pca_model
    rppca: RPPCA = st.session_state.rppca_model

    # Factor returns
    F_pca = X @ pca.loadings_
    F_rp = X @ rppca.loadings_

    rf = cfg.portfolio.risk_free_rate
    td = cfg.portfolio.trading_days

    pc_pca = PortfolioConstructor(pca.loadings_, F_pca, risk_free_rate=rf, trading_days=td)
    pc_rp = PortfolioConstructor(rppca.loadings_, F_rp, risk_free_rate=rf, trading_days=td)

    # Build return series
    ret_dict = {
        "RP-PCA Tangency": pd.Series(pc_rp.tangency_returns(), index=returns.index),
        "RP-PCA Min-Var":  pd.Series(pc_rp.min_var_returns(), index=returns.index),
        "PCA Tangency":    pd.Series(pc_pca.tangency_returns(), index=returns.index),
        "PCA Min-Var":     pd.Series(pc_pca.min_var_returns(), index=returns.index),
        "Equal-Weight":    equal_weighted_returns(returns),
    }
    if "BTC" in returns.columns:
        ret_dict["BTC"] = returns["BTC"]

    # ── Metrics table ─────────────────────────────────────────────────
    st.subheader("In-Sample Performance Metrics")
    metrics = compute_metrics_table(
        {k: v.values for k, v in ret_dict.items()},
        risk_free_rate=rf,
        trading_days=td,
    )
    st.dataframe(_metrics_performance_styler(metrics), use_container_width=True)

    # ── Cumulative returns ────────────────────────────────────────────
    st.subheader("Cumulative Returns")
    log_scale = st.toggle("Log scale", value=False)
    st.plotly_chart(
        plot_cumulative_returns(ret_dict, log_scale=log_scale),
        use_container_width=True,
        config=_ST_PLOTLY_CONFIG,
    )

    # ── Efficient frontier ────────────────────────────────────────────
    st.subheader("Efficient Frontier")
    frontiers = {
        "RP-PCA factors": pc_rp.efficient_frontier(),
        "PCA factors": pc_pca.efficient_frontier(),
    }
    # Highlight portfolio points
    def _pt(pc, w_fn) -> tuple[float, float]:
        w = w_fn()
        F = pc.factor_returns
        r = (F @ w).mean() * td * 100
        v = np.sqrt((F @ w).var() * td) * 100
        return v, r

    portfolios = {
        "RP-PCA Tangency": _pt(pc_rp, pc_rp.tangency_weights),
        "RP-PCA Min-Var":  _pt(pc_rp, pc_rp.min_var_weights),
        "PCA Tangency":    _pt(pc_pca, pc_pca.tangency_weights),
        "PCA Min-Var":     _pt(pc_pca, pc_pca.min_var_weights),
    }
    st.plotly_chart(
        plot_efficient_frontier(frontiers, portfolios),
        use_container_width=True,
        config=_ST_PLOTLY_CONFIG,
    )


# ---------------------------------------------------------------------------
# Tab 4: OOS Backtest
# ---------------------------------------------------------------------------
def tab_backtest(cfg: Config):
    st.header("📈 Out-of-Sample Walk-Forward Backtest")

    if st.session_state.returns is None:
        st.warning("Load data first.")
        return

    returns = st.session_state.returns

    col1, col2, col3 = st.columns(3)
    col1.metric("Cov window", f"{cfg.backtest.cov_window}d")
    col2.metric("Mean window", f"{cfg.backtest.mean_window}d")
    col3.metric("Rebalance every", f"{cfg.backtest.rebalance_days}d")

    if st.button("▶ Run Walk-Forward Backtest", type="primary"):
        with st.spinner("Running walk-forward backtest… (this may take ~30 s)"):
            # Pass market caps for value-weighted benchmark (TAO only)
            vw_mcaps = st.session_state.get("tao_market_caps")
            bt = WalkForwardBacktest(
                returns=returns,
                cov_window=cfg.backtest.cov_window,
                mean_window=cfg.backtest.mean_window,
                rebalance_days=cfg.backtest.rebalance_days,
                n_components=cfg.model.n_components,
                gamma=cfg.model.gamma,
                cov_method=cfg.model.cov_method,
                ewma_cov_halflife=cfg.model.ewma_cov_halflife,
                ewma_mean_halflife=cfg.model.ewma_mean_halflife,
                risk_free_rate=cfg.portfolio.risk_free_rate,
                trading_days=cfg.portfolio.trading_days,
                target_gross_leverage=cfg.backtest.target_gross_leverage,
                concentrated_top_n=cfg.backtest.concentrated_top_n,
                buy_hold_benchmark=cfg.backtest.buy_hold_benchmark,
                vw_benchmark_mcaps=vw_mcaps if cfg.data.data_source == "tao_subnets" else None,
            )
            results = bt.run(include_pca=True, include_benchmarks=True)
            st.session_state.backtest_results = results
        st.success("Backtest complete!")

    if st.session_state.backtest_results is None:
        return

    results = st.session_state.backtest_results

    # Guard: if every strategy produced an empty return series, no rebalance
    # steps were taken (start_idx > T or data too short).
    if all(len(s) == 0 for s in results.return_series.values()):
        st.warning(
            "No backtest periods were produced — the backtest had no rebalance steps. "
            "This usually means the estimation window (cov_window or mean_window) is "
            "longer than the available data, or no full rebalance period fits within the "
            "loaded price history. Try shorter cov_window / mean_window values or load "
            "more data."
        )
        return

    # ── OOS metrics table ─────────────────────────────────────────────
    st.subheader("OOS Performance Metrics")
    m = results.metrics(cfg.portfolio.risk_free_rate, cfg.portfolio.trading_days)
    st.dataframe(_metrics_performance_styler(m), use_container_width=True)

    # ── Cumulative returns ────────────────────────────────────────────
    st.subheader("OOS Cumulative Returns")
    log_s = st.toggle("Log scale (OOS)", value=False)
    st.plotly_chart(
        plot_cumulative_returns(results.return_series, title="OOS Cumulative Returns", log_scale=log_s),
        use_container_width=True,
        config=_ST_PLOTLY_CONFIG,
    )

    # ── Rolling Sharpe ────────────────────────────────────────────────
    st.subheader("Rolling 63-Day Sharpe")
    st.plotly_chart(
        plot_rolling_sharpe(
            results.return_series,
            risk_free_rate=cfg.portfolio.risk_free_rate,
            trading_days=cfg.portfolio.trading_days,
        ),
        use_container_width=True,
        config=_ST_PLOTLY_CONFIG,
    )

    # ── Regime Analysis ──────────────────────────────────────────────
    st.subheader("Regime Analysis")
    st.caption(
        "Classifies each date as bull, bear, sideways, or contagion based on "
        "rolling equal-weight return and cross-asset correlation."
    )
    if st.button("▶ Run Regime Analysis", key="btn_regime"):
        with st.spinner("Classifying regimes…"):
            raw_returns = st.session_state.returns
            regime_labels = classify_regimes(
                raw_returns,
                config=cfg.regime,
                trading_days=cfg.portfolio.trading_days,
            )
            sharpe_dict, regime_metrics_df = compute_regime_metrics(
                results.return_series,
                regime_labels,
                risk_free_rate=cfg.portfolio.risk_free_rate,
                trading_days=cfg.portfolio.trading_days,
            )
            st.session_state["regime_labels"] = regime_labels
            st.session_state["regime_sharpe_dict"] = sharpe_dict
            st.session_state["regime_metrics_df"] = regime_metrics_df

    if st.session_state.get("regime_labels") is not None:
        regime_labels = st.session_state["regime_labels"]
        sharpe_dict = st.session_state["regime_sharpe_dict"]
        regime_metrics_df = st.session_state["regime_metrics_df"]

        # Regime timeline strip
        st.plotly_chart(
            plot_regime_timeline(regime_labels),
            use_container_width=True,
            config=_ST_PLOTLY_CONFIG,
        )

        # Regime day counts
        counts = regime_labels.value_counts()
        cols_regime = st.columns(len(counts))
        for col, (regime, cnt) in zip(cols_regime, counts.items()):
            pct = cnt / len(regime_labels) * 100
            col.metric(regime.capitalize(), f"{cnt} days", f"{pct:.1f}%")

        # Sharpe comparison bar chart
        st.plotly_chart(
            plot_regime_comparison(sharpe_dict),
            use_container_width=True,
            config=_ST_PLOTLY_CONFIG,
        )

        # Detailed metrics table — flatten MultiIndex for cleaner display
        display_df = regime_metrics_df.reset_index()
        display_df.columns = [c.replace("_", " ").title() if c != "N_obs" else "N Obs"
                              for c in display_df.columns]
        st.dataframe(
            display_df.style.format(
                {c: "{:.3f}" for c in display_df.select_dtypes("float").columns},
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ── Export to CSV ──────────────────────────────────────────────────
    st.subheader("Export to CSV")
    st.caption("Download backtest outputs for further analysis.")
    col1, col2, col3, col4 = st.columns(4)
    m = results.metrics(cfg.portfolio.risk_free_rate, cfg.portfolio.trading_days)
    with col1:
        st.download_button(
            label="Returns (daily)",
            data=results.returns_df().to_csv(),
            file_name="backtest_returns.csv",
            mime="text/csv",
            key="dl_backtest_returns",
        )
    with col2:
        st.download_button(
            label="Cumulative returns",
            data=results.cumulative().to_csv(),
            file_name="backtest_cumulative.csv",
            mime="text/csv",
            key="dl_backtest_cumulative",
        )
    with col3:
        st.download_button(
            label="Metrics",
            data=m.to_csv(),
            file_name="backtest_metrics.csv",
            mime="text/csv",
            key="dl_backtest_metrics",
        )
    with col4:
        if results.rebalance_log:
            st.download_button(
                label="Rebalance log (factor weights)",
                data=results.rebalance_log_df().to_csv(index=False),
                file_name="backtest_rebalance_log.csv",
                mime="text/csv",
                key="dl_backtest_rebalance",
            )
        else:
            st.caption("No rebalance log")

    # ── Rebalance history: asset weights (same format as Best Config → signals CSV) ──
    st.subheader("Rebalance history (asset weights)")
    st.caption(
        "One row per rebalance date, one column per asset — target **asset-space** weights "
        "(same as **Generate signals → Download signals CSV** on the Best Config tab). "
        "The file above labelled *factor weights* stores RP-PCA factor weights (tan_w0…), not per-asset weights."
    )
    oos_sig_strategy = st.selectbox(
        "Strategy for asset-weight export",
        STRATEGIES_TO_CAPTURE,
        index=0,
        key="oos_rebalance_asset_strategy",
    )
    oos_sig_df = results.signals_df(oos_sig_strategy)
    if oos_sig_df.empty:
        st.info(
            f"No asset weights recorded for **{oos_sig_strategy}** "
            f"(e.g. concentrated strategies need **concentrated_top_n** > 0 in config, or the backtest produced no steps)."
        )
    else:
        st.dataframe(oos_sig_df.head(10), use_container_width=True)
        st.download_button(
            label="Download rebalance history (asset weights CSV)",
            data=oos_sig_df.to_csv(index=False),
            file_name=f"oos_signals_{oos_sig_strategy.replace(' ', '_').replace('/', '-')}.csv",
            mime="text/csv",
            key="dl_oos_signals_asset_weights",
        )


# ---------------------------------------------------------------------------
# Tab 5: Best Configs (parameter sweep)
# ---------------------------------------------------------------------------
def tab_best_configs(cfg: Config):
    st.header("Best Configurations — Parameter Sweep")

    if st.session_state.returns is None:
        st.warning("Load data first (Data tab).")
        return

    returns = st.session_state.returns

    # ── Grid summary ──────────────────────────────────────────────────
    bc = cfg.backtest
    mc = cfg.model

    with st.expander("Parameter grids (edit in config.py)", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.write(f"**cov_window** {bc.cov_window_grid}")
        col1.write(f"**mean_window** {bc.mean_window_grid}")
        col2.write(f"**rebalance_days** {bc.rebalance_days_grid}")
        col2.write(f"**n_components** {mc.n_components_grid}")
        col3.write(f"**gamma** {mc.gamma_grid + [None]}")
        col3.write(f"**cov_method** {mc.cov_method_grid}")

    # Estimate total scenario count (approximate — before mean_window filter)
    from itertools import product as _product
    all_combos = list(_product(
        bc.cov_window_grid, bc.mean_window_grid, bc.rebalance_days_grid,
        mc.n_components_grid, mc.gamma_grid + [None], mc.cov_method_grid,
    ))
    filtered = sum(1 for cw, mw, *_ in all_combos if mw <= cw)
    st.caption(
        f"Estimated scenarios: **{filtered}** "
        f"(after filtering mean_window ≤ cov_window).  "
        f"Typical runtime: {filtered // 10}–{filtered // 5} minutes."
    )

    # ── Controls ──────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    target_strategy = c1.selectbox(
        "Optimise for strategy",
        STRATEGIES_TO_CAPTURE,
        index=0,
        key="sweep_target_strategy",
    )
    sort_metric = c2.selectbox(
        "Ranked by metric",
        METRICS_TO_CAPTURE,
        index=0,
        key="sweep_sort_metric",
    )

    # ── Run button ────────────────────────────────────────────────────
    if st.button("▶ Run Parameter Sweep", type="primary"):
        st.session_state.grid_sweep_results = None
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        collected_rows: list[dict] = []

        for completed, total, params, row in run_parameter_sweep(
            returns,
            cfg,
            include_pca=True,
            include_benchmarks=False,
        ):
            collected_rows.append(row)
            pct = completed / total
            progress_bar.progress(pct)
            status_text.caption(
                f"Scenario {completed} / {total} — "
                f"cov={params['cov_window']}d  mean={params['mean_window']}d  "
                f"reb={params['rebalance_days']}d  K={params['n_components']}  "
                f"γ={params['gamma']}  {params['cov_method']}"
            )

        progress_bar.progress(1.0)
        status_text.caption(f"Sweep complete — {len(collected_rows)} scenarios evaluated.")
        st.session_state.grid_sweep_results = build_sweep_dataframe(collected_rows)
        st.success(f"Done! {len(collected_rows)} scenarios evaluated.")

    # ── Results ───────────────────────────────────────────────────────
    if st.session_state.grid_sweep_results is None:
        st.info("Run the parameter sweep to see results.")
        return

    sweep_df: pd.DataFrame = st.session_state.grid_sweep_results

    # Metric column for the selected strategy
    selected_col = metric_col(target_strategy, sort_metric)
    ascending = sort_metric in {"Max DD (%)", "Ann. Vol (%)"}

    try:
        top5 = top_n_configs(sweep_df, target_strategy, sort_metric, n=5)
    except KeyError:
        st.error(f"No results found for {target_strategy} / {sort_metric}.")
        return

    # ── Top-5 table ───────────────────────────────────────────────────
    st.subheader(f"Top 5 configurations — {target_strategy} by {sort_metric}")

    best_val = top5[selected_col].iloc[0]
    direction = "lowest" if ascending else "highest"
    st.caption(
        f"{direction.capitalize()} {sort_metric}: **{best_val:.3f}** — "
        f"cov_window={top5['cov_window'].iloc[0]}d  "
        f"mean_window={top5['mean_window'].iloc[0]}d  "
        f"rebalance_days={top5['rebalance_days'].iloc[0]}d  "
        f"K={top5['n_components'].iloc[0]}  "
        f"γ={top5['gamma'].iloc[0]}  "
        f"{top5['cov_method'].iloc[0]}"
    )

    # Show param cols + all metrics for the target strategy
    strat_prefix = {
        "RP-PCA Tangency": "rp_tan",
        "RP-PCA Min-Var":  "rp_mv",
        "PCA Tangency":    "pca_tan",
        "PCA Min-Var":     "pca_mv",
        "alpha_concentrated_RP-PCA-Tangency": "alpha_conc_rp_tan",
        "alpha_concentrated_RP-PCA-Min-Var":  "alpha_conc_rp_mv",
        "alpha_concentrated_PCA-Tangency":    "alpha_conc_pca_tan",
        "alpha_concentrated_PCA-Min-Var":     "alpha_conc_pca_mv",
    }.get(target_strategy, "")
    strat_metric_cols = [c for c in top5.columns if c.startswith(strat_prefix)]
    display_cols = PARAM_COLS + strat_metric_cols

    # Rename metric columns to be more readable
    rename_map = {
        metric_col(target_strategy, m): m for m in METRICS_TO_CAPTURE
    }
    display_df = top5[display_cols].rename(columns=rename_map)

    st.dataframe(
        display_df.style.highlight_max(
            subset=[sort_metric] if not ascending else [],
            color="#1a7a4a",
        ).highlight_min(
            subset=[sort_metric] if ascending else [],
            color="#1a7a4a",
        ),
        use_container_width=True,
    )

    # ── Full sweep download ───────────────────────────────────────────
    st.subheader("Full sweep results")
    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        st.download_button(
            label="Download all scenarios (CSV)",
            data=sweep_df.to_csv(index=False),
            file_name="parameter_sweep_results.csv",
            mime="text/csv",
            key="dl_sweep_full",
        )
    with col_dl2:
        st.caption(
            f"{len(sweep_df)} scenarios × {len(sweep_df.columns)} columns.  "
            "Use this for custom analysis in Python / Excel."
        )

    # ── Scatter overview ──────────────────────────────────────────────
    if selected_col in sweep_df.columns:
        st.subheader(f"Distribution of {sort_metric} — {target_strategy}")
        try:
            import plotly.express as px
            valid = sweep_df.dropna(subset=[selected_col])
            fig = px.histogram(
                valid,
                x=selected_col,
                nbins=40,
                color="cov_method",
                barmode="overlay",
                opacity=0.75,
                labels={selected_col: sort_metric},
                title=f"{sort_metric} distribution across all scenarios ({target_strategy})",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True, config=_ST_PLOTLY_CONFIG)
        except Exception:
            pass  # chart is optional

    # ── Save top config ───────────────────────────────────────────────
    st.subheader("Save top config")
    best_row = top5.iloc[0]
    best_metric_col = metric_col(target_strategy, sort_metric)
    best_config_data = {
        "cov_window":       int(best_row["cov_window"]),
        "mean_window":      int(best_row["mean_window"]),
        "rebalance_days":   int(best_row["rebalance_days"]),
        "n_components":     int(best_row["n_components"]),
        "gamma":            best_row["gamma"],
        "cov_method":       best_row["cov_method"],
        "target_strategy":  target_strategy,
        "sort_metric":      sort_metric,
        best_metric_col:    best_row[best_metric_col] if best_metric_col in best_row else float("nan"),
    }
    best_config_df = pd.DataFrame([best_config_data])
    best_config_path = RESULTS_DIR / "best_config.csv"

    col_save1, col_save2 = st.columns([1, 3])
    with col_save1:
        if st.button("Save top config to CSV", key="btn_save_best_config"):
            best_config_df.to_csv(best_config_path, index=False)
            st.success(f"Saved to {best_config_path}")
    with col_save2:
        st.download_button(
            label="Download top config",
            data=best_config_df.to_csv(index=False),
            file_name="best_config.csv",
            mime="text/csv",
            key="dl_best_config",
        )

    # ── Generate signals from best config ─────────────────────────────
    st.subheader("Generate signals from best config")

    if not best_config_path.exists():
        st.info("Save the top config first (click 'Save top config to CSV' above).")
    elif st.session_state.returns is None:
        st.warning("Load data first (Data tab).")
    else:
        if st.button("Generate signals (run backtest with best config)", key="btn_gen_signals"):
            saved_cfg = pd.read_csv(best_config_path).iloc[0]
            raw_gamma = saved_cfg.get("gamma", None)
            gamma_val = None if (raw_gamma is None or str(raw_gamma).strip() in ("None", "nan", "")) else float(raw_gamma)
            _cov_w = int(saved_cfg["cov_window"])
            _mean_w = int(saved_cfg["mean_window"])
            _reb_days = int(saved_cfg["rebalance_days"])
            _T = len(st.session_state.returns)
            # Match WalkForwardBacktest.run(): shrink windows when the panel is short
            # so saved sweep configs (e.g. cov_window=504) still produce signals.
            _max_train = max(2, _T - 2)
            _eff_cov = min(_cov_w, _max_train)
            _eff_mean = min(_mean_w, _max_train)
            _start_idx = max(_eff_cov, _eff_mean)
            _n_steps = len(range(_start_idx, _T - 1, _reb_days))
            if (_eff_cov, _eff_mean) != (_cov_w, _mean_w):
                st.info(
                    f"Data has **{_T}** rows — using adaptive windows "
                    f"(cov_window {_cov_w}→{_eff_cov}, mean_window {_mean_w}→{_eff_mean}), "
                    f"same as the walk-forward backtest engine. "
                    f"Re-sweep with shorter grids if you need an exact match to the saved CSV."
                )
            if _n_steps == 0:
                st.error(
                    f"Cannot generate signals — even with adaptive windows, need enough "
                    f"history for at least one rebalance. "
                    f"(start index {_start_idx} from eff. cov/mean max, **{_T}** rows, "
                    f"_T−1={_T - 1}.) "
                    f"Load more history or save a config with shorter cov_window / mean_window."
                )
            else:
                sig_bt = WalkForwardBacktest(
                    returns=st.session_state.returns,
                    cov_window=_eff_cov,
                    mean_window=_eff_mean,
                    rebalance_days=_reb_days,
                    n_components=int(saved_cfg["n_components"]),
                    gamma=gamma_val,
                    cov_method=str(saved_cfg["cov_method"]),
                    ewma_cov_halflife=cfg.model.ewma_cov_halflife,
                    ewma_mean_halflife=cfg.model.ewma_mean_halflife,
                    risk_free_rate=cfg.portfolio.risk_free_rate,
                    trading_days=cfg.portfolio.trading_days,
                    target_gross_leverage=cfg.backtest.target_gross_leverage,
                    concentrated_top_n=cfg.backtest.concentrated_top_n,
                    buy_hold_benchmark=cfg.backtest.buy_hold_benchmark,
                )
                with st.spinner("Running backtest with best config to generate signals…"):
                    sig_results = sig_bt.run(include_pca=True, include_benchmarks=False)

                sig_strategy = str(saved_cfg.get("target_strategy", target_strategy))
                sig_df = sig_results.signals_df(sig_strategy)

                if sig_df.empty:
                    st.warning(
                        "No rebalance dates were produced — the backtest had no steps. "
                        "Often this means the estimation window (cov_window or mean_window) "
                        "is longer than your data, or the dataset is too short for at least "
                        "one full rebalance period. Try saving a config that uses shorter "
                        "cov_window / mean_window, or load more data."
                    )
                else:
                    signals_path = RESULTS_DIR / "signals.csv"
                    sig_df.to_csv(signals_path, index=False)
                    st.success(f"Signals saved to {signals_path}")
                    st.dataframe(sig_df.head(10), use_container_width=True)
                    st.download_button(
                        label="Download signals CSV",
                        data=sig_df.to_csv(index=False),
                        file_name="signals.csv",
                        mime="text/csv",
                        key="dl_signals",
                    )
                    # Persist for portfolio performance section below
                    st.session_state["signal_backtest_results"] = sig_results
                    st.session_state["signal_strategy"] = sig_strategy

        # ── Portfolio performance ($100,000) ──────────────────────────
        sig_res = st.session_state.get("signal_backtest_results")
        # Always use the currently-selected strategy dropdown so changing
        # "Optimise for strategy" above immediately updates this section
        # without re-running signals.
        sig_strat = target_strategy

        if sig_res is not None:
            st.subheader("Portfolio performance (starting $100,000)")
            st.caption(
                f"Strategy shown follows **Optimise for strategy** above — "
                f"currently: **{sig_strat}**. "
                "Change the dropdown to compare strategies without re-running."
            )
            import plotly.graph_objects as go

            # ── Cumulative portfolio value ─────────────────────────────
            cumul = sig_res.cumulative()
            if sig_strat not in cumul.columns:
                st.error(f"Strategy '{sig_strat}' not found in backtest results.")
            else:
                portfolio_value: pd.Series = 100_000 * cumul[sig_strat]

                fig_pv = go.Figure()
                fig_pv.add_trace(go.Scatter(
                    x=portfolio_value.index,
                    y=portfolio_value.values,
                    mode="lines",
                    name=sig_strat,
                    line=dict(width=2),
                ))
                start_val = portfolio_value.iloc[0]
                end_val   = portfolio_value.iloc[-1]
                pnl       = end_val - 100_000
                pnl_pct   = (end_val / 100_000 - 1) * 100
                fig_pv.update_layout(
                    title=f"Portfolio value — {sig_strat} (price action between rebalance dates)<br>"
                          f"<sub>Start: ${100_000:,.0f}  →  End: ${end_val:,.0f}  "
                          f"({'+'if pnl >= 0 else ''}{pnl_pct:.1f}%)</sub>",
                    xaxis_title="Date",
                    yaxis_title="Portfolio value ($)",
                    yaxis_tickprefix="$",
                    template="plotly_dark",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_pv, use_container_width=True, config=_ST_PLOTLY_CONFIG)

                # ── Key metrics ───────────────────────────────────────
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Starting capital", "$100,000")
                m_col2.metric("Ending value", f"${end_val:,.0f}")
                m_col3.metric("P&L", f"${pnl:+,.0f}", f"{pnl_pct:+.1f}%")
                m_col4.metric("Peak value", f"${portfolio_value.max():,.0f}")

                # ── Dollar allocations at each rebalance ──────────────
                st.markdown("**Dollar allocations at each rebalance date**")
                st.caption(
                    "Each cell = target weight × portfolio value on that rebalance date. "
                    "Positive = long, negative = short."
                )

                # Reload signals from session state (may have been generated in a prior run)
                cached_sig_df = sig_res.signals_df(sig_strat)
                if not cached_sig_df.empty:
                    reb_dates = pd.to_datetime(cached_sig_df["rebalance_date"])
                    pv_aligned = portfolio_value.reindex(
                        reb_dates, method="ffill"
                    ).values

                    asset_cols = [c for c in cached_sig_df.columns if c != "rebalance_date"]
                    dollar_alloc = cached_sig_df[asset_cols].multiply(pv_aligned, axis=0)
                    dollar_alloc.columns = [f"{c} ($)" for c in asset_cols]

                    alloc_df = pd.concat(
                        [
                            cached_sig_df[["rebalance_date"]].reset_index(drop=True),
                            pd.Series(pv_aligned, name="portfolio_value ($)").round(2),
                            dollar_alloc.round(2).reset_index(drop=True),
                        ],
                        axis=1,
                    )

                    _grad = [
                        c for c in alloc_df.columns
                        if "($)" in c and c != "portfolio_value ($)" and alloc_df[c].notna().any()
                    ]
                    _asty = alloc_df.style.format(
                        {c: "${:,.0f}" for c in alloc_df.columns if "($)" in c}
                    )
                    if _grad:
                        _asty = _asty.background_gradient(
                            subset=_grad, cmap="RdYlGn", axis=None,
                        )
                    st.dataframe(_asty, use_container_width=True)

                    st.download_button(
                        label="Download dollar allocations CSV",
                        data=alloc_df.to_csv(index=False),
                        file_name="dollar_allocations.csv",
                        mime="text/csv",
                        key="dl_dollar_alloc",
                    )


# ---------------------------------------------------------------------------
# Tab 6: Best Features — Factor Selection & Backtest
# ---------------------------------------------------------------------------

def _score_factors(
    factor_sharpe_df: pd.DataFrame,
    sharpe_weight: float = 0.6,
    variance_weight: float = 0.4,
) -> pd.DataFrame:
    """
    Score each factor using a weighted combination of |Sharpe| and variance explained.

    Returns the input DataFrame with added 'score' and 'rank' columns.
    """
    df = factor_sharpe_df.copy()
    # Normalise both metrics to [0, 1]
    s = np.abs(df["ann_sharpe"].values)
    v = df["var_explained_%"].values
    s_norm = s / s.max() if s.max() > 1e-10 else s
    v_norm = v / v.max() if v.max() > 1e-10 else v
    df["score"] = sharpe_weight * s_norm + variance_weight * v_norm
    df["rank"] = df["score"].rank(ascending=False).astype(int)
    return df.sort_values("rank")


def _auto_select_factors(
    scored_df: pd.DataFrame,
    method: str,
    min_sharpe: float = 0.0,
    top_n: int = 3,
    cum_var_threshold: float = 80.0,
) -> list[int]:
    """
    Return 0-based factor indices based on the selection method.

    Methods
    -------
    "Positive Sharpe" : keep factors with ann_sharpe > min_sharpe
    "Top N by Score"  : keep top_n factors by composite score
    "Variance Threshold" : keep factors until cumulative variance > cum_var_threshold
    """
    # Map PC labels back to 0-based index
    pc_to_idx = {label: i for i, label in enumerate(scored_df.index)}

    if method == "Positive Sharpe":
        selected = scored_df[scored_df["ann_sharpe"] > min_sharpe].index
    elif method == "Top N by Score":
        selected = scored_df.head(top_n).index
    elif method == "Variance Threshold":
        # Use original PC order for cumulative variance
        original = scored_df.sort_index()
        cum = original["var_explained_%"].cumsum()
        # Keep factors up to (and including) the one crossing the threshold
        keep = cum <= cum_var_threshold
        # Always keep at least up to the crossing factor
        if not keep.all():
            first_cross = (~keep).idxmax()
            keep[first_cross] = True
        selected = original[keep].index
    else:
        selected = scored_df.head(top_n).index

    indices = sorted([pc_to_idx[pc] for pc in selected])
    return indices if indices else [0]  # always keep at least 1


def tab_best_features(cfg: Config):
    st.header("🎯 Best Features — Factor Selection & Backtest")
    st.caption(
        "Analyses RP-PCA and PCA factors, identifies the most valuable ones "
        "using a composite score (Sharpe × variance explained), then runs "
        "a walk-forward backtest using only the selected factors."
    )

    # ── Strategy Guide expander ────────────────────────────────────────
    with st.expander("📖 How to use this for sophisticated TAO subnet portfolios", expanded=False):
        st.markdown("""
### Building Outperforming TAO Subnet Portfolios with Factor Selection

**The core insight:** Not all risk factors are worth owning. RP-PCA extracts *K* latent factors
from the 128-subnet return panel, but some of those factors are noise — low Sharpe, low economic
content. By identifying and keeping only the *priced* factors (those with statistically meaningful
risk premia), you concentrate capital into the drivers of cross-sectional return differences,
which is exactly how systematic quant funds outperform.

---

#### 🔬 What each factor represents in the TAO ecosystem

Each principal component captures a **distinct pattern of co-movement** across subnets:

| Factor characteristic | Economic interpretation |
|---|---|
| **High variance explained + high Sharpe** | Broad "TAO beta" — systematic exposure to network growth. The TAO ecosystem's market factor. Keep this. |
| **Mid variance, positive Sharpe** | Sector rotation: compute-heavy subnets (image/video) vs. text-inference subnets tend to diverge. A genuine risk premium. Keep this. |
| **Low variance, positive Sharpe** | Idiosyncratic alpha source — small but real. Keep if Sharpe > 0.3 after costs. |
| **Negative Sharpe (any variance)** | Mean-reverting or noise factor — earns no premium in the cross-section. **Drop this.** |
| **High variance, near-zero Sharpe** | Pure volatility exposure with no reward. Holding this dilutes your portfolio's information ratio. **Drop this.** |

> **Key rule of thumb:** In a universe of 128 TAO subnets, typically 2–4 factors pass the
> positive-Sharpe filter. RP-PCA (with γ ≫ 1) finds these faster than standard PCA because it
> penalises zero-mean factors at the decomposition stage.

---

#### ⚙️ How to configure for best results

**Step 1 — Set γ in the sidebar.** Use a large γ (≥ 50) for TAO subnets. The subnet universe has
high dispersion in mean returns (some subnets 10×, others -80% YTD), so RP-PCA's mean-weighting
dramatically improves factor quality vs. plain PCA.

**Step 2 — Choose selection method:**
- **Positive Sharpe** → conservative, production-ready. Use for live portfolios.
- **Top N by Score** → balanced. Start with N=3 for TAO (typically captures >70% of variance with highest risk-adjusted returns).
- **Variance Threshold** → use 60–75% for TAO (lower than equities because subnet returns have fatter tails).

**Step 3 — Run backtest and check the Sharpe delta.** If "Selected Factors" Sharpe > "All Factors"
Sharpe, your selection is filtering genuine noise. A Sharpe improvement of 0.2+ is economically meaningful.

**Step 4 — Export returns and feed into your execution layer.** The downloaded returns CSV gives
you the exact daily P&L series. Use the rebalance log for position targets.

---

#### 📅 Practical trading workflow

```
Monthly (or on each rebalance_days cycle):
  1. Re-fit RP-PCA on rolling window (handled by the walk-forward engine)
  2. Re-run factor scoring → confirm factor selection hasn't changed
  3. If factor set changes: adjust positions gradually (avoid large single-day turnover)
  4. Execute at TAO subnet DEX prices; use TWAP over 3–6 hours for size >1% of ADV

Quarterly:
  5. Re-run this tab with updated data → refresh factor Sharpe table
  6. Check regime (OOS Backtest tab → Regime Analysis):
     - Bull: hold full factor portfolio, lean tangency
     - Contagion: reduce to min-var, cut concentration to top 2 factors
     - Sideways: prefer the lowest-variance selected factor only
  7. Run Fama-MacBeth tab → confirm factors still earn statistically significant risk premia (|t| > 2)
```

---

#### 📐 Position sizing from factor weights

The walk-forward engine outputs **asset-space weights** (L @ w_factor for each subnet).
For a TAO portfolio:

- **Gross leverage = 1.0** (default): weights sum to 1 in absolute value — fully invested, no leverage.
- Interpret positive weight as *long the subnet* (buy SN-X tokens / provide liquidity).
- Interpret negative weight as *short the subnet* (via perpetual on a DEX that supports it, or reduce existing exposure).
- Scale dollar amounts: `position_i ($) = NAV × w_i`. Example: $500k NAV, w_SN1=0.12 → $60k into SN1.

> **Transaction cost note:** TAO subnet liquidity is thin. For positions >$50k in a single subnet,
> model 1–3% round-trip slippage. Run the backtest with a conservative Sharpe target (≥0.5 after 2%
> cost drag) before going live.

---

#### 🏆 Benchmark hierarchy (what "outperform" means)

| Benchmark | Represents | Target to beat |
|---|---|---|
| **TAO (flat/buy-hold)** | Simply holding TAO | Easy to beat in sideways/bear subnet markets |
| **Equal-Weight subnets** | Naive diversification across all 128 subnets | The true baseline — beat this consistently to claim skill |
| **Value-Weight subnets** | Cap-weighted exposure (like a subnet index fund) | Beat this to justify active factor tilts |
| **All-Factors portfolio** | Your own RP-PCA without pruning | Beat this to justify *this tab's* factor selection step |

A robust TAO subnet strategy should beat **all four** in walk-forward tests across at least 2
full market cycles (bull + bear in the TAO ecosystem).
        """)

    # ── Prerequisites ──────────────────────────────────────────────────
    if st.session_state.returns is None:
        st.warning("Load data first (Data tab).")
        return
    if st.session_state.rppca_model is None or st.session_state.pca_model is None:
        st.warning("Run In-Sample Analysis first (In-Sample tab).")
        return

    returns = st.session_state.returns
    rppca: RPPCA = st.session_state.rppca_model
    pca: RPPCA = st.session_state.pca_model

    # ── 1. Factor-Level Analysis ───────────────────────────────────────
    st.subheader("1 · Factor-Level Analysis")
    st.caption(
        "Each row is a latent factor extracted from the subnet return panel.  "
        "**Ann. Sharpe > 0** = factor earns a positive risk premium in-sample — a necessary "
        "(though not sufficient) condition for inclusion in a live portfolio.  "
        "RP-PCA factors (left) should show higher Sharpe than PCA (right) because RP-PCA "
        "explicitly tilts loadings toward high-mean subnets via the γ·μμ′ penalty term."
    )

    rp_fs = rppca.factor_sharpe(trading_days=cfg.portfolio.trading_days)
    pca_fs = pca.factor_sharpe(trading_days=cfg.portfolio.trading_days)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**RP-PCA factors**")
        st.dataframe(
            rp_fs.style.format("{:.3f}").background_gradient(
                subset=["ann_sharpe"], cmap="RdYlGn", axis=None,
            ),
            use_container_width=True,
        )
    with c2:
        st.markdown("**PCA factors**")
        st.dataframe(
            pca_fs.style.format("{:.3f}").background_gradient(
                subset=["ann_sharpe"], cmap="RdYlGn", axis=None,
            ),
            use_container_width=True,
        )

    # ── 2. Scoring & Selection ─────────────────────────────────────────
    st.subheader("2 · Factor Scoring & Selection")
    st.caption(
        "Each factor receives a composite score: "
        "**score = w₁ · |Sharpe|_norm + w₂ · VarExplained_norm**. "
        "Factors highlighted in green are selected for the portfolio backtest."
    )

    sc1, sc2, sc3, sc4 = st.columns(4)
    sharpe_w = sc1.slider("Sharpe weight", 0.0, 1.0, 0.6, 0.05, key="bf_sw")
    var_w = 1.0 - sharpe_w
    sc2.metric("Variance weight", f"{var_w:.2f}")

    model_choice = sc3.radio(
        "Model", ["RP-PCA", "PCA", "Both"], index=0, key="bf_model",
    )
    selection_method = sc4.selectbox(
        "Selection method",
        ["Positive Sharpe", "Top N by Score", "Variance Threshold"],
        index=1,
        key="bf_selection_method",
    )

    # Method-specific controls
    mc1, mc2 = st.columns(2)
    min_sharpe_val = 0.0
    top_n_val = 3
    cum_var_val = 80.0
    K = cfg.model.n_components

    if selection_method == "Positive Sharpe":
        min_sharpe_val = mc1.number_input(
            "Min Sharpe threshold", value=0.0, step=0.1, key="bf_min_sharpe",
        )
    elif selection_method == "Top N by Score":
        top_n_val = mc1.slider(
            "Top N factors", 1, K, min(3, K), key="bf_top_n",
        )
    elif selection_method == "Variance Threshold":
        cum_var_val = mc1.slider(
            "Cumulative variance threshold (%)", 30.0, 99.0, 80.0, 1.0,
            key="bf_cum_var",
        )

    # Score factors
    rp_scored = _score_factors(rp_fs, sharpe_weight=sharpe_w, variance_weight=var_w)
    pca_scored = _score_factors(pca_fs, sharpe_weight=sharpe_w, variance_weight=var_w)

    rp_selected = _auto_select_factors(
        rp_scored, selection_method,
        min_sharpe=min_sharpe_val, top_n=top_n_val, cum_var_threshold=cum_var_val,
    )
    pca_selected = _auto_select_factors(
        pca_scored, selection_method,
        min_sharpe=min_sharpe_val, top_n=top_n_val, cum_var_threshold=cum_var_val,
    )

    from rp_pca.analysis.plots import plot_factor_scorecard

    if model_choice in ("RP-PCA", "Both"):
        st.plotly_chart(
            plot_factor_scorecard(rp_scored, rp_selected, model_name="RP-PCA"),
            use_container_width=True,
            config=_ST_PLOTLY_CONFIG,
        )
        rp_labels = [rp_scored.index[i] for i in rp_selected]
        st.success(f"RP-PCA selected factors ({len(rp_selected)}): {', '.join(rp_labels)}")

    if model_choice in ("PCA", "Both"):
        st.plotly_chart(
            plot_factor_scorecard(pca_scored, pca_selected, model_name="PCA"),
            use_container_width=True,
            config=_ST_PLOTLY_CONFIG,
        )
        pca_labels = [pca_scored.index[i] for i in pca_selected]
        st.success(f"PCA selected factors ({len(pca_selected)}): {', '.join(pca_labels)}")

    # ── 3. Backtest with Selected Factors ──────────────────────────────
    st.subheader("3 · Walk-Forward Backtest — Selected vs All Factors")
    st.caption(
        "Runs a full out-of-sample walk-forward backtest using only the selected "
        "factors for portfolio construction, then compares to the all-factors baseline."
    )

    with st.expander("ℹ️ How to interpret the backtest results", expanded=False):
        st.markdown("""
**Walk-forward = gold-standard out-of-sample validation.**
At each rebalance date the model sees *only* past data — no look-ahead bias. This is the same
methodology used in institutional quant strategies.

| Metric | What it tells you | Good threshold for TAO subnets |
|---|---|---|
| **Ann. Sharpe** | Risk-adjusted return per unit of volatility | > 0.5 after estimated costs |
| **Ann. Geo. Ret (%)** | Compound annual growth — what you actually keep | > 30% given crypto risk premium |
| **Max DD (%)** | Worst peak-to-trough drawdown | < 60% (crypto has fat tails) |
| **Sortino** | Sharpe using only downside deviation | > 0.7 = good downside protection |

**Selected vs All Factors Sharpe delta:**
- Positive delta (green) → factor pruning removed noise, information ratio improved.
- Negative delta → the dropped factors were contributing return; consider loosening selection criteria or raising the Variance Threshold.

**Rolling Sharpe** tells you *when* the strategy works — look for periods where Selected outperforms
All during drawdowns (contagion/bear regimes). That is the hallmark of a genuinely better model:
it loses less when the market dislocates.
        """)


    if st.button("▶ Run Best-Features Backtest", type="primary", key="btn_bf_bt"):
        vw_mcaps = st.session_state.get("tao_market_caps")

        with st.spinner("Running all-factors backtest…"):
            bt_all = WalkForwardBacktest(
                returns=returns,
                cov_window=cfg.backtest.cov_window,
                mean_window=cfg.backtest.mean_window,
                rebalance_days=cfg.backtest.rebalance_days,
                n_components=cfg.model.n_components,
                gamma=cfg.model.gamma,
                cov_method=cfg.model.cov_method,
                risk_free_rate=cfg.portfolio.risk_free_rate,
                trading_days=cfg.portfolio.trading_days,
                target_gross_leverage=cfg.backtest.target_gross_leverage,
                buy_hold_benchmark=cfg.backtest.buy_hold_benchmark,
                vw_benchmark_mcaps=vw_mcaps,
                selected_factors=None,  # all factors
            )
            res_all = bt_all.run(include_pca=True, include_benchmarks=True)

        with st.spinner("Running selected-factors (RP-PCA) backtest…"):
            bt_sel = WalkForwardBacktest(
                returns=returns,
                cov_window=cfg.backtest.cov_window,
                mean_window=cfg.backtest.mean_window,
                rebalance_days=cfg.backtest.rebalance_days,
                n_components=cfg.model.n_components,
                gamma=cfg.model.gamma,
                cov_method=cfg.model.cov_method,
                risk_free_rate=cfg.portfolio.risk_free_rate,
                trading_days=cfg.portfolio.trading_days,
                target_gross_leverage=cfg.backtest.target_gross_leverage,
                buy_hold_benchmark=cfg.backtest.buy_hold_benchmark,
                vw_benchmark_mcaps=vw_mcaps,
                selected_factors=rp_selected,
            )
            res_sel = bt_sel.run(include_pca=True, include_benchmarks=True)

        st.session_state["bf_results_all"] = res_all
        st.session_state["bf_results_selected"] = res_sel
        st.session_state["bf_selected_factors"] = rp_selected
        st.session_state["bf_pca_selected_factors"] = pca_selected

    # ── 4. Display Results ─────────────────────────────────────────────
    res_all = st.session_state.get("bf_results_all")
    res_sel = st.session_state.get("bf_results_selected")

    if res_all is None or res_sel is None:
        st.info("Click the button above to run the backtest.")
        return

    sel_factors = st.session_state.get("bf_selected_factors", [])
    sel_labels = [f"PC{i+1}" for i in sel_factors]

    rf = cfg.portfolio.risk_free_rate
    td = cfg.portfolio.trading_days
    m_all = res_all.metrics(risk_free_rate=rf, trading_days=td)
    m_sel = res_sel.metrics(risk_free_rate=rf, trading_days=td)

    # ── Metrics comparison ─────────────────────────────────────────
    st.markdown("### Metrics Comparison")

    # Rename selected-factors strategies to distinguish
    m_sel_renamed = m_sel.copy()
    rename_map = {}
    for idx in m_sel_renamed.index:
        if "RP-PCA" in idx or "PCA" in idx:
            rename_map[idx] = f"{idx} (sel)"
    m_sel_renamed = m_sel_renamed.rename(index=rename_map)

    # Keep only the main strategies for a clean comparison
    core_strats_all = [s for s in ["RP-PCA Tangency", "RP-PCA Min-Var", "PCA Tangency", "PCA Min-Var"]
                       if s in m_all.index]
    core_strats_sel = [f"{s} (sel)" for s in core_strats_all if f"{s} (sel)" in m_sel_renamed.index]
    bench_strats = [s for s in m_all.index if s not in core_strats_all]

    combined = pd.concat([
        m_all.loc[core_strats_all],
        m_sel_renamed.loc[core_strats_sel],
        m_all.loc[bench_strats],
    ])
    st.dataframe(
        _metrics_performance_styler(combined),
        use_container_width=True,
    )

    # ── KPI delta cards ────────────────────────────────────────────
    rp_tan_all_sharpe = m_all.loc["RP-PCA Tangency", "Ann. Sharpe"] if "RP-PCA Tangency" in m_all.index else np.nan
    rp_tan_sel_sharpe = m_sel.loc["RP-PCA Tangency", "Ann. Sharpe"] if "RP-PCA Tangency" in m_sel.index else np.nan
    delta_sharpe = rp_tan_sel_sharpe - rp_tan_all_sharpe

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Factors used", f"{len(sel_factors)}/{K}", f"dropped {K - len(sel_factors)}")
    k2.metric(
        "All-Factors Sharpe",
        f"{rp_tan_all_sharpe:.3f}" if np.isfinite(rp_tan_all_sharpe) else "—",
    )
    k3.metric(
        "Selected Sharpe",
        f"{rp_tan_sel_sharpe:.3f}" if np.isfinite(rp_tan_sel_sharpe) else "—",
        f"{delta_sharpe:+.3f}" if np.isfinite(delta_sharpe) else None,
    )
    improvement = "✅ Selection improved Sharpe" if delta_sharpe > 0 else "⚠️ All factors performed better"
    k4.markdown(f"**Verdict**\n\n{improvement}" if np.isfinite(delta_sharpe) else "")

    # ── Cumulative returns ─────────────────────────────────────────
    st.markdown("### Cumulative Returns — Selected vs All Factors")

    # Build a combined return series dict for the comparison chart
    combined_rs: dict[str, pd.Series] = {}
    for name, series in res_all.return_series.items():
        if name in core_strats_all:
            combined_rs[f"{name} (all)"] = series
    for name, series in res_sel.return_series.items():
        if name in core_strats_all:
            combined_rs[f"{name} (sel)"] = series
    # Add benchmarks from all-factors run
    for name, series in res_all.return_series.items():
        if name not in core_strats_all:
            combined_rs[name] = series

    st.plotly_chart(
        plot_cumulative_returns(combined_rs, title="Selected Factors vs All Factors"),
        use_container_width=True,
        config=_ST_PLOTLY_CONFIG,
    )

    # ── Rolling Sharpe comparison ──────────────────────────────────
    st.markdown("### Rolling Sharpe — RP-PCA Tangency")
    rolling_rs: dict[str, pd.Series] = {}
    if "RP-PCA Tangency" in res_all.return_series:
        rolling_rs["All Factors"] = res_all.return_series["RP-PCA Tangency"]
    if "RP-PCA Tangency" in res_sel.return_series:
        rolling_rs["Selected Factors"] = res_sel.return_series["RP-PCA Tangency"]
    if rolling_rs:
        st.plotly_chart(
            plot_rolling_sharpe(
                rolling_rs,
                risk_free_rate=rf,
                trading_days=td,
                title="Rolling 63-Day Sharpe: All vs Selected Factors",
            ),
            use_container_width=True,
            config=_ST_PLOTLY_CONFIG,
        )

    # ── Portfolio performance ($100k) ──────────────────────────────
    st.markdown("### Portfolio Performance ($100,000)")

    import plotly.graph_objects as go

    strat_choice = st.selectbox(
        "Strategy to visualise",
        [s for s in core_strats_all if s in res_sel.return_series],
        index=0,
        key="bf_strat_choice",
    )

    if strat_choice:
        cum_all = res_all.cumulative()
        cum_sel = res_sel.cumulative()
        fig_pv = go.Figure()

        if strat_choice in cum_all.columns:
            pv_all = 100_000 * cum_all[strat_choice]
            fig_pv.add_trace(go.Scatter(
                x=pv_all.index, y=pv_all.values,
                mode="lines", name=f"{strat_choice} (all {K} factors)",
                line=dict(width=2, dash="dot"),
            ))

        if strat_choice in cum_sel.columns:
            pv_sel = 100_000 * cum_sel[strat_choice]
            fig_pv.add_trace(go.Scatter(
                x=pv_sel.index, y=pv_sel.values,
                mode="lines", name=f"{strat_choice} (selected: {', '.join(sel_labels)})",
                line=dict(width=2.5),
            ))
            end_val = pv_sel.iloc[-1]
            pnl = end_val - 100_000
            pnl_pct = (end_val / 100_000 - 1) * 100
        else:
            end_val = pnl = pnl_pct = np.nan

        fig_pv.update_layout(
            title=f"Portfolio Value — {strat_choice}",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            yaxis_tickprefix="$",
            template="plotly_dark",
            hovermode="x unified",
        )
        st.plotly_chart(fig_pv, use_container_width=True, config=_ST_PLOTLY_CONFIG)

        if np.isfinite(end_val):
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Starting capital", "$100,000")
            p2.metric("Ending value", f"${end_val:,.0f}")
            p3.metric("P&L", f"${pnl:+,.0f}", f"{pnl_pct:+.1f}%")
            p4.metric("Peak value", f"${pv_sel.max():,.0f}")

    # ── Export ─────────────────────────────────────────────────────
    st.subheader("Export")
    ec1, ec2 = st.columns(2)
    with ec1:
        sel_metrics_csv = m_sel.to_csv()
        st.download_button(
            "Download selected-factors metrics (CSV)",
            data=sel_metrics_csv,
            file_name="best_features_metrics.csv",
            mime="text/csv",
            key="dl_bf_metrics",
        )
    with ec2:
        sel_returns_csv = res_sel.returns_df().to_csv()
        st.download_button(
            "Download selected-factors returns (CSV)",
            data=sel_returns_csv,
            file_name="best_features_returns.csv",
            mime="text/csv",
            key="dl_bf_returns",
        )

    # ── Trading Playbook ───────────────────────────────────────────
    st.divider()
    st.subheader("🗺️ TAO Subnet Portfolio Trading Playbook")
    st.markdown("""
This section translates the factor selection output into a concrete, repeatable execution process
for building and managing a systematic TAO subnet portfolio.

---

#### Step 1 — Factor Discovery (this tab, monthly)

Run the **In-Sample** tab to fit the RP-PCA model on all available history.
Come here to score factors. A healthy setup looks like:

```
PC1: Sharpe ~1.2, VarExpl 38% → TAO network growth factor   ✅ Keep
PC2: Sharpe ~0.6, VarExpl 22% → compute vs inference rotation ✅ Keep
PC3: Sharpe ~0.3, VarExpl 14% → emerging subnet momentum     ✅ Keep (borderline)
PC4: Sharpe ~0.0, VarExpl  9% → mean-reverting noise          ❌ Drop
PC5: Sharpe -0.2, VarExpl  8% → pure volatility load          ❌ Drop
```

Typical TAO subnet result: **keep 2–3 factors, drop 2–3**. This gives you a
*concentrated factor portfolio* — the structural edge of RP-PCA over naive diversification.

---

#### Step 2 — Portfolio Construction (Best Configs tab, quarterly)

Use **Best Configs** to grid-search over (γ, cov_window, mean_window) *with* your
selected factor set. The optimal hyperparameter combination can shift between regimes.
Save the top config and regenerate signals.

Key hyperparameter intuitions for TAO:
- **Short mean_window (7–14 days):** TAO subnet sentiment rotates fast; a 2-week EWMA
  on means adapts quickly to new emission schedules or subnet launches.
- **Long cov_window (252 days):** Covariance structure is stickier; use a full year to
  avoid overfitting to a single bull or bear leg.
- **γ ≥ 50:** In a universe with >100 subnets and high cross-sectional return dispersion,
  a large γ is needed to give the mean vector enough weight vs. the large covariance matrix.

---

#### Step 3 — Risk Management (OOS Backtest → Regime Analysis tab)

Before sizing positions, classify the current regime:

| Regime | Action |
|---|---|
| **Bull** (rolling ann. ret > 20%, low correlation) | Full capital deployment, tangency weights, all selected factors |
| **Sideways** (ret between -20% and +20%) | 70–80% capital, mix tangency + min-var, trim lowest-Sharpe factor |
| **Bear** (rolling ann. ret < -20%) | 50% capital, min-var weights only, keep only PC1 (market factor) |
| **Contagion** (high cross-asset correlation) | 25–30% capital, min-var + TAO buy-hold, exit factor portfolio until correlation normalises |

---

#### Step 4 — Validation (Fama-MacBeth tab, quarterly)

Before increasing position sizes, confirm the statistical case:
- **Risk premia t-stat > 2.0** → the factor earns a *statistically significant* return
  in the cross-section of subnets. Shanken-corrected t-stats are conservative — prefer these.
- **Cross-sectional R² > 10%** → your factor betas actually explain subnet return differences.
  Low R² means the factor isn't pricing the cross-section — consider dropping it.

---

#### Step 5 — Execution Checklist

```
Pre-trade:
  □ Rebalance date confirmed (check rebalance_days setting)
  □ Download latest signals CSV from Best Configs tab
  □ Check current TAO/subnet DEX liquidity (target < 2% price impact per subnet)
  □ Confirm factor set hasn't changed since last rebalance

Execution:
  □ Calculate dollar allocations: NAV × w_asset_i for each subnet
  □ Net out existing positions to find trades (delta = target - current)
  □ Execute largest trades first (most liquid subnets) using TWAP
  □ Complete all trades within 4-hour window to minimise tracking error

Post-trade:
  □ Record actual fill prices vs. model close prices
  □ Update transaction cost log (feeds into future slippage estimates)
  □ Set alert for next rebalance_days days
```

---

> **Performance target:** A well-calibrated TAO subnet RP-PCA portfolio (2–3 selected factors,
> γ≈100, monthly rebalance) should target **Ann. Sharpe 0.8–1.5** over a full cycle, with
> **Max DD < 55%** — materially better than equal-weight subnet exposure (typically Sharpe 0.3–0.5,
> Max DD 70–85% in historical data).
    """)


# ---------------------------------------------------------------------------
# Tab 7: Fama-MacBeth Cross-Sectional Tests
# ---------------------------------------------------------------------------
def tab_fama_macbeth(cfg: Config):
    st.header("📐 Fama-MacBeth Cross-Sectional Tests")
    st.caption(
        "Two-pass Fama-MacBeth (1973) procedure: validates whether RP-PCA/PCA "
        "factors are *priced* risk factors earning statistically significant "
        "risk premia in the cross-section of asset returns.  "
        "Shanken (1992) correction applied to t-statistics by default."
    )

    if st.session_state.returns is None:
        st.warning("Load data first (Data tab).")
        return
    if st.session_state.pca_model is None:
        st.warning("Run In-Sample Analysis first (In-Sample tab).")
        return

    returns = st.session_state.returns
    X = returns.values
    pca: RPPCA = st.session_state.pca_model
    rppca: RPPCA = st.session_state.rppca_model

    col1, col2 = st.columns(2)
    split_frac = col1.slider(
        "Pass-1 split fraction",
        0.3, 0.8, cfg.fama_macbeth.split_frac, step=0.05,
        help="Fraction of sample for time-series beta estimation (pass 1). "
             "Remainder used for cross-sectional regressions (pass 2).",
    )
    shanken = col2.checkbox("Shanken correction", value=cfg.fama_macbeth.shanken_correction)

    if st.button("▶ Run Fama-MacBeth Tests", type="primary", key="btn_fm"):
        with st.spinner("Running Fama-MacBeth two-pass procedure…"):
            F_rp = X @ rppca.loadings_
            F_pca = X @ pca.loadings_
            rp_res, pca_res = compare_fama_macbeth(
                X, F_rp, F_pca,
                shanken_correction=shanken,
                split_frac=split_frac,
            )
            st.session_state["fm_rp_result"] = rp_res
            st.session_state["fm_pca_result"] = pca_res

    rp_res = st.session_state.get("fm_rp_result")
    pca_res = st.session_state.get("fm_pca_result")

    if rp_res is None:
        return

    # ── Summary tables ─────────────────────────────────────────────
    st.subheader("Risk Premia Summary")
    col_rp, col_pca = st.columns(2)
    with col_rp:
        st.write(f"**RP-PCA** (γ={rppca.gamma_used_:.0f})")
        rp_df = rp_res.summary_df()
        st.dataframe(rp_df, use_container_width=True)
        st.metric("Mean cross-sectional R²", f"{rp_res.mean_r2:.4f}")
    with col_pca:
        st.write("**PCA** (γ=1)")
        pca_df = pca_res.summary_df()
        st.dataframe(pca_df, use_container_width=True)
        st.metric("Mean cross-sectional R²", f"{pca_res.mean_r2:.4f}")

    # ── Risk premia comparison chart ───────────────────────────────
    st.subheader("Risk Premia Comparison")
    st.plotly_chart(
        plot_fama_macbeth_comparison([rp_res, pca_res]),
        use_container_width=True,
        config=_ST_PLOTLY_CONFIG,
    )

    # ── Cross-sectional R² over time ──────────────────────────────
    st.subheader("Cross-Sectional R² Over Time")
    st.plotly_chart(
        plot_cross_sectional_r2([rp_res, pca_res], dates=returns.index),
        use_container_width=True,
        config=_ST_PLOTLY_CONFIG,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    st.title("Risk-Premium PCA")
    st.markdown(
        "_Lettau & Pelger (2020) RP-PCA applied to 30 USDT cryptocurrency pairs_  "
        "| Kiefer & Nowotny (2025)"
    )

    cfg = render_sidebar()

    tab_names = [
        "📥 Data",
        "🔬 In-Sample",
        "💼 Portfolio",
        "📈 OOS Backtest",
        "🔭 Best Configs",
        "🎯 Best Features",
        "📐 Fama-MacBeth",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        tab_data(cfg)
    with tabs[1]:
        tab_insample(cfg)
    with tabs[2]:
        tab_portfolio(cfg)
    with tabs[3]:
        tab_backtest(cfg)
    with tabs[4]:
        tab_best_configs(cfg)
    with tabs[5]:
        tab_best_features(cfg)
    with tabs[6]:
        tab_fama_macbeth(cfg)


if __name__ == "__main__":
    main()
