"""
Selling the wait — Streamlit dashboard.

    streamlit run app.py

Live figures and book numbers from `src/anth_ipo_book.py` and
`src/anth_vol_surface.py`. Snapshot 16 September 2026.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

from anth_ipo_book import (
    FUNDING_ANNUAL,
    PAPER_LIQ_RALLY,
    PAPER_SCENARIOS,
    PERP_NOTIONAL,
    PHASE1_IV,
    SCENARIO_WEIGHTS,
    SPOT,
    SURFACE_2026_09_16,
    apply_early_listing_rule,
    approx_liquidation_rally,
    funding_over_days,
    hyperliquid_margin,
    phase_one_greeks,
    phase_two_book,
    weighted_expected_return,
)
from dashboard_charts import (
    attribution_fig,
    calendar_violations_table,
    delta_surface_3d_fig,
    density_fig,
    event_move_table,
    event_variance_fig,
    forward_vol_oct_nov,
    heatmap_fig,
    ks_surface_3d_fig,
    returns_fig,
    rr_fly_fig,
    smiles_fig,
    surface_quotes_table,
    term_structure_fig,
    weekend_fwd,
)

st.set_page_config(
    page_title="Selling the wait",
    page_icon="⏳",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

PAPER = "#F4F1EA"
INK = "#142033"
MUTED = "#5B6573"
GOLD = "#B8893A"


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {PAPER}; }}
        [data-testid="stSidebar"] {{ background: #EDE8DD; }}
        h1, h2, h3 {{ color: {INK} !important; letter-spacing: -0.02em; }}
        .block-container {{ padding-top: 1.4rem; max-width: 1280px; }}
        div[data-testid="stMetric"] {{
            background: white;
            border: 1px solid #D7DCE3;
            border-radius: 12px;
            padding: 0.85rem 1rem;
        }}
        div[data-testid="stMetric"] label {{
            color: {GOLD} !important;
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {INK} !important;
            font-weight: 600;
        }}
        .stDeployButton {{ display: none; }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def money(value: float, *, compact: bool = False) -> str:
    sign = "−" if value < 0 else ""
    magnitude = abs(value)
    if compact and magnitude >= 1000:
        return f"{sign}${magnitude / 1000:.1f}k"
    return f"{sign}${magnitude:,.0f}"


def signed_money(value: float, *, compact: bool = False) -> str:
    prefix = "+" if value > 0 else ("−" if value < 0 else "")
    magnitude = abs(value)
    if compact and magnitude >= 1000:
        body = f"${magnitude / 1000:.1f}k"
    else:
        body = f"${magnitude:,.0f}"
    return f"{prefix}{body}"


def pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def signed_pct(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}%}"


def pct_pts(value: float, digits: int = 2) -> str:
    """Kernel breakevens are already in percent points (3.40, not 0.034)."""
    return f"{value:.{digits}f}%"


def md_money(value: float, **kwargs) -> str:
    return money(value, **kwargs).replace("$", r"\$")


def md_signed_money(value: float, **kwargs) -> str:
    return signed_money(value, **kwargs).replace("$", r"\$")


CHART_REV = 2


@st.cache_data(show_spinner=False)
def cached_term_fig(rev: int = CHART_REV):
    del rev
    return term_structure_fig()


@st.cache_data(show_spinner=False)
def cached_event_fig(spot: float, rev: int = CHART_REV):
    del rev
    return event_variance_fig(spot)


@st.cache_data(show_spinner=False)
def cached_rr_fig():
    return rr_fly_fig()


@st.cache_data(show_spinner=False)
def cached_smiles(spot: float, labels: tuple[str, ...]):
    return smiles_fig(spot, labels)


@st.cache_data(show_spinner=False)
def cached_heatmap(spot: float):
    return heatmap_fig(spot)


@st.cache_data(show_spinner="Meshing interpolate_iv…")
def cached_ks_surface(spot: float):
    return ks_surface_3d_fig(spot)


@st.cache_data(show_spinner="Meshing PCHIP delta sheet…")
def cached_delta_surface():
    return delta_surface_3d_fig()


@st.cache_data(show_spinner=False)
def cached_density(spot: float, label: str):
    return density_fig(spot, label)


@st.cache_data(show_spinner=False)
def cached_returns(funding_on: bool):
    return returns_fig(funding_on)


@st.cache_data(show_spinner=False)
def cached_attribution(funding_on: bool):
    return attribution_fig(funding_on)


def plot(fig) -> None:
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def sidebar_params() -> dict:
    with st.sidebar:
        st.markdown("### Snapshot")
        st.caption("16 Sep 2026 · HIP-3 units, 1 = $1B cap")
        spot = st.number_input(
            "io:ANTH mid",
            min_value=500.0,
            max_value=5000.0,
            value=float(SPOT),
            step=10.0,
        )
        notional = st.number_input(
            "Perp notional ($)",
            min_value=100_000.0,
            max_value=10_000_000.0,
            value=float(PERP_NOTIONAL),
            step=100_000.0,
            format="%.0f",
        )
        st.markdown("### Phase one")
        iv = st.slider("Sold-put IV", min_value=0.40, max_value=1.00, value=float(PHASE1_IV), step=0.01)
        days = st.slider("Put tenor (days)", min_value=1, max_value=21, value=7)
        funding = st.slider(
            "Funding (annual)",
            min_value=0.0,
            max_value=0.60,
            value=float(FUNDING_ANNUAL),
            step=0.01,
        )
        leverage = st.select_slider("HL isolated leverage", options=[1.5, 2.0, 3.0, 6.0], value=2.0)
        st.markdown("### Phase two")
        listing_days = st.slider("Assumed listing (days from snapshot)", min_value=1, max_value=80, value=54)
        funding_on = st.toggle("Count 24% funding in returns", value=True)
        st.caption(
            "Never quote the weighted return without the 60/30/10 weights and the funding on/off pair."
        )
        if st.button("Reset to note"):
            st.session_state.clear()
            st.rerun()
    return {
        "spot": spot,
        "notional": notional,
        "iv": iv,
        "days": float(days),
        "funding": funding,
        "leverage": float(leverage),
        "listing_days": float(listing_days),
        "funding_on": funding_on,
    }


def overview_tab(params: dict, greeks, p2) -> None:
    st.markdown(
        "A two-phase book on Anthropic’s IPO: **sell variance while the S-1 is private, "
        "buy the listing window after it**. Both phases keep a short `io:ANTH` perp to collect funding."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot", f"{params['spot']:.0f}", help="HIP-3 price units. 2170 = $2.17T.")
    c2.metric("Oct 17 → Nov 21 fwd vol", pct(forward_vol_oct_nov()))
    c3.metric("Weekend fwd vol", pct(weekend_fwd()), help="Sep 20 → Sep 21. The only arb-shaped hole.")
    c4.metric(
        "Weighted expected return",
        signed_pct(weighted_expected_return(params["funding_on"])),
        help="60% in-window / 30% slip / 10% early.",
    )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Phase one · before the S-1**")
            st.markdown(
                f"""
- Short `io:ANTH` ({md_money(params['notional'])} notional)
- Sell 2× weekly ATM Strike puts at {pct(params['iv'], 0)} IV
- Δ ≈ 0 by put-call parity → short straddle financed on Hyperliquid
- Collect {pct(params['funding'], 0)} funding
- Kill: S-1 public, or 72h RV > IV
                """
            )
            st.caption(
                f"{abs(greeks.n_puts):.0f} puts · premium {md_money(greeks.premium, compact=True)} · "
                f"theta {md_signed_money(greeks.theta_per_day)}/day"
            )
    with right:
        with st.container(border=True):
            st.markdown("**Phase two · after the S-1**")
            st.markdown(
                f"""
- Keep the short perp
- Long Nov 21 ATM calls ({p2.n_call_far:.1f})
- Short Oct 17 ATM puts ({p2.n_put_near:.1f})
- Long {pct(p2.fwd_vol)} fwd vol vs SPCX 105–130% realized
- Listing < Oct 17: swap into Nov calls · listing > Nov 21: exit
                """
            )
            rule = apply_early_listing_rule(params["listing_days"])
            st.caption(f"Timing rule at day {params['listing_days']:.0f}: `{rule}`")

    st.markdown("##### Kill switches sit on the book, not in the footnote")
    k1, k2, k3 = st.columns(3)
    with k1:
        with st.container(border=True):
            st.markdown("**Phase one**")
            st.markdown("S-1 goes public.\n\n72-hour realized > implied.\n\nNo new short-vol weeks after either.")
    with k2:
        with st.container(border=True):
            st.markdown("**Phase two**")
            st.markdown(
                "Listing set before Oct 17: buy back Oct puts, add Nov calls.\n\nListing after Nov 21: exit."
            )
    with k3:
        with st.container(border=True):
            st.markdown("**Always**")
            st.markdown(
                "HL margin ratio at 2× isolated.\n\nSustained negative funding.\n\nStrike feed ≠ live `io:ANTH` mark."
            )


def _format_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    fmt = quotes.copy()
    fmt["ATM"] = quotes["ATM"].map(lambda x: f"{x:.1%}")
    fmt["25Δ RR"] = quotes["25Δ RR"].map(lambda x: f"{x * 100:+.1f}")
    fmt["25Δ fly"] = quotes["25Δ fly"].map(lambda x: f"{x * 100:+.1f}")
    fmt["25Δ call IV"] = quotes["25Δ call IV"].map(lambda x: f"{x:.1%}")
    fmt["25Δ put IV"] = quotes["25Δ put IV"].map(lambda x: f"{x:.1%}")
    fmt["K put"] = quotes["K put"].map(lambda x: f"{x:.0f}")
    fmt["K call"] = quotes["K call"].map(lambda x: f"{x:.0f}")
    return fmt


def term_tab(params: dict) -> None:
    plot(cached_term_fig())
    st.caption(
        "Front-end put-skew market; weekend strip is the cheap calendar. "
        "Weekday realized 59% sits under every listed tenor except Sep 21."
    )
    col_a, col_b = st.columns((1.15, 0.85))
    with col_a:
        plot(cached_event_fig(params["spot"]))
    with col_b:
        table = event_move_table(params["spot"])
        show = table.drop(columns=["Interpolated"]).copy()
        for col in ("ATM IV", "b = 45%", "b = 50%", "b = 60%"):
            show[col] = show[col].map(lambda x: f"{x:.1%}")
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.caption("Oct 31* is ATM interpolated in total variance, not a listed expiry.")
    plot(cached_rr_fig())
    st.markdown("##### Listed surface")
    st.dataframe(_format_quotes(surface_quotes_table(params["spot"])), hide_index=True, use_container_width=True)


def surface_tab(params: dict) -> None:
    listed = [e.label for e in SURFACE_2026_09_16]
    default = ["Sep 23", "Oct 17", "Nov 7", "Nov 21"]
    labels = st.multiselect("Smiles", listed, default=default)
    if labels:
        plot(cached_smiles(params["spot"], tuple(labels)))
        st.caption("Solid only between the 25Δ wing dots. Three quotes pin three points, not 10Δ wings.")
    heat, dens = st.columns((1.15, 0.85))
    with heat:
        plot(cached_heatmap(params["spot"]))
        st.caption("Dark trench near day 3–4 is the weekend forward (~59%). Do not smooth it.")
    with dens:
        density_label = st.selectbox("Density expiry", listed, index=listed.index("Nov 21"))
        plot(cached_density(params["spot"], density_label))
        st.caption("Negative fly → thinner wings than a 79% lognormal.")

    st.markdown("##### 3D surface")
    kind = st.radio(
        "Mesh",
        ["K/S · linear in total variance", "Call delta · PCHIP in w(T)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if kind.startswith("K/S"):
        plot(cached_ks_surface(params["spot"]))
        st.caption(
            "Mesh `interpolate_iv`, not a triangulation of the nine ATM quotes. "
            "Each tenor is clipped to its own 25Δ band. White dots are listed ATM."
        )
    else:
        plot(cached_delta_surface())
        st.caption(
            "Rectangle in call-delta × time. Calendar interpolation is PCHIP in total variance. "
            "Weekend trench stays. ATM pins at Δ = 0.50."
        )

    hits = calendar_violations_table(params["spot"])
    st.markdown("##### Calendar violations")
    if hits.empty:
        st.caption("None at the sampled moneyness grid.")
    else:
        show = hits.copy()
        show["K/S"] = show["K/S"].map(lambda x: f"{x:.2f}")
        show["Δw"] = show["Δw"].map(lambda x: f"{x:.5f}")
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.caption("Sep 20 → 21 on the wings is the snapshot, not a bug. ATM does not fall after Sep 23.")


def book_tab(params: dict, greeks, p2) -> None:
    st.markdown(
        f"Phase one greeks · {money(params['notional'])} short perp, "
        f"{abs(greeks.n_puts):.0f} weekly ATM puts, {pct(params['iv'], 0)} IV, spot {params['spot']:.0f}"
    )
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Premium", money(greeks.premium, compact=True))
    m2.metric("Theta / day", signed_money(greeks.theta_per_day))
    m3.metric("Vega / vol pt", signed_money(greeks.vega_per_vol_point))
    m4.metric("BE daily", pct_pts(greeks.breakeven_daily))
    m5.metric("Funding / day", signed_money(greeks.funding_per_day))
    m6.metric("Strike IM", money(greeks.margin_entry, compact=True))

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Dollar γ / 1%²", signed_money(greeks.dollar_gamma_per_pct2))
    b2.metric("BE hourly", pct_pts(greeks.breakeven_hourly))
    b3.metric("Strike MM", money(greeks.margin_maint, compact=True))
    b4.metric("Net Δ", f"{greeks.delta:.1f}")
    st.caption(
        f"Hourly BE {pct_pts(greeks.breakeven_hourly)} vs ANTH hourly σ 0.57%. "
        "Hedge on a delta band — do not clock-hedge."
    )

    st.markdown("##### Phase two legs")
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Fwd vol", pct(p2.fwd_vol))
    l2.metric("Far vega / vol pt", signed_money(p2.vega_per_vol_point_far))
    l3.metric("Far theta / day", signed_money(p2.theta_per_day_far))
    l4.metric("Net Δ", f"{p2.delta:.1f}")

    st.markdown("##### Capital and isolation")
    hl = hyperliquid_margin(params["notional"], params["leverage"])
    capital = hl + greeks.margin_entry
    liq = PAPER_LIQ_RALLY.get(params["leverage"], approx_liquidation_rally(params["leverage"]))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HL isolated IM", money(hl, compact=True))
    c2.metric("Strike entry IM", money(greeks.margin_entry, compact=True))
    c3.metric("Phase-one capital", money(capital, compact=True))
    c4.metric("Liq. rally", pct(liq, 0))
    st.caption(
        "Entropy is strict isolated and does not cross-margin with Strike. "
        "Run the perp at 2× or less and sweep Strike marks back daily."
    )

    st.markdown("##### Timing rule")
    rule = apply_early_listing_rule(params["listing_days"])
    labels = {
        "swap_near_to_far": "Listing before Oct 17 — buy back the short Oct puts, replace with Nov 21 calls.",
        "hold": "Listing inside the window — hold the calendar through Nov 21.",
        "exit_after_far": "Listing confirmed after Nov 21 — exit phase two.",
    }
    st.info(f"**{rule}.** {labels[rule]}")


def returns_tab(params: dict) -> None:
    plot(cached_returns(params["funding_on"]))
    plot(cached_attribution(params["funding_on"]))

    rows = []
    for s in PAPER_SCENARIOS:
        total = s.total if params["funding_on"] else s.total - s.funding
        funding = s.funding if params["funding_on"] else 0.0
        rows.append(
            {
                "Scenario": s.name,
                "Listing": s.listing,
                "Weight": SCENARIO_WEIGHTS[s.name],
                "Total": total,
                "5th": s.p5 if params["funding_on"] else s.p5 - s.funding,
                "95th": s.p95 if params["funding_on"] else s.p95 - s.funding,
                "Options": s.options,
                "Hedge": s.hedge,
                "Funding": funding,
                "Fees": s.fees,
            }
        )
    frame = pd.DataFrame(rows)
    show = frame.copy()
    for col in ("Total", "5th", "95th", "Options", "Hedge", "Funding", "Fees", "Weight"):
        show[col] = frame[col].map(lambda x: f"{x:.1%}")
    st.dataframe(show, hide_index=True, use_container_width=True)

    fund_65 = funding_over_days(65.0, params["funding"], params["notional"]) / params["notional"]
    r1, r2, r3 = st.columns(3)
    r1.metric("60/30/10 with funding", signed_pct(weighted_expected_return(True)))
    r2.metric("60/30/10 without funding", signed_pct(weighted_expected_return(False)))
    r3.metric("65-day funding", signed_pct(fund_65))
    st.caption(
        "The in-window cone reuses one SPCX path. An early listing will not let you buy November at today’s 79%. "
        "Every listed scenario is positive with funding; the weak case is a slip past November."
    )


def main() -> None:
    _inject_css()
    params = sidebar_params()
    greeks = phase_one_greeks(
        spot=params["spot"],
        notional=params["notional"],
        iv=params["iv"],
        days=params["days"],
        funding_annual=params["funding"],
    )
    p2 = phase_two_book(spot=params["spot"], notional=params["notional"])

    page = st.radio(
        "Section",
        ["Overview", "Term", "Surface", "Book", "Returns"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.title("Selling the wait, buying the listing")
    st.caption(
        "Anthropic IPO book across Hyperliquid `io:ANTH` and Strike ANTH options. "
        "Research snapshot 16 September 2026. Vols are 24/7-annualized."
    )
    if page == "Overview":
        overview_tab(params, greeks, p2)
    elif page == "Term":
        term_tab(params)
    elif page == "Surface":
        surface_tab(params)
    elif page == "Book":
        book_tab(params, greeks, p2)
    else:
        returns_tab(params)

    st.divider()
    st.caption(
        "Not a market-making reservation-price book. Not a PreStocks conversion. "
        "Kernel identities live in `src/anth_ipo_book.py` and `src/anth_vol_surface.py`."
    )


main()
