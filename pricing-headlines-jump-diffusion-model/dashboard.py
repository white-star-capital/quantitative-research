#!/usr/bin/env python3
"""
Streamlit dashboard for the jump-diffusion pricing agent.

Reads `output/market_history.json` (cross-snapshot index keyed by market_id) for
historical charts and tables, and `output/latest.json` for the current board.

Because market discovery is dynamic (the top-10 set changes hour to hour), the
per-market history is built across the UNION of all markets ever seen, while the
current board prioritizes latest.json.

Run:  streamlit run dashboard.py   (or `make start`, which launches it on :8501)
"""

import glob
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from json_store import MARKET_HISTORY_FILENAME, rebuild_market_history

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")

CONFIDENCE_MAP = {"Low": 1, "Medium": 2, "High": 3}


# ==================== DATA LOADING ====================

def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_raw_snapshots(output_dir: str):
    """Load snapshot metadata for current_ts and snapshot count.

    Returns (snapshots, current_ts). Snapshots are used only for counts and
    current-board timestamp resolution; history comes from market_history.json.
    """
    files = glob.glob(os.path.join(output_dir, "*.json"))
    by_created_at = {}
    current_ts = None

    for path in files:
        basename = os.path.basename(path)
        if basename in (MARKET_HISTORY_FILENAME,):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        created_at = (data.get("metadata") or {}).get("created_at")
        results = data.get("results") or []
        if not created_at or not results:
            continue

        if basename == "latest.json":
            current_ts = created_at

        if created_at not in by_created_at:
            by_created_at[created_at] = results

    snapshots = []
    for created_at, results in by_created_at.items():
        try:
            ts = datetime.fromisoformat(created_at)
        except ValueError:
            continue
        snapshots.append({"ts": ts, "created_at": created_at, "results": results})

    snapshots.sort(key=lambda s: s["ts"])

    if current_ts is None and snapshots:
        current_ts = snapshots[-1]["created_at"]

    return snapshots, current_ts


def _ensure_market_history(output_dir: str) -> None:
    """Rebuild market_history.json if missing or older than newest snapshot."""
    history_path = os.path.join(output_dir, MARKET_HISTORY_FILENAME)
    newest_snapshot_mtime = 0.0
    for path in glob.glob(os.path.join(output_dir, "pricing_results_*.json")):
        try:
            newest_snapshot_mtime = max(newest_snapshot_mtime, os.path.getmtime(path))
        except OSError:
            continue

    needs_rebuild = not os.path.isfile(history_path)
    if not needs_rebuild and newest_snapshot_mtime > 0:
        try:
            needs_rebuild = os.path.getmtime(history_path) < newest_snapshot_mtime
        except OSError:
            needs_rebuild = True

    if needs_rebuild:
        rebuild_market_history(output_dir)


def _flatten_history_to_dataframes(history: dict):
    """Flatten market_history.json into per-market and per-outcome DataFrames."""
    outcome_rows = []
    market_rows = []

    for market_id, market in (history.get("markets") or {}).items():
        slug = market.get("slug") or market_id
        question = market.get("question") or slug
        market_type = market.get("market_type", "binary")

        for obs in market.get("observations") or []:
            ts_str = obs.get("ts")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue

            analysis = obs.get("analysis") or {}
            market_rows.append({
                "ts": ts,
                "market_id": str(market_id),
                "slug": slug,
                "question": question,
                "market_type": market_type,
                "volume": _to_float(obs.get("volume")),
                "liquidity": _to_float(obs.get("liquidity")),
                "end_date": market.get("end_date"),
                "active": obs.get("active"),
                "closed": obs.get("closed"),
                "confidence": analysis.get("confidence"),
                "confidence_num": CONFIDENCE_MAP.get(analysis.get("confidence")),
                "news_count": analysis.get("news_articles_analyzed"),
                "regime": obs.get("regime"),
                "analysis_text": analysis.get("analysis"),
                "key_factors": analysis.get("key_factors") or [],
                "prediction": analysis.get("prediction"),
                "outcome_probabilities": analysis.get("outcome_probabilities"),
            })

            if market_type == "binary":
                pricing = obs.get("pricing") or {}
                market_price = _to_float((obs.get("current_prices") or {}).get("Yes"))
                fair_price = _to_float(pricing.get("fair_price_yes"))
                outcome_rows.append({
                    "ts": ts,
                    "market_id": str(market_id),
                    "slug": slug,
                    "question": question,
                    "market_type": market_type,
                    "outcome": "Yes",
                    "market_price": market_price,
                    "fair_price": fair_price,
                    "ai_probability": None,
                    "edge_pct": (fair_price - market_price) * 100,
                    "kelly_fraction": _to_float(pricing.get("kelly_fraction")),
                    "recommendation": pricing.get("recommendation"),
                })
            else:
                for op in obs.get("outcome_pricings") or []:
                    market_price = _to_float(op.get("market_price"))
                    fair_price = _to_float(op.get("fair_price"))
                    outcome_rows.append({
                        "ts": ts,
                        "market_id": str(market_id),
                        "slug": slug,
                        "question": question,
                        "market_type": market_type,
                        "outcome": op.get("outcome", "?"),
                        "market_price": market_price,
                        "fair_price": fair_price,
                        "ai_probability": _to_float(op.get("ai_probability"), None),
                        "edge_pct": (fair_price - market_price) * 100,
                        "kelly_fraction": _to_float(op.get("kelly_fraction")),
                        "recommendation": op.get("recommendation"),
                    })

    return pd.DataFrame(outcome_rows), pd.DataFrame(market_rows)


@st.cache_data(show_spinner=False)
def load_data(output_dir: str, cache_key: str):
    """Build DataFrames from market_history.json and resolve current snapshot ts."""
    _ensure_market_history(output_dir)
    snapshots, current_ts = _load_raw_snapshots(output_dir)

    history_path = os.path.join(output_dir, MARKET_HISTORY_FILENAME)
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    outcomes_df, markets_df = _flatten_history_to_dataframes(history)
    return outcomes_df, markets_df, current_ts, len(snapshots)


def _cache_key(output_dir: str) -> str:
    parts = []
    for path in sorted(glob.glob(os.path.join(output_dir, "*.json"))):
        try:
            parts.append(f"{os.path.basename(path)}:{os.path.getmtime(path):.0f}")
        except OSError:
            continue
    return "|".join(parts)


# ==================== UI HELPERS ====================

def _fmt_money(v) -> str:
    try:
        return f"${float(v):,.0f}"
    except (TypeError, ValueError):
        return "-"


def render_current_board(outcomes_df, markets_df, current_ts):
    """Table of the markets in the current (latest.json) snapshot."""
    st.subheader("📋 Current board")
    cur_markets = markets_df[markets_df["ts"] == pd.to_datetime(current_ts)]
    cur_outcomes = outcomes_df[outcomes_df["ts"] == pd.to_datetime(current_ts)]
    if cur_markets.empty:
        st.info("No current snapshot found.")
        return

    rows = []
    for _, m in cur_markets.iterrows():
        ops = cur_outcomes[cur_outcomes["market_id"] == m["market_id"]]
        if not ops.empty:
            best = ops.iloc[ops["edge_pct"].abs().argmax()]
            best_outcome = best["outcome"]
            best_edge = best["edge_pct"]
            best_rec = best["recommendation"]
        else:
            best_outcome, best_edge, best_rec = "-", None, "-"
        rows.append({
            "Market": m["question"],
            "Type": m["market_type"],
            "Top outcome (|edge|)": best_outcome,
            "Edge %": round(best_edge, 2) if best_edge is not None else None,
            "Signal": best_rec,
            "Confidence": m["confidence"],
            "Volume": m["volume"],
            "Liquidity": m["liquidity"],
            "End date": (m["end_date"] or "")[:10],
        })
    board = pd.DataFrame(rows).sort_values(
        by="Edge %", key=lambda s: s.abs(), ascending=False, na_position="last"
    )
    st.dataframe(
        board,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Volume": st.column_config.NumberColumn(format="$%d"),
            "Liquidity": st.column_config.NumberColumn(format="$%d"),
            "Edge %": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def _pivot(df, value_col):
    """Pivot a per-outcome frame to wide (index=ts, columns=outcome) for line charts."""
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot_table(index="ts", columns="outcome", values=value_col, aggfunc="last")
    return wide.sort_index()


def _fmt_ts(ts) -> str:
    return str(pd.to_datetime(ts))[:16]


def _render_analysis_timeline(m_hist: pd.DataFrame) -> None:
    """Render per-snapshot analysis, prediction, and key factors (oldest to newest)."""
    if m_hist.empty:
        st.info("No analysis history for this market.")
        return

    sorted_hist = m_hist.sort_values("ts")
    latest_ts = sorted_hist["ts"].max()
    rows = list(sorted_hist.iterrows())

    for row_index, (_, row) in enumerate(rows):
        confidence = row.get("confidence") or "Unknown"
        news_count = row.get("news_count")
        news_label = f"{int(news_count)} articles" if pd.notna(news_count) else "— articles"
        header = f"{_fmt_ts(row['ts'])} · {confidence} · {news_label}"
        is_latest = row["ts"] == latest_ts

        with st.expander(header, expanded=is_latest):
            probs = row.get("outcome_probabilities")
            if isinstance(probs, dict) and probs:
                st.markdown("**AI outcome probabilities**")
                prob_rows = [
                    {
                        "Outcome": outcome,
                        "Probability": (
                            f"{float(prob):.0%}" if prob is not None else "—"
                        ),
                    }
                    for outcome, prob in sorted(
                        probs.items(),
                        key=lambda item: -float(item[1] or 0),
                    )
                ]
                st.dataframe(pd.DataFrame(prob_rows), use_container_width=True, hide_index=True)
            elif row.get("prediction"):
                st.markdown(f"**Prediction:** {row['prediction']}")

            st.markdown("**Analysis**")
            st.write(row.get("analysis_text") or "_No analysis text._")

            key_factors = row.get("key_factors") or []
            if len(key_factors) > 0:
                st.markdown("**Key factors**")
                for factor in key_factors:
                    st.markdown(f"- {factor}")

        if row_index < len(rows) - 1:
            st.divider()


def render_market_detail(outcomes_df, markets_df, market_id):
    m_hist = markets_df[markets_df["market_id"] == market_id].sort_values("ts")
    o_hist = outcomes_df[outcomes_df["market_id"] == market_id].sort_values("ts")
    if m_hist.empty:
        st.warning("No data for this market.")
        return

    latest_m = m_hist.iloc[-1]
    st.subheader(latest_m["question"])
    st.caption(f"`{market_id}` · `{latest_m['slug']}` · {latest_m['market_type']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Volume", _fmt_money(latest_m["volume"]))
    c2.metric("Liquidity", _fmt_money(latest_m["liquidity"]))
    c3.metric("End date", (latest_m["end_date"] or "-")[:10])
    status = "closed" if latest_m["closed"] else ("active" if latest_m["active"] else "-")
    c4.metric("Status", status)
    st.caption(f"Snapshots covering this market: {m_hist['ts'].nunique()} · "
               f"regime: {latest_m['regime'] or '-'}")

    all_outcomes = list(o_hist["outcome"].unique())
    selected = all_outcomes
    if len(all_outcomes) > 1:
        latest_o = o_hist[o_hist["ts"] == o_hist["ts"].max()]
        top_default = (latest_o.reindex(latest_o["edge_pct"].abs()
                                        .sort_values(ascending=False).index)
                       ["outcome"].head(4).tolist())
        selected = st.multiselect(
            "Outcomes", options=all_outcomes,
            default=top_default or all_outcomes[:4],
        )
    o_sel = o_hist[o_hist["outcome"].isin(selected)] if selected else o_hist

    tab_price, tab_edge, tab_kelly, tab_ai, tab_hist = st.tabs(
        ["Price vs Fair", "Edge", "Kelly / Recommendation", "AI confidence", "History table"]
    )

    with tab_price:
        if len(all_outcomes) == 1:
            wide = o_sel.set_index("ts")[["market_price", "fair_price"]].sort_index()
            wide.columns = ["Market price", "Fair price"]
            st.line_chart(wide)
        else:
            st.caption("Market price (solid) vs our fair price, per outcome.")
            st.markdown("**Market price**")
            st.line_chart(_pivot(o_sel, "market_price"))
            st.markdown("**Our fair price**")
            st.line_chart(_pivot(o_sel, "fair_price"))

    with tab_edge:
        st.caption("Edge % = (our fair price − market price) × 100. Positive = underpriced.")
        st.line_chart(_pivot(o_sel, "edge_pct"))

    with tab_kelly:
        st.markdown("**Kelly fraction over time**")
        st.line_chart(_pivot(o_sel, "kelly_fraction"))
        st.markdown("**Recommendation history**")
        rec = (o_sel.pivot_table(index="ts", columns="outcome",
                                 values="recommendation", aggfunc="last")
               .sort_index())
        st.dataframe(rec, use_container_width=True)

    with tab_ai:
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**AI confidence (1=Low, 2=Medium, 3=High)**")
            conf = m_hist.set_index("ts")[["confidence_num"]].rename(
                columns={"confidence_num": "confidence"})
            st.line_chart(conf)
        with cb:
            st.markdown("**News articles analyzed**")
            news = m_hist.set_index("ts")[["news_count"]]
            st.line_chart(news)
        st.divider()
        st.markdown("**Analysis timeline**")
        _render_analysis_timeline(m_hist)

    with tab_hist:
        cols = ["ts", "market_id", "outcome", "market_price", "fair_price", "edge_pct",
                "kelly_fraction", "recommendation", "ai_probability"]
        table = o_hist[cols].sort_values(["ts", "outcome"])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"{market_id}_history.csv",
            mime="text/csv",
        )


# ==================== MAIN ====================

def main():
    st.set_page_config(page_title="Jump-Diffusion Pricing Dashboard",
                       page_icon="📈", layout="wide")
    st.title("📈 Jump-Diffusion Pricing — Market Track Record")

    if st.button("🔄 Refresh data"):
        st.cache_data.clear()

    if not os.path.isdir(OUTPUT_DIR):
        st.error(f"Output directory not found: {OUTPUT_DIR}")
        return

    outcomes_df, markets_df, current_ts, n_snapshots = load_data(
        OUTPUT_DIR, _cache_key(OUTPUT_DIR)
    )

    if markets_df.empty:
        st.warning("No snapshots found in output/. Run the agent (`make start`) first.")
        return

    cur_markets = markets_df[markets_df["ts"] == pd.to_datetime(current_ts)]
    hist_min = markets_df["ts"].min()
    hist_max = markets_df["ts"].max()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Markets (current)", cur_markets["market_id"].nunique())
    m2.metric("Snapshots", n_snapshots)
    m3.metric("Markets tracked (all-time)", markets_df["market_id"].nunique())
    m4.metric("Current snapshot", str(pd.to_datetime(current_ts))[:16])
    st.caption(f"History: {str(hist_min)[:16]} → {str(hist_max)[:16]}")

    st.divider()
    render_current_board(outcomes_df, markets_df, current_ts)

    st.divider()
    st.subheader("🔎 Market history")

    current_market_ids = set(cur_markets["market_id"])
    latest_by_market = (markets_df.sort_values("ts")
                          .groupby("market_id").last().reset_index())
    latest_by_market["label"] = latest_by_market.apply(
        lambda r: ("🟢 " if r["market_id"] in current_market_ids else "⚪ ") + r["question"],
        axis=1,
    )
    latest_by_market = latest_by_market.sort_values(
        by=["market_id"],
        key=lambda s: s.isin(current_market_ids).map({True: 0, False: 1}),
    )
    options = latest_by_market["market_id"].tolist()
    labels = dict(zip(latest_by_market["market_id"], latest_by_market["label"]))

    selected_market_id = st.selectbox(
        "Select a market (🟢 = in current snapshot)",
        options=options,
        format_func=lambda mid: labels.get(mid, mid),
    )
    if selected_market_id:
        render_market_detail(outcomes_df, markets_df, selected_market_id)


if __name__ == "__main__":
    main()
