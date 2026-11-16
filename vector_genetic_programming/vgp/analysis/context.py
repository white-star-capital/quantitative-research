"""Market context for the dashboard: what the assets did while we measured them.

The run matrix reports a Sharpe per window and nothing about the market those
windows covered, so "+1.207, -1.759, -0.533" reads as three unexplained
numbers. The first question anyone asks is whether the good window was simply a
trending stretch. This writes the series needed to answer that.

Two series, both computed on the REALISED universe — the assets the run actually
used, not the ones it asked for, because a context chart drawn on a different
universe is not context for this run:

  index        equal-weight cumulative return of the retained assets, rebased
               to 1.0 at the first bar. Equal weight rather than cap weight
               because the strategies are cross-sectional and dollar-neutral:
               this is the benchmark they are implicitly measured against.

  breadth      rolling effective number of independent bets, from the
               participation ratio of the correlation matrix's eigenvalues.
               The headline figure (2.17 of 21 assets) is a full-sample number,
               and whether it holds or collapses in stress is the question a
               single figure cannot answer. Breadth vanishing exactly when it is
               needed is the standard way a diversified book fails.

Written as JSON by run.py so that scripts/build_dashboard.py — which is pure
standard library, because the viewing image carries no pandas or pyarrow — can
read it without the scientific stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Rolling window for the breadth estimate. 60 bars is long enough that the
# correlation matrix is not dominated by noise (21 assets need >> 21 samples)
# and short enough to show a regime turning over inside a 4-month OOS window.
BREADTH_WINDOW = 60


def effective_bets(corr: np.ndarray) -> float:
    """Participation ratio of the eigenvalue spectrum.

    1.0 when every asset is the same asset, N when all are independent. This is
    the quantity that makes "21 assets" and "2.17 bets" both true statements
    about the same panel.
    """
    ev = np.linalg.eigvalsh(corr)
    ev = ev[ev > 0]
    if ev.size == 0:
        return float("nan")
    return float((ev.sum() ** 2) / (ev**2).sum())


def market_context(
    ohlcv: dict[str, pd.DataFrame],
    assets: list[str],
    dates: pd.DatetimeIndex,
    window: int = BREADTH_WINDOW,
) -> dict:
    """Index and rolling breadth over `dates`, on `assets` only."""
    close = pd.DataFrame({t: ohlcv[t]["close"] for t in assets if t in ohlcv})
    close = close.sort_index().reindex(dates)
    ret = np.log(close).diff()

    # Equal-weight index: mean log return across whatever is listed that bar, so
    # an asset that lists mid-sample joins the index instead of voiding it.
    mean_ret = ret.mean(axis=1, skipna=True).fillna(0.0)
    index = np.exp(mean_ret.cumsum())

    breadth: list[float | None] = []
    arr = ret.to_numpy()
    for i in range(len(dates)):
        if i + 1 < window:
            breadth.append(None)
            continue
        block = arr[i + 1 - window : i + 1]
        # Keep only assets with a complete block; a column of NaNs would make
        # the correlation matrix singular and the eigenvalues meaningless.
        keep = np.isfinite(block).all(axis=0)
        if keep.sum() < 3:
            breadth.append(None)
            continue
        c = np.corrcoef(block[:, keep], rowvar=False)
        n = effective_bets(c)
        breadth.append(None if not np.isfinite(n) else round(n, 3))

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "index": [round(float(v), 5) for v in index],
        "breadth": breadth,
        "breadth_window": window,
        "n_assets": int(close.shape[1]),
        "assets": list(close.columns),
    }


def write_market_context(path: str | Path, **kwargs) -> Path:
    """Compute and write market_context.json. Returns the path written."""
    path = Path(path)
    path.write_text(json.dumps(market_context(**kwargs), separators=(",", ":")) + "\n")
    return path


def equity_band(
    observed_rows: list[dict],
    null_equity_runs: list[list[list[float]]],
    n_null: int,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> dict:
    """Per-window observed equity against the percentile envelope of the nulls.

    This is the histogram comparison drawn through time: the question is not
    whether the observed curve rises, but whether it leaves the band that
    signal-free data produces. A rising curve inside the band is what noise
    looks like over a few months, which is exactly why a bare equity chart is
    misleading and this one is not.

    Curves are stored as percentiles rather than as 99 raw series: the chart
    needs the envelope, and the raw set is two orders of magnitude larger for
    no extra information.

    `null_equity_runs` is one entry per surrogate run, each holding that run's
    per-window curves in window order — the shape run.py accumulates.
    """
    by_window: dict[int, list[float]] = {}
    for row in observed_rows:
        curve = row.get("oos_equity") or []
        if not curve:
            continue
        w = int(row["window_id"])
        # Several seeds per window: the median across seeds is the observed
        # line, matching the TYPICAL statistic the verdict is read from.
        by_window.setdefault(w, []).append(curve)

    windows = []
    for w in sorted(by_window):
        obs_curves = by_window[w]
        n = min(len(c) for c in obs_curves)
        if n < 2:
            continue
        obs = np.median(np.array([c[:n] for c in obs_curves], dtype=float), axis=0)

        # Null curves for the same window index, trimmed to a common length.
        nulls = [run[w][:n] for run in null_equity_runs if len(run) > w and len(run[w]) >= n]
        if len(nulls) < 5:
            continue  # too few to describe a band honestly
        arr = np.array(nulls, dtype=float)

        row = next(r for r in observed_rows if int(r["window_id"]) == w)
        windows.append(
            {
                "w": w,
                "dates": _dates_for(row, n),
                "observed": [round(float(v), 5) for v in obs],
                "p05": [round(float(v), 5) for v in np.percentile(arr, lo_pct, axis=0)],
                "p50": [round(float(v), 5) for v in np.percentile(arr, 50.0, axis=0)],
                "p95": [round(float(v), 5) for v in np.percentile(arr, hi_pct, axis=0)],
                "n_null_curves": int(arr.shape[0]),
            }
        )

    return {"windows": windows, "n_null": int(n_null), "lo_pct": lo_pct, "hi_pct": hi_pct}


def _dates_for(row: dict, n: int) -> list[str]:
    """Business-day labels spanning the row's OOS period, length n.

    The curve comes back without an index, so the labels are reconstructed from
    the window the row declares. They are axis annotation, not data: the chart
    plots position, and only the endpoints are shown.
    """
    start = pd.Timestamp(row["test_start"])
    end = pd.Timestamp(row["test_end"])
    idx = pd.date_range(start, end, periods=n)
    return [d.strftime("%Y-%m-%d") for d in idx]
