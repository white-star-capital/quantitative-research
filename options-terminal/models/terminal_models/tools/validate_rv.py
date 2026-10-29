"""Validate rv_har against its baselines on a recording (RV_FORECAST_IMPLEMENTATION_GUIDE.md §9).

    uv run python -m terminal_models.tools.validate_rv --recording <dir> --out <report.json> \\
        [--assets BTC,ETH] [--folds 8] [--z-threshold 1.0] [--bias-tolerance 0.10]

Runs only on a recording (tools/record_deribit.py) that includes rv_har: each asset's `rv_daily`
and, for BTC/ETH, `dvol_history` are replayed through the production adapters exactly as a run
fetches them. Nothing is fetched live and nothing is filled: a missing day stays missing.

Per asset, an expanding-origin backtest over every origin with `min_history_days[asset]` of prior
RV, split into `--folds` contiguous folds. Each fold refits on days ≤ its first origin and holds
that fit for the rest of the fold (§9: refit per fold, no refit inside one). Three forecasters,
all from core/quant/rv.py with the rv_har manifest params:
- `rv_har`: HAR + day-of-week + event-window dummies, median-adjusted path (as `signal`);
- `har`: plain HAR, no calendar terms, median-adjusted path (as `context.rv_har_30d`);
- `ewma`: EWMA(`ewma_lambda`) daily variance, flat over the horizon (as `context.rv_ewma_30d`).

Scores (MASE and RMSSE, in-sample naive scale from the days before the first origin):
- per horizon h = 1..`horizon_days` on log daily variance (the model target, §3), then averaged
  over horizons; a pair is in-event when its target day lies in an in-scope event window;
- per tenor in `tenors` on annualized RV over days t+1..t+T (§12.2), naive scale = the previous
  T-day window; in-event when any of those days lies in an event window.
- bias on ordinary (ex-event) target days, in annualized daily vol, pooled over horizons.

Signal P&L (VRP-A-10, BTC/ETH at 30d): a variance-swap PROXY, not tradeable P&L. At origin t, short
30-day variance struck at DVOL_t², P&L = DVOL_t² − RV_realized(t+1..t+30)² per unit variance
notional, jump days included. Entries are non-overlapping, every 30 days back from the last origin
whose window has matured, so each period's P&L is independent of the next; Sharpe =
mean/sd(ddof=1) · √(365/30). The rule is short when vrp_z > threshold, DVOL > RV^0.9(30) (band)
and comp_event(30) ≤ `event_hump_share`, else flat (0); naive is always short. vrp_z is scored as
in the model (§12.6) on this backtest's 30d forecasts.

Gates (§9 table): rv_har RMSSE ≤ (1 + 5%) × har ex-event; rv_har RMSSE < har in event windows;
|relative bias| ≤ `--bias-tolerance` on ordinary days (the guide sets no number); rule Sharpe > 0
and > naive Sharpe; every type in `events_listed_through` listed through the last day plus the
widest window (otherwise the in/ex-event splits mislabel unlisted event days). Each gate is pass,
fail, unavailable (not enough data: never a pass) or not_applicable (no DVOL basis). Exit 0 when
every asset passes every gate, 1 otherwise, 2 when the recording can't be used. The report is
written either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from terminal_models.adapters.base import AdapterError
from terminal_models.adapters.deribit import DeribitAdapter
from terminal_models.adapters.deribit.replay import RecordedClient, RecordingError
from terminal_models.adapters.hyperliquid import HyperliquidAdapter
from terminal_models.adapters.hyperliquid.replay import RecordedHyperliquidClient
from terminal_models.core.base import InputData, ModelError
from terminal_models.core.manifest import Manifest, load_manifest
from terminal_models.core.paths import library_dir
from terminal_models.core.quant.calendar import (
    events_for,
    incomplete_types,
    weekdays,
    window_dummies,
)
from terminal_models.core.quant.rv import (
    ANNUAL_DAYS,
    DAY_MS,
    as_array,
    cone_paths,
    counterfactuals,
    decompose,
    dow_dummies,
    ewma_var,
    fit_direct,
    mase,
    predict_path,
    residual_quantiles,
    rmsse,
    rv_term,
    trailing_rv,
    trailing_zscore,
)
from terminal_models.library.rv_har.model import DVOL_ASSETS, RV_WINDOW_DAYS, Z_TENOR, day_close_ms
from terminal_models.runner.execute import git_sha, load_model
from terminal_models.runner.snapshots import input_hash

IMPLEMENTATION = "rv_har"
SERIES = ("rv_daily", "dvol_history")  # the only inputs the backtest reads
MODELS = ("rv_har", "har", "ewma")
SPLITS = ("all", "ex_event", "in_event")
RMSSE_EX_EVENT_TOLERANCE = 0.05  # §9: within 5% of plain HAR ex-event
REPORT_VERSION = 1


@dataclass(frozen=True)
class Config:
    folds: int = 8
    z_threshold: float = 1.0
    bias_tolerance: float = 0.10


def _opt(x: Any) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def _close(days: list[str], i: int) -> str:
    """Day i's close as ISO UTC (§12.1)."""
    when = datetime.fromtimestamp(day_close_ms(days[i]) / 1000, UTC)
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def params_hash(params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


# ---- calendar and backtest ------------------------------------------------------------------


def build_calendar(
    asset: str, days: list[str], params: dict[str, Any], horizon: int
) -> tuple[np.ndarray, int, np.ndarray]:
    """(calendar, n_dow, event_day) over the input days plus `horizon` forecast days, built as
    rv_har's Forecast builds it. event_day[i] is True where some in-scope event window covers
    day i."""
    closes = [day_close_ms(d) for d in days]
    future = [closes[-1] + k * DAY_MS for k in range(1, horizon + 1)]
    windows = params["event_windows"]
    pad = timedelta(days=max(max(w) for w in windows.values()))
    start = datetime.combine(date.fromisoformat(days[0]), time(), UTC) - pad
    last = date.fromisoformat(days[-1]) + timedelta(days=1 + horizon)
    end = datetime.combine(last, time(), UTC) + pad
    events = events_for(asset, params["events"], start, end)
    dow = dow_dummies(weekdays(closes + future))
    _, evt = window_dummies(closes + future, events, windows)
    return np.hstack([dow, evt]), dow.shape[1], evt.any(axis=1)


@dataclass
class Backtest:
    origins: np.ndarray  # day indices
    paths: dict[str, np.ndarray]  # daily variance per model, (len(origins), horizon)
    rv_high_30: np.ndarray  # RV^0.9(30) of rv_har per origin
    event_share_30: np.ndarray  # comp_event(30) of rv_har per origin
    folds: list[dict[str, Any]] = field(default_factory=list)


def eligible_origins(y: np.ndarray, min_history: int) -> np.ndarray:
    """Origins with ≥ min_history non-null days up to and including them, and at least one day
    after them to score."""
    have = np.cumsum(np.isfinite(y))
    return np.array([t for t in range(len(y) - 1) if have[t] >= min_history], dtype=int)


def run_backtest(
    y: np.ndarray,
    rv: np.ndarray,
    calendar: np.ndarray,
    n_dow: int,
    origins: np.ndarray,
    params: dict[str, Any],
    folds: int,
) -> Backtest:
    """Expanding-origin forecasts at every origin; one fit per fold, made on days ≤ the fold's
    first origin (no look-ahead). A fold whose fit fails leaves its rows NaN and is reported."""
    if not 1 <= folds <= len(origins):
        raise ValueError(f"folds must be 1..{len(origins)} (eligible origins), got {folds}")
    H = int(params["horizon_days"])  # noqa: N806
    lags = tuple(int(x) for x in params["har_lags"])
    q = [float(x) for x in params["quantiles"]]
    i_mid, i_hi = q.index(0.5), q.index(max(q))
    paths = {m: np.full((len(origins), H), np.nan) for m in MODELS}
    hi30, evt30 = np.full(len(origins), np.nan), np.full(len(origins), np.nan)
    s2 = ewma_var(rv, float(params["ewma_lambda"]))
    paths["ewma"][:] = s2[origins][:, None]
    info, i = [], 0
    for chunk in np.array_split(origins, folds):
        t0 = int(chunk[0])
        rows = range(i, i + len(chunk))
        i += len(chunk)
        try:
            fit = fit_direct(y[: t0 + 1], calendar, H, lags)
            plain = fit_direct(y[: t0 + 1], None, H, lags)
        except ModelError as exc:
            info.append({"fit_at": t0, "n_origins": len(chunk), "error": exc.message})
            continue
        info.append({"fit_at": t0, "n_origins": len(chunk), "error": None})
        rq = residual_quantiles(fit, q)
        plain_rq = residual_quantiles(plain, [0.5])[0]
        for r, t in zip(rows, chunk, strict=True):
            hist, cal = y[: t + 1], calendar[t + 1 : t + 1 + H]
            no_evt, no_dow = counterfactuals(cal, n_dow)
            cone = cone_paths(predict_path(fit, hist, cal), rq)
            paths["rv_har"][r] = cone[i_mid]
            paths["har"][r] = np.exp(predict_path(plain, hist, None) + plain_rq)
            if not np.isfinite(cone).all():
                continue  # an origin lag lacks RV (har_design)
            hi30[r] = rv_term(cone[i_hi])[Z_TENOR - 1]
            path_noevt = np.exp(predict_path(fit, hist, no_evt) + rq[i_mid])
            path_nodow = np.exp(predict_path(fit, hist, no_dow) + rq[i_mid])
            evt30[r] = decompose(cone[i_mid], path_noevt, path_nodow, Z_TENOR)[2]
    return Backtest(origins, paths, hi30, evt30, info)


# ---- scores -------------------------------------------------------------------------------


def _mean(values: list[float | None]) -> float | None:
    ok = [v for v in values if v is not None]
    return float(np.mean(ok)) if ok else None


def _split_masks(in_event: np.ndarray) -> dict[str, np.ndarray]:
    return {"all": np.ones(len(in_event), bool), "ex_event": ~in_event, "in_event": in_event}


def horizon_scores(
    origins: np.ndarray,
    log_paths: np.ndarray,
    y: np.ndarray,
    event_day: np.ndarray,
    insample: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """MASE/RMSSE per horizon on log daily variance, averaged over the horizons that score, per
    split. `log_paths[i, h−1]` forecasts y at origins[i] + h; naive scale m = 1 on `insample`."""
    per: dict[str, dict[str, list]] = {s: {"mase": [], "rmsse": [], "n": []} for s in SPLITS}
    for h in range(1, log_paths.shape[1] + 1):
        target = origins + h
        keep = target < len(y)
        a, f = y[target[keep]], log_paths[keep, h - 1]
        for split, mask in _split_masks(event_day[target[keep]]).items():
            per[split]["mase"].append(mase(a[mask], f[mask], insample))
            per[split]["rmsse"].append(rmsse(a[mask], f[mask], insample))
            per[split]["n"].append(int((np.isfinite(a[mask]) & np.isfinite(f[mask])).sum()))
    return {
        split: {
            "mase": _mean(v["mase"]),
            "rmsse": _mean(v["rmsse"]),
            "n_pairs": int(sum(v["n"])),
            "n_horizons": sum(1 for x in v["rmsse"] if x is not None),
        }
        for split, v in per.items()
    }


def tenor_scores(
    origins: np.ndarray,
    forecast: np.ndarray,
    realized: np.ndarray,
    event_day: np.ndarray,
    tenor: int,
    insample: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """MASE/RMSSE of a T-day annualized RV forecast per origin against `realized` (trailing RV,
    indexed by its last day) at origin + T; naive scale m = T (the previous window)."""
    end = origins + tenor
    keep = end < len(realized)
    a, f = realized[end[keep]], forecast[keep]
    in_event = np.array([event_day[t + 1 : t + 1 + tenor].any() for t in origins[keep]], bool)
    out = {}
    for split, mask in _split_masks(in_event).items():
        out[split] = {
            "mase": mase(a[mask], f[mask], insample, tenor),
            "rmsse": rmsse(a[mask], f[mask], insample, tenor),
            "n_pairs": int((np.isfinite(a[mask]) & np.isfinite(f[mask])).sum()),
        }
    return out


def ordinary_bias(
    origins: np.ndarray, var_paths: np.ndarray, rv: np.ndarray, event_day: np.ndarray
) -> dict[str, Any]:
    """Mean forecast − actual on ex-event target days, pooled over horizons, in annualized daily
    vol √(365·v); `relative` divides by the mean actual. Also in log daily variance."""
    errs, actual, log_errs = [], [], []
    for h in range(1, var_paths.shape[1] + 1):
        target = origins + h
        keep = target < len(rv)
        a, f = rv[target[keep]], var_paths[keep, h - 1]
        ok = np.isfinite(a) & (a > 0) & np.isfinite(f) & ~event_day[target[keep]]
        errs.append(np.sqrt(ANNUAL_DAYS * f[ok]) - np.sqrt(ANNUAL_DAYS * a[ok]))
        actual.append(np.sqrt(ANNUAL_DAYS * a[ok]))
        log_errs.append(np.log(f[ok]) - np.log(a[ok]))
    e, a, le = np.concatenate(errs), np.concatenate(actual), np.concatenate(log_errs)
    if not len(e):
        return {"mean_error_vol": None, "mean_actual_vol": None, "relative": None,
                "mean_error_log_var": None, "n_pairs": 0}  # fmt: skip
    return {
        "mean_error_vol": float(e.mean()),
        "mean_actual_vol": float(a.mean()),
        "relative": float(e.mean() / a.mean()),
        "mean_error_log_var": float(le.mean()),
        "n_pairs": len(e),
    }


# ---- signal P&L proxy (VRP-A-10) ------------------------------------------------------------


def sharpe(pnl: np.ndarray, periods_per_year: float) -> float | None:
    """mean / sd (ddof = 1) · √periods_per_year; None below 2 periods or with sd = 0."""
    if len(pnl) < 2:
        return None
    sd = float(np.std(pnl, ddof=1))
    return float(pnl.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else None


def pnl_proxy(
    first: int,
    dvol: np.ndarray,
    realized: np.ndarray,
    signal: np.ndarray,
    tenor: int = Z_TENOR,
) -> dict[str, Any]:
    """Short `tenor`-day variance at DVOL² from each non-overlapping entry t (every `tenor` days
    back from the last t whose window has matured, not before `first`), P&L = DVOL_t² −
    realized[t + tenor]². Periods without DVOL or a realized window are skipped for both
    strategies; the rule is flat (0) where `signal[t]` is False."""
    last = len(realized) - 1 - tenor
    entries = [t for t in range(last, first - 1, -tenor)][::-1]
    entries = [t for t in entries if math.isfinite(dvol[t]) and math.isfinite(realized[t + tenor])]
    t_idx = np.array(entries, dtype=int)
    naive = (
        dvol[t_idx] ** 2 - realized[t_idx + tenor] ** 2 if len(t_idx) else np.zeros(0, dtype=float)
    )
    short = signal[t_idx] if len(t_idx) else np.zeros(0, dtype=bool)
    rule = np.where(short, naive, 0.0)
    per_year = ANNUAL_DAYS / tenor

    def stats(p: np.ndarray, n_short: int) -> dict[str, Any]:
        return {
            "sharpe": sharpe(p, per_year),
            "mean_pnl": _opt(p.mean()) if len(p) else None,
            "total_pnl": float(p.sum()),
            "worst_pnl": _opt(p.min()) if len(p) else None,
            "n_short": n_short,
            "hit_rate": float((p[p != 0] > 0).mean()) if (p != 0).any() else None,
        }

    return {
        "n_periods": len(t_idx),
        "entries": t_idx.tolist(),
        "rule": stats(rule, int(short.sum())),
        "naive": stats(naive, len(t_idx)),
    }


# ---- gates (§9) -----------------------------------------------------------------------------


def _gate(value: float | None, threshold: float, rule: str, passed: bool | None) -> dict:
    status = "unavailable" if passed is None else ("pass" if passed else "fail")
    return {"value": value, "threshold": threshold, "rule": rule, "status": status}


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b is not None and b > 0 else None


def evaluate_gates(
    horizons: dict[str, dict[str, dict]],
    bias: dict[str, Any],
    pnl: dict[str, Any] | None,
    cfg: Config,
) -> dict[str, dict[str, Any]]:
    """The §9 table on one asset's scores. `horizons[model][split]["rmsse"]`, `bias` is rv_har's
    ordinary_bias, `pnl` is pnl_proxy's result or None off the DVOL basis."""
    ex = _ratio(horizons["rv_har"]["ex_event"]["rmsse"], horizons["har"]["ex_event"]["rmsse"])
    inn = _ratio(horizons["rv_har"]["in_event"]["rmsse"], horizons["har"]["in_event"]["rmsse"])
    rel = bias["relative"]
    limit = 1.0 + RMSSE_EX_EVENT_TOLERANCE
    gates = {
        "rmsse_ex_event_vs_har": _gate(
            ex, limit, "rmsse(rv_har)/rmsse(har) <= threshold", None if ex is None else ex <= limit
        ),
        "rmsse_in_event_vs_har": _gate(
            inn, 1.0, "rmsse(rv_har)/rmsse(har) < threshold", None if inn is None else inn < 1.0
        ),
        "bias_ordinary_days": _gate(
            rel,
            cfg.bias_tolerance,
            "|mean error / mean actual| <= threshold",
            None if rel is None else abs(rel) <= cfg.bias_tolerance,
        ),
    }
    if pnl is None:
        gates["pnl_sharpe_vs_naive"] = {
            "value": None, "threshold": 0.0, "rule": "no DVOL basis", "status": "not_applicable"
        }  # fmt: skip
    else:
        rule, naive = pnl["rule"]["sharpe"], pnl["naive"]["sharpe"]
        passed = None if rule is None or naive is None else (rule > 0.0 and rule > naive)
        gates["pnl_sharpe_vs_naive"] = _gate(
            rule, 0.0, "sharpe(rule) > threshold and sharpe(rule) > sharpe(naive)", passed
        )
    return gates


def verdict(gates: dict[str, dict[str, Any]]) -> str:
    """pass only when every applicable gate passes; unavailable is not a pass."""
    ok = all(g["status"] in ("pass", "not_applicable") for g in gates.values())
    return "pass" if ok else "fail"


# ---- one asset ------------------------------------------------------------------------------


def validate_series(
    asset: str,
    days: list[str],
    rv_values: list[float | None],
    dvol_by_day: dict[str, float | None] | None,
    params: dict[str, Any],
    cfg: Config,
) -> dict[str, Any]:
    """Backtest, scores, P&L proxy and gates for one asset's completed-day series. Raises
    ModelError("insufficient_history") when no origin has `min_history_days` of data."""
    rv = as_array(rv_values)
    ok = np.isfinite(rv) & (rv > 0)
    y = np.full(len(rv), np.nan)
    y[ok] = np.log(rv[ok])  # as rv_har: a zero or null day is missing
    H = int(params["horizon_days"])  # noqa: N806
    origins = eligible_origins(y, int(params["min_history_days"][asset]))
    if not len(origins):
        raise ModelError(
            "insufficient_history",
            f"{asset}: {int(ok.sum())} days with RV, need {params['min_history_days'][asset]} "
            "before the first origin",
        )
    calendar, n_dow, event_day = build_calendar(asset, days, params, H)
    bt = run_backtest(y, rv, calendar, n_dow, origins, params, min(cfg.folds, len(origins)))
    t0 = int(origins[0])

    horizons = {
        m: horizon_scores(origins, np.log(bt.paths[m]), y, event_day, y[: t0 + 1]) for m in MODELS
    }
    tenors: dict[str, Any] = {}
    min_days = int(params["window_min_days"])
    for T in (int(x) for x in params["tenors"]):  # noqa: N806
        realized = trailing_rv(rv, T, math.ceil(T * min_days / RV_WINDOW_DAYS))
        tenors[str(T)] = {
            m: tenor_scores(
                origins,
                np.array([rv_term(p)[T - 1] for p in bt.paths[m]]),
                realized,
                event_day,
                T,
                realized[: t0 + 1],
            )
            for m in MODELS
        }
    bias = {m: ordinary_bias(origins, bt.paths[m], rv, event_day) for m in MODELS}

    pnl = None
    if asset in DVOL_ASSETS and dvol_by_day is not None:
        n = len(days)
        dvol = as_array([dvol_by_day.get(d) for d in days])
        fc30, hi30, evt30 = (np.full(n, np.nan) for _ in range(3))
        fc30[origins] = [rv_term(p)[Z_TENOR - 1] for p in bt.paths["rv_har"]]
        hi30[origins], evt30[origins] = bt.rv_high_30, bt.event_share_30
        z = trailing_zscore(dvol - fc30, int(params["z_window_days"]), int(params["z_min_obs"]))[0]
        with np.errstate(invalid="ignore"):
            signal = (
                (z > cfg.z_threshold)
                & (dvol > hi30)
                & np.isfinite(evt30)
                & (evt30 <= float(params["event_hump_share"]))
            )
        realized30 = trailing_rv(rv, Z_TENOR, min_days)
        pnl = pnl_proxy(t0, dvol, realized30, signal)
        pnl["entries"] = [_close(days, t) for t in pnl["entries"]]
        pnl |= {
            "unit": "annualized variance (decimal²) per unit variance notional",
            "label": "variance-swap PROXY (VRP-A-10), not tradeable P&L",
            "tenor_days": Z_TENOR,
            "z_threshold": cfg.z_threshold,
            "z_scored": int(np.isfinite(z).sum()),
            "sharpe_basis": f"non-overlapping {Z_TENOR}-day periods, x sqrt(365/{Z_TENOR})",
        }

    gates = evaluate_gates({m: horizons[m] for m in MODELS}, bias["rv_har"], pnl, cfg)
    pad = max(max(w) for w in params["event_windows"].values())
    last = date.fromisoformat(days[-1]) + timedelta(days=pad)
    gaps = incomplete_types(params["events_listed_through"], last)
    gates["event_calendar_complete"] = {
        "value": gaps, "threshold": last.isoformat(), "status": "fail" if gaps else "pass",
        "rule": "every type in events_listed_through is listed through threshold",
    }  # fmt: skip
    return {
        "status": "ok",
        "backtest": {
            "first_origin": _close(days, t0),
            "last_origin": _close(days, int(origins[-1])),
            "n_origins": len(origins),
            "horizon_days": H,
            "folds": [f | {"fit_at": _close(days, f["fit_at"])} for f in bt.folds],
        },
        "accuracy": {
            "horizons": {"unit": "log daily variance", "naive_m": 1} | horizons,
            "tenors": {"unit": "annualized realized vol", "naive_m": "tenor"} | tenors,
        },
        "bias_ordinary_days": bias,
        "pnl_proxy": pnl,
        "gates": gates,
        "result": verdict(gates),
    }


# ---- recording → report ---------------------------------------------------------------------


def load_series(recording: Path, asset: str, manifest: Manifest) -> dict[str, Any]:
    """rv_daily (required) and dvol_history (optional) for one asset, replayed through the
    production adapters with the run's InputSpecs, at the recording's start time."""
    client = RecordedClient(recording)
    adapters = {
        "deribit": DeribitAdapter(client),
        "hyperliquid": HyperliquidAdapter(RecordedHyperliquidClient(recording)),
    }
    as_of = datetime.fromisoformat(client.manifest["started_at"])
    inputs: dict[str, InputData] = {}
    unavailable: dict[str, str] = {}
    for spec in load_model(manifest).required_inputs({"underlying": asset}):
        if spec.name not in SERIES:
            continue
        try:
            inputs[spec.name] = adapters[spec.source].fetch(spec, as_of)
        except AdapterError as exc:
            if not spec.optional:
                raise
            unavailable[spec.name] = f"{exc.code}: {exc.message}"
    return {"inputs": inputs, "unavailable": unavailable}


def describe_input(d: InputData) -> dict[str, Any]:
    out = {"hash": input_hash({d.name: d}), "snapshot_at": d.snapshot_at.isoformat()}
    days = d.data["day"]
    values = d.data["rv" if d.name == "rv_daily" else "dvol"]
    out |= {k: d.data[k] for k in ("venue", "instrument", "resolution") if k in d.data}
    return out | {
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "n_days": len(days),
        "n_days_missing": sum(1 for v in values if v is None),
    }


def validate(recording: Path, assets: list[str] | None, cfg: Config) -> dict[str, Any]:
    manifest_path = recording / "manifest.json"
    if not manifest_path.exists():
        raise RecordingError(f"{recording}: no manifest.json (incomplete or not a recording)")
    rec = json.loads(manifest_path.read_text())
    if IMPLEMENTATION not in rec.get("implementations", []):
        raise RecordingError(f"{recording}: {IMPLEMENTATION} is not in this recording")
    manifest = load_manifest(library_dir() / IMPLEMENTATION / "manifest.yaml")
    params = manifest.params
    recorded = [a for a, s in rec["assets"].items() if s["ok"]]
    out_assets: dict[str, Any] = {}
    for asset in assets or recorded:
        if asset not in recorded:
            out_assets[asset] = {
                "status": "failed",
                "error": {"code": "not_recorded", "message": f"{asset} has no ok recording"},
                "result": "fail",
            }
            continue
        try:
            loaded = load_series(recording, asset, manifest)
            rv_in = loaded["inputs"]["rv_daily"]
            dvol_in = loaded["inputs"].get("dvol_history")
            entry: dict[str, Any] = {
                "inputs": {
                    "hash": input_hash(loaded["inputs"]),
                    "rv_daily": describe_input(rv_in),
                    "dvol_history": describe_input(dvol_in) if dvol_in else None,
                    "unavailable": loaded["unavailable"],
                }
            }
            dvol = (
                dict(zip(dvol_in.data["day"], dvol_in.data["dvol"], strict=True))
                if dvol_in
                else None
            )
            if not rv_in.data["day"]:
                raise ModelError("insufficient_history", f"{asset}: rv_daily has no completed day")
            entry |= validate_series(
                asset, list(rv_in.data["day"]), list(rv_in.data["rv"]), dvol, params, cfg
            )
            entry["status"] = "ok"
        except (AdapterError, ModelError) as exc:
            entry = {"status": "failed", "error": {"code": exc.code, "message": exc.message},
                     "result": "fail"}  # fmt: skip
        out_assets[asset] = entry
    return {
        "report_version": REPORT_VERSION,
        "tool": "terminal_models.tools.validate_rv",
        "git_sha": git_sha(),
        "recording": {
            "name": recording.name,
            "started_at": rec["started_at"],
            "recorder_git_sha": rec["git_sha"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "implementation": {"id": IMPLEMENTATION, "version": manifest.version},
        "params_hash": params_hash(params),
        "config": {
            "folds": cfg.folds,
            "z_threshold": cfg.z_threshold,
            "bias_tolerance": cfg.bias_tolerance,
            "rmsse_ex_event_tolerance": RMSSE_EX_EVENT_TOLERANCE,
        },
        "assets": out_assets,
        "result": "pass" if all(a["result"] == "pass" for a in out_assets.values()) else "fail",
    }


def summary(report: dict[str, Any]) -> list[str]:
    lines = [f"recording {report['recording']['name']}: {report['result']}"]
    for asset, a in report["assets"].items():
        if a["status"] != "ok":
            lines.append(f"{asset}: FAILED {a['error']['code']}: {a['error']['message']}")
            continue
        lines.append(f"{asset}: {a['result']} ({a['backtest']['n_origins']} origins)")
        lines += [
            f"  {name}: {g['status']} (value {g['value']}, threshold {g['threshold']})"
            for name, g in a["gates"].items()
        ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m terminal_models.tools.validate_rv")
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    parser.add_argument("--assets", default="", help="comma-separated; default: all recorded")
    parser.add_argument("--folds", type=int, default=Config.folds)
    parser.add_argument("--z-threshold", type=float, default=Config.z_threshold)
    parser.add_argument("--bias-tolerance", type=float, default=Config.bias_tolerance)
    args = parser.parse_args(argv)
    if args.folds < 1:
        parser.error("--folds must be ≥ 1")
    assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()] or None
    cfg = Config(args.folds, args.z_threshold, args.bias_tolerance)
    try:
        report = validate(args.recording, assets, cfg)
    except RecordingError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print("\n".join(summary(report)))
    print(f"wrote {args.out}")
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
