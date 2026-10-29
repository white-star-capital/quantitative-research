"""rv_har: calendar- and event-aware RV forecast against ATM IV; the vrp_signal slot.

Methodology: RV_FORECAST_IMPLEMENTATION_GUIDE.md §3–6 with its amendments (VRP-A-1 HAR in log
variance, direct h-step OLS, empirical residual cone). Payload: §12. Pure: everything comes from
ctx.inputs (MODELS M-5); the fit is redone every run from completed UTC days (VRP-A-4).

Days are UTC days identified by their close (the 00:00 that ends them, §12.1). The `rv_daily`
input lists days by date; index i of every per-day array below is that input's day i. C is the
close of the last one; forecast day k (1..horizon) is the k-th day from C.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import numpy as np

from terminal_models.core.base import InputSpec, Model, ModelError, RunContext, Scope
from terminal_models.core.payloads import axis, column, item, section, series, unit
from terminal_models.core.quant.calendar import (
    Event,
    days_left,
    events_for,
    incomplete_types,
    weekdays,
    window,
    window_dummies,
)
from terminal_models.core.quant.metrics import Metric, atm_iv
from terminal_models.core.quant.rv import (
    ANNUAL_DAYS,
    DAY_MS,
    aggregate_rv,
    as_array,
    cone_paths,
    counterfactuals,
    decompose,
    dow_dummies,
    ewma_var,
    expanding_forecasts,
    fit_direct,
    mean_path,
    predict_path,
    residual_quantiles,
    rv_term,
    smearing_factors,
    to_optional,
    trailing_rv,
    trailing_zscore,
)
from terminal_models.core.quant.smile import Surface
from terminal_models.core.quant.surface import build_expiries

DVOL_ASSETS = ("BTC", "ETH")  # the z basis exists only where Deribit publishes DVOL (VRP-A-5)
Z_TENOR = 30  # DVOL is a 30-day index
RV_WINDOW_DAYS = 30  # backward-looking realized window (§12.4, §12.6)

IV = unit("implied_vol", "decimal")
RV = unit("realized_vol", "decimal")
VOL_SPREAD = unit("vol_spread", "decimal")
ZSCORE = unit("zscore")
RATIO = unit("ratio", "decimal")
COUNT = unit("count")
TENOR = unit("tenor_days")


def day_close_ms(day: str) -> int:
    """The close (next 00:00 UTC) of the UTC day `YYYY-MM-DD`."""
    return int(datetime.combine(date.fromisoformat(day), time(), UTC).timestamp() * 1000) + DAY_MS


def utc_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _opt(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


class RvHar(Model):
    def __init__(self, manifest) -> None:
        self.manifest = manifest

    def required_inputs(self, scope: Scope) -> list[InputSpec]:
        u = scope["underlying"]
        src = self.manifest.params["rv_sources"][u]
        specs = [
            InputSpec("chain", "deribit", {"kind": "chain", "underlying": u}),
            InputSpec("index", "deribit", {"kind": "index", "underlying": u}),
            InputSpec(
                "rv_daily",
                src["venue"],
                {
                    "kind": "rv_daily",
                    "underlying": u,
                    "resolution": src["resolution"],
                    "lookback_days": src["lookback_days"],
                    "min_coverage": src["min_coverage"],
                },
            ),
        ]
        if u in DVOL_ASSETS:
            specs.append(
                InputSpec(
                    "dvol_history",
                    "deribit",
                    {
                        "kind": "dvol_history",
                        "underlying": u,
                        "lookback_days": src["lookback_days"],
                    },
                    optional=True,
                )
            )
        return specs

    def run(self, ctx: RunContext) -> dict[str, Any]:
        f = Forecast(ctx)
        return {
            "sections": [
                signal_section(f),
                context_section(f),
                cone_section(f),
                history_section(f),
                events_section(f),
            ]
        }


IMPLEMENTATION = RvHar


# ---- forecast (§12.1–12.2) ------------------------------------------------------------------


class Forecast:
    """Everything the sections read, computed once per run."""

    def __init__(self, ctx: RunContext) -> None:
        p, u = ctx.params, ctx.scope["underlying"]
        self.ctx, self.p, self.u = ctx, p, u
        self.tenors = [int(t) for t in p["tenors"]]
        self.H = int(p["horizon_days"])
        self.lags = tuple(int(x) for x in p["har_lags"])

        rv_in = ctx.inputs["rv_daily"].data
        self.days = list(rv_in["day"])
        if not self.days:
            raise ModelError("insufficient_history", f"{u}: rv_daily has no completed day")
        self.rv = as_array(rv_in["rv"])
        ok = np.isfinite(self.rv) & (self.rv > 0)
        self.y = np.full(len(self.rv), np.nan)
        self.y[ok] = np.log(self.rv[ok])  # y_d = ln RV_d (§3); a zero or null day is missing
        self.n = len(self.y)
        self.n_days = int(ok.sum())
        need = int(p["min_history_days"][u])
        if self.n_days < need:
            raise ModelError(
                "insufficient_history", f"{u}: {self.n_days} days with RV, need {need}"
            )

        self.closes = [day_close_ms(d) for d in self.days]
        self.C = self.closes[-1]  # noqa: N815
        self.day1 = date.fromisoformat(self.days[-1]) + timedelta(days=1)  # forecast day 1
        future = [self.C + k * DAY_MS for k in range(1, self.H + 1)]
        self.windows = p["event_windows"]
        pad = timedelta(days=max(max(w) for w in self.windows.values()))
        start = datetime.combine(date.fromisoformat(self.days[0]), time(), UTC) - pad
        end = datetime.combine(self.day1 + timedelta(days=self.H), time(), UTC) + pad
        self.events: list[Event] = events_for(u, p["events"], start, end)
        # Per tenor, the listed types whose list stops before that tenor's last window: its days
        # read "no event" where there may be one, so that row is not signal_ok (§12.4). The fit
        # itself uses past days only, which the lists cover, so the shorter tenors stay live.
        listed = p["events_listed_through"]
        self.tenor_gaps = {
            t: incomplete_types(listed, self.day1 + timedelta(days=t - 1) + pad)
            for t in self.tenors
        }
        # The whole signal is disabled only once even the shortest tenor is uncovered.
        self.calendar_gaps = self.tenor_gaps[min(self.tenors)]
        # Margin of the longest tenor, the first to lose coverage; negative once it has, None
        # when no listed type is configured (nothing to run out).
        self.calendar_days_left = days_left(listed, self.day1 + timedelta(days=self.H - 1) + pad)
        dow = dow_dummies(weekdays(self.closes + future))
        _, evt = window_dummies(self.closes + future, self.events, self.windows)
        self.calendar = np.hstack([dow, evt])
        self.n_dow = dow.shape[1]

        self._fit()
        self._iv(ctx)
        self._history()

    # The direct HAR+events fit, its cone and the counterfactual paths.
    def _fit(self) -> None:
        p, y = self.p, self.y
        fit = fit_direct(y, self.calendar, self.H, self.lags)
        cal_f = self.calendar[self.n :]
        y_hat = predict_path(fit, y, cal_f)
        if not np.isfinite(y_hat).all():
            raise ModelError(
                "insufficient_history",
                f"{self.u}: the last day has no RV, or a lag window ending there has too few "
                "days with RV; no forecast origin",
            )
        q = [float(x) for x in p["quantiles"]]
        paths = cone_paths(y_hat, residual_quantiles(fit, q))
        smear = smearing_factors(fit)  # central path is the mean, the band quantiles (VRP-A-15)
        no_evt, no_dow = counterfactuals(cal_f, self.n_dow)
        self.mid = mean_path(y_hat, smear)
        self.lo, self.hi = (paths[q.index(x)] for x in (min(q), max(q)))
        self.terms = {"mid": rv_term(self.mid), "lo": rv_term(self.lo), "hi": rv_term(self.hi)}
        self.no_evt = mean_path(predict_path(fit, y, no_evt), smear)
        self.no_dow = mean_path(predict_path(fit, y, no_dow), smear)
        self.fit_r2 = float(fit.r2[0])
        self.resid_sd = float(np.std(fit.resid[0], ddof=1))

        plain = fit_direct(y, None, Z_TENOR, self.lags)  # plain-HAR baseline (VRP-A-1)
        plain_path = mean_path(predict_path(plain, y, None), smearing_factors(plain))
        self.rv_har_30d = aggregate_rv(plain_path, Z_TENOR)
        s2 = ewma_var(self.rv, float(p["ewma_lambda"]))[-1]
        self.rv_ewma_30d = math.sqrt(ANNUAL_DAYS * s2) if math.isfinite(s2) else math.nan

        lo_ok, hi_ok = p["rv_sanity"]["min"], p["rv_sanity"]["max"]
        paths_all = (self.mid, self.lo, self.hi, self.no_evt, self.no_dow)
        if not all(np.isfinite(v).all() and (v > 0).all() for v in paths_all):
            raise ModelError(
                "degenerate_forecast", f"{self.u}: non-finite or non-positive variance"
            )
        for name, key in (("rv_forecast", "mid"), ("rv_low", "lo"), ("rv_high", "hi")):
            term = self.terms[key]
            if term.min() < lo_ok or term.max() > hi_ok:
                raise ModelError(
                    "degenerate_forecast",
                    f"{self.u}: {name} spans [{term.min():.4f}, {term.max():.4f}], "
                    f"outside rv_sanity [{lo_ok}, {hi_ok}]",
                )
        baselines = (self.fit_r2, self.resid_sd, self.rv_har_30d, self.rv_ewma_30d)
        if not all(math.isfinite(x) for x in baselines):
            raise ModelError("degenerate_forecast", f"{self.u}: non-finite fit diagnostics")

    # ATM IV per tenor on the chain surface (VRP-A-6).
    def _iv(self, ctx: RunContext) -> None:
        spot = ctx.inputs["index"].data["spot"]
        rows = ctx.inputs["chain"].data["rows"]
        expiries, _ = build_expiries(rows, ctx, spot, float(self.p["usd_rate"]))
        surface = Surface(spot, expiries)
        self.atm: dict[int, Metric] = {t: atm_iv(surface, t) for t in self.tenors}

    # Expanding-origin 30d forecasts, DVOL and the trailing z (§12.6, VRP-A-5).
    def _history(self) -> None:
        p, n = self.p, self.n
        span = int(p["history_days"]) + max(int(p["z_window_days"]), RV_WINDOW_DAYS)
        have = np.cumsum(np.isfinite(self.y))
        need = int(p["min_history_days"][self.u])
        origins = [t for t in range(max(0, n - span), n) if have[t] >= need]
        paths = expanding_forecasts(
            self.y,
            self.calendar,
            origins,
            Z_TENOR,
            "mean",
            self.lags,
            refit_every=int(p["history_refit_every"]),
        )
        self.fc30 = np.full(n, np.nan)
        for t, path in zip(origins, paths, strict=True):
            self.fc30[t] = rv_term(path)[Z_TENOR - 1]  # same arithmetic as the cone
        self.matured = np.full(n, np.nan)
        self.matured[Z_TENOR:] = self.fc30[:-Z_TENOR]
        self.realized = trailing_rv(self.rv, RV_WINDOW_DAYS, int(p["window_min_days"]))

        self.dvol = np.full(n, np.nan)
        if "dvol_history" in self.ctx.inputs:
            d = self.ctx.inputs["dvol_history"].data
            by_day = dict(zip(d["day"], d["dvol"], strict=True))
            # Non-zero when Deribit's 1D candles don't close at 00:00 UTC: the adapter keys each
            # to the day it closes in, so the z basis is offset from the RV day close.
            if d.get("close_offset_h"):
                self.ctx.warn(
                    "dvol_offset",
                    f"DVOL 1D candles close at {d['close_offset_h']:g}h UTC, not 00:00",
                )
            self.dvol = as_array([by_day.get(day) for day in self.days])
        self.vrp30 = self.dvol - self.fc30
        self.z, self.z_mean, self.z_sd, self.z_n = trailing_zscore(
            self.vrp30, int(p["z_window_days"]), int(p["z_min_obs"])
        )

    # ---- derived per tenor ----

    def z_cell(self, tenor: int) -> tuple[float | None, str]:
        """(vrp_z, flag) for one signal row (§12.3)."""
        if self.u not in DVOL_ASSETS or tenor != Z_TENOR:
            return None, "z_unavailable"
        z = _opt(self.z[-1])
        return (z, "z_basis_dvol") if z is not None else (None, "z_insufficient_history")

    def forecast_dates(self, tenor: int) -> tuple[date, date]:
        return self.day1, self.day1 + timedelta(days=tenor - 1)

    def overlaps(self, e: Event, tenor: int) -> bool:
        first, last = window(e, self.windows)
        d1, dt = self.forecast_dates(tenor)
        return first <= dt and last >= d1

    @property
    def disabled_by(self) -> tuple[str, ...]:
        """Why the signal is off (§12.4): the RV input lags (C is before the latest 00:00 UTC
        ≤ as_of), or a listed event type's calendar stops before even the shortest tenor."""
        floor = datetime.combine(self.ctx.as_of.astimezone(UTC).date(), time(), UTC)
        lagging = self.C < int(floor.timestamp() * 1000)
        return (("rv_input_lagging",) if lagging else ()) + (
            ("event_calendar_incomplete",) if self.calendar_gaps else ()
        )

    @property
    def signal_enabled(self) -> bool:
        return not self.disabled_by


# ---- sections (§12.3–12.7) -------------------------------------------------------------------

SIGNAL_COLUMNS = [
    column("tenor_days", "Tenor", "number", TENOR),
    column("iv_impl", "ATM IV", "number", IV),
    column("rv_forecast", "RV forecast", "number", RV),
    column("rv_low", "RV low (q0.1)", "number", RV),
    column("rv_high", "RV high (q0.9)", "number", RV),
    column("vrp", "VRP", "number", VOL_SPREAD),
    column("vrp_z", "VRP z", "number", ZSCORE),
    column("comp_baseline", "Baseline share", "number", RATIO),
    column("comp_seasonal", "Seasonal share", "number", RATIO),
    column("comp_event", "Event share", "number", RATIO),
    column("event_in_horizon", "Event in horizon", "boolean"),
    column("rich", "Rich (IV > RV high)", "boolean"),
    column("event_hump", "Event hump", "boolean"),
    column("signal_ok", "Signal OK", "boolean"),
    column("flags", "Flags", "string"),
]


def signal_section(f: Forecast) -> dict:
    share = float(f.p["event_hump_share"])
    rows = []
    for t in sorted(f.tenors):
        iv = f.atm[t].value
        rv, lo, hi = (float(f.terms[k][t - 1]) for k in ("mid", "lo", "hi"))  # = cone at x = t
        base, sea, evt = decompose(f.mid, f.no_evt, f.no_dow, t)
        z, z_flag = f.z_cell(t)
        hump = evt > share
        flags = (["iv_unavailable"] if iv is None else []) + list(f.atm[t].flags) + [z_flag]
        covered = not f.tenor_gaps[t]
        if not covered:
            flags.append("event_calendar_incomplete")  # event_in_horizon may miss an event
        rows.append(
            [
                t,
                iv,
                rv,
                lo,
                hi,
                None if iv is None else iv - rv,
                z,
                base,
                sea,
                evt,
                any(f.overlaps(e, t) for e in f.events),
                None if iv is None else iv > hi,
                hump,
                z is not None and iv is not None and not hump and covered and f.signal_enabled,
                ",".join(flags),
            ]
        )
    return section("signal", "Signal by tenor", "table", {"columns": SIGNAL_COLUMNS, "rows": rows})


def context_section(f: Forecast) -> dict:
    p = f.p
    trailing = _opt(f.realized[-1])
    dvol, vrp30 = _opt(f.dvol[-1]), _opt(f.vrp30[-1])
    z, z_flag = f.z_cell(Z_TENOR)
    z_flags = () if z is not None else (z_flag,)
    no_dvol = () if dvol is not None else ("dvol_unavailable",)
    enabled = f.signal_enabled
    items = [
        item(
            "last_complete_day",
            "Last complete day",
            f.C // 1000,
            unit("timestamp", convention="unix_seconds"),
        ),
        item("n_days", "Days with RV", f.n_days, COUNT),
        item("n_days_missing", "Days missing", f.n - f.n_days, COUNT),
        item("fit_r2", "Fit R² (h=1)", f.fit_r2, RATIO),
        item("resid_sd", "Residual SD (h=1)", f.resid_sd, unit("other", convention="log_variance")),
        item(
            "rv_trailing_30d",
            "Realized 30d",
            trailing,
            RV,
            () if trailing is not None else ("insufficient_days",),
        ),
        item("rv_ewma_30d", "EWMA 30d", f.rv_ewma_30d, RV),
        item("rv_har_30d", "Plain HAR 30d", f.rv_har_30d, RV),
        item("dvol", "DVOL", dvol, IV, no_dvol),
        item("vrp_30d_dvol", "VRP 30d (DVOL)", vrp30, VOL_SPREAD, no_dvol),
        item("vrp_z_mean", "VRP z mean", _opt(f.z_mean[-1]) if z is not None else None,
             VOL_SPREAD, z_flags),
        item("vrp_z_sd", "VRP z SD", _opt(f.z_sd[-1]) if z is not None else None, VOL_SPREAD,
             z_flags),
        item("vrp_z_n", "VRP z window", int(f.z_n[-1]), COUNT),
        item("event_hump_share", "Event hump threshold", float(p["event_hump_share"]), RATIO),
        item("calendar_days_left", "Event calendar days left", f.calendar_days_left, COUNT,
             () if f.calendar_days_left is not None else ("no_listed_types",)),
        item(
            "signal_enabled",
            "Signal enabled",
            enabled,
            unit("other"),
            f.disabled_by,
        ),
    ]  # fmt: skip
    return section("context", "Context", "scalar_set", {"items": items})


def cone_section(f: Forecast) -> dict:
    x = list(range(1, f.H + 1))
    atm = [f.atm[t].value if t in f.atm else None for t in x]  # sparse: signal tenors only
    return section(
        "cone",
        "RV cone vs ATM IV",
        "curve",
        {
            "x": axis("Horizon (days)", TENOR, x),
            "series": [
                series("rv_forecast", "RV forecast", RV, to_optional(f.terms["mid"])),
                series("rv_low", "RV low (q0.1)", RV, to_optional(f.terms["lo"])),
                series("rv_high", "RV high (q0.9)", RV, to_optional(f.terms["hi"])),
                series("atm_iv", "ATM IV", IV, atm),
                series("rv_path", "Daily path", RV, to_optional(np.sqrt(ANNUAL_DAYS * f.mid))),
                series(
                    "rv_path_ex_event",
                    "Daily path ex events",
                    RV,
                    to_optional(np.sqrt(ANNUAL_DAYS * f.no_evt)),
                ),
            ],
        },
    )


def history_section(f: Forecast) -> dict:
    k = min(int(f.p["history_days"]), f.n)

    def tail(values: np.ndarray) -> list[float | None]:
        return to_optional(values[-k:])

    return section(
        "history",
        "History (30d)",
        "timeseries",
        {
            "t": [utc_ms(c) for c in f.closes[-k:]],
            "series": [
                series("rv_realized_30d", "Realized 30d", RV, tail(f.realized)),
                series("rv_forecast_30d", "Forecast 30d", RV, tail(f.fc30)),
                series("rv_forecast_30d_matured", "Forecast 30d (matured)", RV, tail(f.matured)),
                series("dvol", "DVOL", IV, tail(f.dvol)),
                series("vrp_30d", "VRP 30d (DVOL)", VOL_SPREAD, tail(f.vrp30)),
                series("vrp_z_30d", "VRP z (30d, DVOL)", ZSCORE, tail(f.z)),
            ],
        },
    )


EVENT_COLUMNS = [
    column("ts", "Time", "timestamp"),
    column("type", "Type", "string"),
    column("scope", "Scope", "string"),
    column("window_before", "Days before", "number", COUNT),
    column("window_after", "Days after", "number", COUNT),
    column("horizon_day", "Forecast day", "number", COUNT),
    column("source", "Source", "string"),
]


def events_section(f: Forecast) -> dict:
    longest = max(f.tenors)
    rows = []
    for e in f.events:  # already sorted by (ts, type)
        if not f.overlaps(e, longest):
            continue
        before, after = f.windows[e.type]
        rows.append(
            [
                e.ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                e.type,
                e.scope,
                int(before),
                int(after),
                (e.ts.date() - f.day1).days + 1,
                e.source,
            ]
        )
    return section("events", "Events in horizon", "table", {"columns": EVENT_COLUMNS, "rows": rows})
