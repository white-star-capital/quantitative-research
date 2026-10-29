"""tools/validate_rv.py: metric plumbing and gate logic (RV_FORECAST_IMPLEMENTATION_GUIDE.md §9).

Hand-built arrays check the scores, the P&L proxy and the gates; the end-to-end checks run on the
TEST-ONLY synthetic series of library/rv_har/tests/synthetic.py. None of it is market data, and no
result here says anything about rv_har's real accuracy. The recording path is covered in
test_replay_pipeline.py.
"""

import json
import math

import numpy as np
import pytest

from terminal_models.core.base import ModelError
from terminal_models.core.manifest import load_manifest
from terminal_models.core.paths import library_dir
from terminal_models.library.rv_har.tests import synthetic as syn
from terminal_models.tools import validate_rv as v

PARAMS = load_manifest(library_dir() / "rv_har" / "manifest.yaml").params
FAST = PARAMS | {"min_history_days": {a: 200 for a in syn.SPOT}, "z_window_days": 40,
                 "z_min_obs": 20}  # fmt: skip
N_DAYS = 500


def series(asset="BTC", n=N_DAYS):
    rv = syn.rv_series(n)
    days = syn.days(n)
    dvol = dict(zip(days, syn.dvol_series(rv), strict=True)) if asset in ("BTC", "ETH") else None
    return days, rv, dvol


# ---- scores on hand-built arrays ----------------------------------------------------------


def test_horizon_scores_split_and_scale():
    y = 0.1 * np.arange(40.0)  # naive one-step diffs are all 0.1
    origins = np.arange(20, 30)
    H = 3  # noqa: N806
    event_day = np.zeros(40, bool)
    event_day[[22, 25, 31]] = True
    log_paths = np.array([[y[t + h] for h in range(1, H + 1)] for t in origins])
    for i, t in enumerate(origins):
        for h in range(1, H + 1):
            if event_day[t + h]:
                log_paths[i, h - 1] += 0.3  # error only on event days
    out = v.horizon_scores(origins, log_paths, y, event_day, y[:21])
    assert out["ex_event"]["rmsse"] == pytest.approx(0.0)
    assert out["ex_event"]["mase"] == pytest.approx(0.0)
    assert out["in_event"]["rmsse"] == pytest.approx(3.0)  # 0.3 / 0.1 at every horizon
    assert out["in_event"]["mase"] == pytest.approx(3.0)
    assert out["all"]["n_pairs"] == len(origins) * H
    assert out["in_event"]["n_pairs"] + out["ex_event"]["n_pairs"] == out["all"]["n_pairs"]
    assert out["all"]["n_horizons"] == H


def test_horizon_scores_skip_targets_past_the_data_and_nulls():
    y = 0.1 * np.arange(10.0)
    y[8] = np.nan
    log_paths = np.zeros((2, 2))
    out = v.horizon_scores(np.array([7, 8]), log_paths, y, np.zeros(10, bool), y[:7])
    # targets: h=1 → 8 (null), 9; h=2 → 9, 10 (past the data)
    assert out["all"]["n_pairs"] == 2
    assert out["in_event"] == {"mase": None, "rmsse": None, "n_pairs": 0, "n_horizons": 0}


def test_tenor_scores_window_is_in_event_when_any_day_is():
    realized = np.linspace(0.4, 0.6, 50)
    origins = np.array([10, 20])
    event_day = np.zeros(50, bool)
    event_day[25] = True  # inside (20, 27] only
    forecast = realized[origins + 7] + 0.01
    out = v.tenor_scores(origins, forecast, realized, event_day, 7, realized[:11])
    assert out["in_event"]["n_pairs"] == 1 and out["ex_event"]["n_pairs"] == 1
    assert out["all"]["n_pairs"] == 2


def test_ordinary_bias_is_ex_event_and_in_vol():
    rv = np.full(20, 0.25**2 / 365)
    event_day = np.zeros(20, bool)
    event_day[12] = True
    paths = np.full((2, 2), 0.3**2 / 365)  # 0.30 forecast vs 0.25 actual
    paths[0, 1] = 10.0  # targets day 12 (in-event): must not count
    out = v.ordinary_bias(np.array([10, 11]), paths, rv, event_day)
    assert out["n_pairs"] == 2  # day 12 is the target of (10, h=2) and (11, h=1)
    assert out["mean_error_vol"] == pytest.approx(0.05)
    assert out["relative"] == pytest.approx(0.2)
    assert out["mean_error_log_var"] == pytest.approx(math.log(0.09 / 0.0625))


# ---- P&L proxy -----------------------------------------------------------------------------


def pnl_arrays(n=100):
    dvol = np.full(n, 0.5)
    realized = np.full(n, np.nan)
    realized[[39, 69, 99]] = [0.4, 0.6, 0.3]  # windows from entries 9, 39, 69
    signal = np.zeros(n, bool)
    signal[39] = True
    return dvol, realized, signal


def test_pnl_entries_are_non_overlapping_from_the_last_matured_window():
    dvol, realized, signal = pnl_arrays()
    out = v.pnl_proxy(0, dvol, realized, signal)
    assert out["entries"] == [9, 39, 69]
    naive = np.array([0.25 - 0.16, 0.25 - 0.36, 0.25 - 0.09])
    assert out["naive"]["total_pnl"] == pytest.approx(naive.sum())
    assert out["naive"]["sharpe"] == pytest.approx(
        naive.mean() / naive.std(ddof=1) * math.sqrt(365 / 30)
    )
    assert out["naive"]["n_short"] == 3 and out["rule"]["n_short"] == 1
    rule = np.array([0.0, naive[1], 0.0])  # flat unless the signal is on
    assert out["rule"]["total_pnl"] == pytest.approx(naive[1])
    assert out["rule"]["sharpe"] == pytest.approx(
        rule.mean() / rule.std(ddof=1) * math.sqrt(365 / 30)
    )
    assert out["rule"]["hit_rate"] == 0.0


def test_pnl_skips_periods_without_dvol_or_realized_and_respects_first():
    dvol, realized, signal = pnl_arrays()
    dvol[39] = np.nan
    assert v.pnl_proxy(0, dvol, realized, signal)["entries"] == [9, 69]
    assert v.pnl_proxy(10, *pnl_arrays())["entries"] == [39, 69]


def test_sharpe_is_null_when_undefined():
    assert v.sharpe(np.array([0.1]), 12) is None
    assert v.sharpe(np.zeros(5), 12) is None


# ---- gates ---------------------------------------------------------------------------------


def horizons(ex=(1.0, 1.0), inn=(0.9, 1.0)):
    def split(r):
        return {"rmsse": r, "mase": r, "n_pairs": 1, "n_horizons": 1}

    return {
        "rv_har": {"ex_event": split(ex[0]), "in_event": split(inn[0])},
        "har": {"ex_event": split(ex[1]), "in_event": split(inn[1])},
    }


def pnl(rule, naive):
    return {"rule": {"sharpe": rule}, "naive": {"sharpe": naive}}


CFG = v.Config()


@pytest.mark.parametrize(
    ("ex", "status"), [((1.04, 1.0), "pass"), ((1.06, 1.0), "fail"), ((None, 1.0), "unavailable")]
)
def test_gate_ex_event_within_five_percent(ex, status):
    gates = v.evaluate_gates(horizons(ex=ex), {"relative": 0.0}, None, CFG)
    assert gates["rmsse_ex_event_vs_har"]["status"] == status


@pytest.mark.parametrize(("inn", "status"), [((0.99, 1.0), "pass"), ((1.0, 1.0), "fail")])
def test_gate_in_event_must_beat_har(inn, status):
    gates = v.evaluate_gates(horizons(inn=inn), {"relative": 0.0}, None, CFG)
    assert gates["rmsse_in_event_vs_har"]["status"] == status


@pytest.mark.parametrize(
    ("rel", "status"), [(-0.09, "pass"), (0.11, "fail"), (-0.11, "fail"), (None, "unavailable")]
)
def test_gate_bias(rel, status):
    gates = v.evaluate_gates(horizons(), {"relative": rel}, None, CFG)
    assert gates["bias_ordinary_days"]["status"] == status


@pytest.mark.parametrize(
    ("p", "status"),
    [
        (None, "not_applicable"),
        (pnl(0.5, 0.3), "pass"),
        (pnl(0.5, 0.7), "fail"),  # positive but worse than always-short
        (pnl(-0.1, -0.3), "fail"),  # better than naive but not positive
        (pnl(None, 0.3), "unavailable"),  # the rule never traded
    ],
)
def test_gate_pnl(p, status):
    gates = v.evaluate_gates(horizons(), {"relative": 0.0}, p, CFG)
    assert gates["pnl_sharpe_vs_naive"]["status"] == status


def test_verdict_needs_every_applicable_gate():
    gates = v.evaluate_gates(horizons(), {"relative": 0.0}, None, CFG)
    assert v.verdict(gates) == "pass"
    gates["bias_ordinary_days"]["status"] = "unavailable"
    assert v.verdict(gates) == "fail"


# ---- one asset on TEST-ONLY synthetic series ----------------------------------------------


def test_validate_series_report_shape():
    days, rv, dvol = series()
    out = v.validate_series("BTC", days, rv, dvol, FAST, v.Config(folds=5))
    json.dumps(out, allow_nan=False)
    assert out["status"] == "ok"
    assert len(out["backtest"]["folds"]) == 5
    fits = [f["fit_at"] for f in out["backtest"]["folds"]]
    assert fits == sorted(set(fits)) and fits[0] == out["backtest"]["first_origin"]
    assert all(f["error"] is None for f in out["backtest"]["folds"])
    acc = out["accuracy"]
    for m in v.MODELS:
        assert set(acc["horizons"][m]) == set(v.SPLITS)
        assert acc["horizons"][m]["all"]["n_horizons"] == PARAMS["horizon_days"]
    assert set(acc["tenors"]) - {"unit", "naive_m"} == {"7", "14", "30", "60"}
    assert set(out["gates"]) == {
        "rmsse_ex_event_vs_har",
        "rmsse_in_event_vs_har",
        "bias_ordinary_days",
        "pnl_sharpe_vs_naive",
        "event_calendar_complete",
    }
    # The shipped CPI list stops at 2025-10-31; the synthetic days run to 2026-10-13.
    assert out["gates"]["event_calendar_complete"]["status"] == "fail"
    assert out["gates"]["event_calendar_complete"]["value"] == ["cpi"]
    assert out["pnl_proxy"]["n_periods"] >= 2
    assert out["result"] == v.verdict(out["gates"])


def test_event_calendar_gate_passes_when_the_lists_cover_the_sample():
    days, rv, dvol = series()
    covered = FAST | {"events_listed_through": {"fomc": "2026-12-31"}}
    out = v.validate_series("BTC", days, rv, dvol, covered, CFG)
    assert out["gates"]["event_calendar_complete"]["status"] == "pass"
    assert out["gates"]["event_calendar_complete"]["value"] == []


def test_backtest_has_no_look_ahead():
    days, rv, _ = series()
    cut = 400
    later = rv[:cut] + [x * 3.0 for x in rv[cut:]]
    outs = []
    for values in (rv, later):
        arr = np.array(values)
        y = np.log(arr)
        calendar, n_dow, _ = v.build_calendar("BTC", days, FAST, FAST["horizon_days"])
        origins = v.eligible_origins(y, 200)
        outs.append(v.run_backtest(y, arr, calendar, n_dow, origins, FAST, 4))
    early = outs[0].origins < cut
    for m in v.MODELS:
        assert np.array_equal(outs[0].paths[m][early], outs[1].paths[m][early], equal_nan=True)
        assert not np.allclose(outs[0].paths[m][~early], outs[1].paths[m][~early])


def test_sol_has_no_pnl_gate():
    days, rv, _ = series("SOL")
    out = v.validate_series("SOL", days, rv, None, FAST, CFG)
    assert out["pnl_proxy"] is None
    assert out["gates"]["pnl_sharpe_vs_naive"]["status"] == "not_applicable"


def test_btc_without_dvol_has_no_pnl():
    days, rv, _ = series()
    out = v.validate_series("BTC", days, rv, None, FAST, CFG)
    assert out["gates"]["pnl_sharpe_vs_naive"]["status"] == "not_applicable"


def test_short_history_fails():
    days, rv, dvol = series(n=150)
    with pytest.raises(ModelError) as exc:
        v.validate_series("BTC", days, rv, dvol, FAST, CFG)
    assert exc.value.code == "insufficient_history"


def test_folds_out_of_range():
    days, rv, _ = series()
    y = np.log(np.array(rv))
    calendar, n_dow, _ = v.build_calendar("BTC", days, FAST, FAST["horizon_days"])
    origins = v.eligible_origins(y, 200)
    with pytest.raises(ValueError, match="folds"):
        v.run_backtest(y, np.array(rv), calendar, n_dow, origins, FAST, len(origins) + 1)


def test_cli_refuses_a_recording_without_rv_har(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"implementations": ["bs_pricer_black76"]}))
    assert v.main(["--recording", str(tmp_path), "--out", str(tmp_path / "r.json")]) == 2
    assert not (tmp_path / "r.json").exists()
