"""rv_har on TEST-ONLY synthetic inputs (see synthetic.py) and failure paths
(RV_FORECAST_IMPLEMENTATION_GUIDE.md §12, P3 tests). Golden tests on recordings: test_golden.py.

Most tests run a FAST manifest (shorter history and z window) so the expanding-origin loop stays
cheap; the payload shape and every identity in §12 are the same at the shipped values, and
test_shipped_params_run_within_budget runs those. FAST keeps the shipped event calendar.
"""

import json
import math
import time
from datetime import UTC, datetime, timedelta

import pytest
import yaml

from terminal_models.core.manifest import load_manifest
from terminal_models.core.paths import library_dir
from terminal_models.core.payloads import get_section, records
from terminal_models.core.slots import load_slots
from terminal_models.core.validation import ValidationFailure, vrp_band_ordering
from terminal_models.runner.cli import main as runner
from terminal_models.runner.execute import RunSpec, execute, load_model
from terminal_models.runner.snapshots import SnapshotStore

from . import synthetic as syn

SHIPPED = load_manifest(library_dir() / "rv_har" / "manifest.yaml")
FAST = SHIPPED.model_copy(
    update={
        "params": SHIPPED.params
        | {
            "history_days": 60,
            "z_window_days": 40,
            "z_min_obs": 20,
            "min_history_days": {a: 200 for a in syn.SPOT},
        }
    }
)
SLOT = load_slots()["vrp_signal"]
N_DAYS = 400


def spec(asset="BTC", manifest=FAST):
    return RunSpec(SLOT, manifest, "shadow", {"underlying": asset}, syn.AS_OF)


def run(inputs, asset="BTC", manifest=FAST):
    return execute(spec(asset, manifest), {}, None, "a1b2c3d", inputs, isolate=False)


def ok(asset="BTC", **kw):
    env = run(syn.inputs(asset, n_days=kw.pop("n_days", N_DAYS), **kw), asset)
    assert env["status"] == "success", env.get("error")
    return env


def signal(env):
    return {r["tenor_days"]: r for r in records(get_section(env["payload"], "signal"))}


def ctx_items(env):
    return {i["key"]: i for i in get_section(env["payload"], "context")["items"]}


def named(section, key):
    return next(s["values"] for s in section["series"] if s["key"] == key)


def test_shipped_manifest_is_registered_as_shadow():
    registry = yaml.safe_load((library_dir().parent.parent / "registry.yaml").read_text())
    slot = registry["slots"]["vrp_signal"]
    assert slot["production"] is None
    assert slot["shadow"] == [{"implementation": "rv_har", "version": SHIPPED.version}]
    assert [s["underlying"] for s in slot["scopes"]] == ["BTC", "ETH", "SOL", "HYPE"]


def test_required_inputs_follow_params():
    model = load_model(SHIPPED)
    btc = {s.name: s for s in model.required_inputs({"underlying": "BTC"})}
    assert list(btc) == ["chain", "index", "rv_daily", "dvol_history"]
    assert btc["chain"].params == {"kind": "chain", "underlying": "BTC"}  # shared with bs_pricer
    assert btc["rv_daily"].source == "deribit" and btc["rv_daily"].params["resolution"] == "60"
    assert btc["dvol_history"].optional and not btc["rv_daily"].optional
    hype = {s.name: s for s in model.required_inputs({"underlying": "HYPE"})}
    assert list(hype) == ["chain", "index", "rv_daily"]
    assert hype["rv_daily"].source == "hyperliquid"


def test_payload_sections_and_identities():
    env = ok()
    assert env["warnings"] == []
    payload = env["payload"]
    assert [s["key"] for s in payload["sections"]] == [
        "signal", "context", "cone", "history", "events",
    ]  # fmt: skip
    rows = signal(env)
    assert list(rows) == [7, 14, 30, 60]
    cone = get_section(payload, "cone")
    assert cone["x"]["values"] == list(range(1, 61))
    for t, r in rows.items():
        assert r["rv_low"] <= r["rv_forecast"] <= r["rv_high"]
        assert r["comp_baseline"] + r["comp_seasonal"] + r["comp_event"] == pytest.approx(
            1, abs=1e-12
        )
        assert min(r["comp_baseline"], r["comp_seasonal"], r["comp_event"]) >= 0
        assert r["vrp"] == pytest.approx(r["iv_impl"] - r["rv_forecast"])
        assert r["rich"] == (r["iv_impl"] > r["rv_high"])
        assert r["event_hump"] == (r["comp_event"] > 0.35)
        assert named(cone, "rv_forecast")[t - 1] == r["rv_forecast"]
        assert named(cone, "rv_low")[t - 1] == r["rv_low"]
        assert named(cone, "atm_iv")[t - 1] == r["iv_impl"]
    assert sum(v is not None for v in named(cone, "atm_iv")) == 4
    # cumulative cone at T is the mean of the daily path's variance over days 1..T
    path = named(cone, "rv_path")
    assert rows[30]["rv_forecast"] == pytest.approx(
        math.sqrt(sum(v * v for v in path[:30]) / 30), rel=1e-12
    )

    hist = get_section(payload, "history")
    assert len(hist["t"]) == 60 and hist["t"][-1] == "2026-10-14T00:00:00Z"
    items = ctx_items(env)
    assert items["last_complete_day"]["value"] == int(syn.AS_OF.replace(hour=0).timestamp())
    assert named(hist, "rv_forecast_30d")[-1] == rows[30]["rv_forecast"]
    assert named(hist, "vrp_30d")[-1] == items["vrp_30d_dvol"]["value"]
    assert named(hist, "vrp_z_30d")[-1] == rows[30]["vrp_z"]
    matured, fc = named(hist, "rv_forecast_30d_matured"), named(hist, "rv_forecast_30d")
    assert matured[30:] == fc[:30]
    assert items["n_days"]["value"] == N_DAYS and items["n_days_missing"]["value"] == 0
    assert items["signal_enabled"]["value"] is True


def test_z_on_the_dvol_basis_btc_30d_only():
    rows = signal(ok("BTC"))
    assert rows[30]["vrp_z"] is not None and rows[30]["flags"] == "z_basis_dvol"
    for t in (7, 14, 60):
        assert rows[t]["vrp_z"] is None and rows[t]["flags"] == "z_unavailable"
        assert rows[t]["signal_ok"] is False
    assert rows[30]["signal_ok"] == (not rows[30]["event_hump"])


@pytest.mark.parametrize("asset", ["SOL", "HYPE"])
def test_z_null_off_basis(asset):
    env = ok(asset)
    for r in signal(env).values():
        assert r["vrp_z"] is None and r["flags"] == "z_unavailable" and not r["signal_ok"]
    items = ctx_items(env)
    assert items["dvol"]["value"] is None and items["dvol"]["flags"] == ["dvol_unavailable"]
    assert items["vrp_z_n"]["value"] == 0
    assert items["vrp_z_mean"]["flags"] == ["z_unavailable"]
    hist = get_section(env["payload"], "history")
    assert set(named(hist, "dvol")) == {None} and set(named(hist, "vrp_z_30d")) == {None}


def test_missing_dvol_input_is_insufficient_history_for_z():
    rows = signal(ok("BTC", with_dvol=False))
    assert rows[30]["vrp_z"] is None and rows[30]["flags"] == "z_insufficient_history"


def test_extrapolation_flags_pass_through():
    chain = syn.chain_rows("BTC", tenors=(10, 31))
    rows = signal(ok("BTC", chain=chain))
    assert rows[7]["flags"].split(",") == ["tenor_extrapolated_short", "z_unavailable"]
    assert rows[60]["flags"].split(",") == ["tenor_extrapolated_long", "z_unavailable"]
    assert rows[14]["flags"] == "z_unavailable"


def test_lagging_rv_input_disables_the_signal():
    env = ok("BTC", lag_days=2)
    item = ctx_items(env)["signal_enabled"]
    assert item["value"] is False and item["flags"] == ["rv_input_lagging"]
    assert not any(r["signal_ok"] for r in signal(env).values())


def test_events_in_horizon():
    env = ok()
    events = records(get_section(env["payload"], "events"))
    assert events == sorted(events, key=lambda e: (e["ts"], e["type"]))
    opex = [e for e in events if e["source"] == "rule"]
    assert opex and all(e["scope"] == "GLOBAL" for e in opex)
    day1 = syn.AS_OF.date()
    for e in events:
        assert e["horizon_day"] == (datetime.fromisoformat(e["ts"]).date() - day1).days + 1
        assert (
            e["horizon_day"] - e["window_before"] <= 60
            and e["horizon_day"] + e["window_after"] >= 1
        )
    for t, r in signal(env).items():
        expected = any(
            e["horizon_day"] - e["window_before"] <= t and e["horizon_day"] + e["window_after"] >= 1
            for e in events
        )
        assert r["event_in_horizon"] == expected, t


def test_insufficient_history_fails():
    env = run(syn.inputs("BTC", n_days=150))
    assert env["status"] == "failed" and env["error"]["code"] == "insufficient_history"


def test_missing_day_at_the_origin_fails():
    rv = syn.rv_series(N_DAYS)
    rv[-1] = None  # C itself: the lag-1 window has no day
    env = run(syn.inputs("BTC", rv=rv))
    assert env["error"]["code"] == "insufficient_history"
    assert "no forecast origin" in env["error"]["message"]


def test_isolated_missing_days_in_the_lag_windows_still_forecast():
    rv = syn.rv_series(N_DAYS)
    rv[-3] = rv[-10] = None  # 4 of 5 and 20 of 22: enough for the lag means (rv.LAG_MIN_COVERAGE)
    env = run(syn.inputs("BTC", rv=rv))
    assert env["status"] == "success", env.get("error")
    assert ctx_items(env)["n_days_missing"]["value"] == 2
    rv[-2] = None  # now 3 of 5
    assert run(syn.inputs("BTC", rv=rv))["error"]["code"] == "insufficient_history"


def test_incomplete_event_calendar_disables_the_signal():
    shipped_calendar = FAST.model_copy(
        update={"params": FAST.params | {"events_listed_through": {"fomc": "2026-12-31",
                                                                    "cpi": "2025-10-31"}}}
    )  # fmt: skip
    env = run(syn.inputs("BTC", n_days=N_DAYS), manifest=shipped_calendar)
    assert env["status"] == "success", env.get("error")
    item = ctx_items(env)["signal_enabled"]
    assert item["value"] is False and item["flags"] == ["event_calendar_incomplete"]
    for r in signal(env).values():
        assert not r["signal_ok"] and "event_calendar_incomplete" in r["flags"].split(",")
    assert ctx_items(env)["calendar_days_left"]["value"] < 0


def test_no_listed_event_types_leaves_the_gate_open():
    no_lists = FAST.model_copy(update={"params": FAST.params | {"events_listed_through": {}}})
    env = run(syn.inputs("BTC", n_days=N_DAYS), manifest=no_lists)
    assert env["status"] == "success", env.get("error")
    items = ctx_items(env)
    assert items["calendar_days_left"]["value"] is None
    assert items["calendar_days_left"]["flags"] == ["no_listed_types"]
    assert items["signal_enabled"]["value"] is True


@pytest.mark.parametrize(("offset", "warns"), [(8.0, True), (0.0, False), (None, False)])
def test_dvol_close_offset_warns(offset, warns):
    """TEST-ONLY: the synthetic DVOL series tagged as the adapter tags 1D candles that close at
    `offset` h UTC (plan §3c); only a non-zero offset warns."""
    inputs = syn.inputs("BTC", n_days=N_DAYS)
    inputs["dvol_history"].data["close_offset_h"] = offset
    env = run(inputs)
    assert env["status"] == "success", env.get("error")
    codes = [w["code"] for w in env["warnings"]]
    assert codes == (["dvol_offset"] if warns else [])
    if warns:
        assert env["warnings"][0]["message"] == "DVOL 1D candles close at 8h UTC, not 00:00"


@pytest.mark.parametrize(
    ("as_of", "left", "live_tenors", "enabled"),
    [
        # C = 2026-10-29 00:00; horizon days 1..60 end 2026-12-27, plus the widest window (3)
        # → 2026-12-30; CPI is listed through 2027-01-13, so 14 days of margin.
        (datetime(2026, 10, 29, 12, tzinfo=UTC), 14, {7, 14, 30, 60}, True),
        (datetime(2026, 11, 12, 12, tzinfo=UTC), 0, {7, 14, 30, 60}, True),  # last day for 60d
        (datetime(2026, 11, 13, 12, tzinfo=UTC), -1, {7, 14, 30}, True),  # 60d drops first
        (datetime(2027, 1, 4, 12, tzinfo=UTC), -53, {7}, True),  # last day for 7d
        (datetime(2027, 1, 5, 12, tzinfo=UTC), -54, set(), False),  # needs the 2027 CPI dates
    ],
)
def test_shipped_calendar_gate_and_days_left(monkeypatch, as_of, left, live_tenors, enabled):
    """The shipped events_listed_through at real dates: each tenor loses calendar coverage on
    its own, longest first; the signal is disabled only once the shortest has. The countdown
    tracks the longest tenor."""
    monkeypatch.setattr(syn, "AS_OF", as_of)
    monkeypatch.setattr(syn, "LAST_DAY", (as_of - timedelta(days=1)).date())
    env = run(syn.inputs("BTC", n_days=N_DAYS))
    assert env["status"] == "success", env.get("error")
    items = ctx_items(env)
    assert items["calendar_days_left"]["value"] == left
    assert items["calendar_days_left"]["unit"]["quantity"] == "count"
    item = items["signal_enabled"]
    assert item["value"] is enabled
    assert ("event_calendar_incomplete" in item.get("flags", [])) is (not enabled)
    for t, r in signal(env).items():
        incomplete = "event_calendar_incomplete" in r["flags"].split(",")
        assert incomplete is (t not in live_tenors), t
        if incomplete:
            assert not r["signal_ok"]


def test_shipped_params_run_within_budget():
    """Shipped history_days, z window and lookback: the hourly pass runs every vrp scope after
    bs_pricer and parity, so a run must stay far inside the timeout (VRP-A-6)."""
    start = time.monotonic()
    env = run(syn.inputs("BTC", n_days=1000), manifest=SHIPPED)
    elapsed = time.monotonic() - start
    assert env["status"] == "success", env.get("error")
    assert len(get_section(env["payload"], "history")["t"]) == 365
    assert elapsed < SHIPPED.timeout_seconds / 4, elapsed


def test_missing_days_elsewhere_are_counted_not_filled():
    rv = syn.rv_series(N_DAYS)
    for i in (50, 51, 200):
        rv[i] = None
    env = run(syn.inputs("BTC", rv=rv))
    assert env["status"] == "success", env.get("error")
    assert ctx_items(env)["n_days_missing"]["value"] == 3


def test_degenerate_forecast_fails():
    rv = [v * 400 for v in syn.rv_series(N_DAYS)]  # ~10 annualized: outside rv_sanity
    env = run(syn.inputs("BTC", rv=rv))
    assert env["status"] == "failed" and env["error"]["code"] == "degenerate_forecast"


def test_no_chain_expiry_fails():
    env = run(syn.inputs("BTC", chain=[]))
    assert env["error"]["code"] == "insufficient_market_data"


def test_band_ordering_check_rejects_a_crossed_row():
    env = ok()
    assert vrp_band_ordering(env["payload"], FAST.params) == []
    table = get_section(env["payload"], "signal")
    col = [c["key"] for c in table["columns"]].index("rv_low")
    table["rows"][0][col] = table["rows"][0][col + 1] * 1.01  # rv_low above rv_high
    with pytest.raises(ValidationFailure) as exc:
        vrp_band_ordering(env["payload"], FAST.params)
    assert exc.value.code == "vrp_band_ordering"


def test_context_baselines():
    items = ctx_items(ok())
    for key in ("rv_trailing_30d", "rv_ewma_30d", "rv_har_30d", "fit_r2", "resid_sd"):
        assert isinstance(items[key]["value"], float) and items[key]["value"] > 0, key
    assert items["vrp_z_n"]["value"] == 40
    assert items["event_hump_share"]["value"] == 0.35


def test_deterministic():
    inputs = syn.inputs("BTC", n_days=N_DAYS)
    first = json.dumps(run(inputs)["payload"], sort_keys=True)
    for _ in range(9):
        assert json.dumps(run(inputs)["payload"], sort_keys=True) == first


class StubAdapter:
    """Serves the synthetic inputs by name, so execute() runs the fetch stage."""

    def __init__(self, data):
        self.data, self.calls = data, []

    def fetch(self, input_spec, as_of):
        self.calls.append((input_spec.source, input_spec.name))
        return self.data[input_spec.name]


def test_execute_fetches_and_replays_from_the_snapshot_store(tmp_path, monkeypatch):
    monkeypatch.setenv("SNAPSHOT_DIR", str(tmp_path))
    adapter = StubAdapter(syn.inputs("BTC", n_days=N_DAYS))
    env = execute(spec(), {"deribit": adapter}, SnapshotStore(), "a1b2c3d", isolate=False)
    assert env["status"] == "success", env.get("error")
    assert [c[1] for c in adapter.calls] == ["chain", "index", "rv_daily", "dvol_history"]
    stored = SnapshotStore().load(env["inputs"]["hash"])
    replayed = execute(spec(), {}, None, "a1b2c3d", stored, isolate=False)
    assert replayed["payload"] == env["payload"]


def test_stale_input_fails():
    env = run(syn.inputs("BTC", n_days=N_DAYS, age_s=int(timedelta(minutes=5).total_seconds())))
    assert env["error"]["code"] == "stale_input"


def test_runner_replay_cli(tmp_path, monkeypatch):
    """`runner replay --envelope` re-runs the registered manifest on the stored inputs."""
    monkeypatch.setenv("SNAPSHOT_DIR", str(tmp_path))
    adapter = StubAdapter(syn.inputs("SOL", n_days=N_DAYS))
    env = execute(spec("SOL", SHIPPED), {"hyperliquid": adapter, "deribit": adapter},
                  SnapshotStore(), "a1b2c3d", isolate=False)  # fmt: skip
    assert env["status"] == "success", env.get("error")
    assert ("hyperliquid", "rv_daily") in adapter.calls
    path = tmp_path / "env.json"
    path.write_text(json.dumps(env))
    assert runner(["replay", "--envelope", str(path)]) == 0
