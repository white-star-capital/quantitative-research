"""bs_pricer_black76 behavior on synthetic inputs (see synthetic.py) and failure paths.
Golden tests on recorded Deribit snapshots: test_golden.py."""

import math

import pytest

from terminal_models.adapters.base import AdapterError
from terminal_models.core.manifest import load_manifest
from terminal_models.core.paths import library_dir
from terminal_models.core.payloads import get_section, records, scalars
from terminal_models.core.quant.black76 import black76
from terminal_models.core.quant.evaluate import (
    EvaluateError,
    Request,
    evaluate,
    surface_from_payload,
)
from terminal_models.core.slots import load_slots
from terminal_models.runner.execute import RunSpec, execute

from . import synthetic as syn

MANIFEST = load_manifest(library_dir() / "bs_pricer_black76" / "manifest.yaml")
SLOT = load_slots()["bs_pricer"]


def run(inputs=None, underlying="BTC", manifest=MANIFEST, isolate=False, adapters=None):
    spec = RunSpec(SLOT, manifest, "shadow", {"underlying": underlying}, syn.AS_OF)
    return execute(
        spec,
        adapters or {},
        None,
        "a1b2c3d",
        inputs if inputs is not None else syn.inputs(),
        isolate=isolate,
    )


@pytest.fixture(scope="module")
def envelope():
    env = run()
    assert env["status"] == "success", env.get("error")
    return env


def test_sections_and_counts(envelope):
    payload = envelope["payload"]
    assert [s["key"] for s in payload["sections"]] == [
        "context",
        "term_structure",
        "smile",
        "surface",
        "expiries",
        "chain",
    ]
    ctx = scalars(get_section(payload, "context"))
    assert ctx["n_expiries"] == 4
    # Synthetic wings price near 1e-12, where the 1e-10 price tolerance can't pin a bid IV;
    # those few nodes are not "good". Real USD prices don't get that small.
    assert ctx["n_nodes_good"] >= 0.95 * ctx["n_nodes"]
    assert ctx["dvol"] == 0.62 and ctx["usd_rate"] == 0.0
    assert envelope["warnings"] == []


def test_atm_and_skew_follow_the_smile(envelope):
    payload = envelope["payload"]
    ctx = scalars(get_section(payload, "context"))
    # ATM IV on a listed node grid is close to smile(0) = 0.6 (strikes straddle the forward).
    assert ctx["atm_iv_30d"] == pytest.approx(0.6, abs=0.01)
    assert ctx["rr25_30d"] < 0 < ctx["bf25_30d"]  # put wing richer; both wings above ATM
    assert ctx["rv_30d"] == pytest.approx(0.004 * math.sqrt(8760))
    assert ctx["vrp_30d"] == pytest.approx(ctx["atm_iv_30d"] - ctx["rv_30d"])
    assert get_section(payload, "context")["items"][2]["flags"] == ["tenor_extrapolated_short"]


def test_chain_reproduces_marks_and_parity(envelope):
    payload = envelope["payload"]
    chain = records(get_section(payload, "chain"))
    nodes = [r for r in chain if r["is_node"]]
    assert nodes and {r["node_quality"] for r in nodes} <= {"good", "wide", "one_sided"}
    for r in nodes:
        assert r["fair_usd"] == pytest.approx(r["venue_mark_usd"], rel=1e-12)
    itm = [r for r in chain if not r["is_node"]]
    assert itm and all(r["node_quality"] == "interpolated" for r in itm)
    for r in chain:
        assert 0.02 <= abs(r["delta"]) <= 0.98


def test_display_grids_never_extrapolate(envelope):
    smile = get_section(envelope["payload"], "smile")
    expiries = records(get_section(envelope["payload"], "expiries"))
    for s, e in zip(smile["series"], expiries, strict=True):
        for k, v in zip(smile["x"]["values"], s["values"], strict=True):
            assert (v is None) == (k < e["k_min"] or k > e["k_max"])


def test_thin_expiry_dropped_with_warning():
    rows = [
        r
        for r in syn.rows()
        if not r["expiry_ts"].startswith("2026-10-16") or r["strike"] in (100.0, 102.5)
    ]
    env = run(syn.inputs(rows, with_dvol=False), underlying="SOL")
    assert env["status"] == "success"
    assert [w["code"] for w in env["warnings"]] == ["expiry_dropped_insufficient_nodes"]
    assert scalars(get_section(env["payload"], "context"))["n_expiries"] == 3


def test_no_surviving_expiry_fails_the_run():
    env = run(syn.inputs(syn.rows(strikes=[100.0, 102.5]), with_dvol=False), underlying="SOL")
    assert env["status"] == "failed" and env["payload"] is None
    assert env["error"]["code"] == "insufficient_market_data"


def test_units_bug_is_a_hard_failure():
    env = run(syn.inputs(syn.rows(mark_scale=1.1)))
    assert env["status"] == "failed"
    assert env["error"]["code"] == "mark_reproduction"


def test_stale_input_fails():
    env = run(syn.inputs(age_s=MANIFEST.max_input_age_seconds + 1))
    assert env["error"]["code"] == "stale_input"


def test_venue_outage_fails_with_venue_code():
    class Down:
        def fetch(self, spec, as_of):
            raise AdapterError("venue_unavailable", "deribit unreachable after retries")

    spec = RunSpec(SLOT, MANIFEST, "shadow", {"underlying": "ETH"}, syn.AS_OF)
    env = execute(spec, {"deribit": Down()}, None, "a1b2c3d", isolate=False)
    assert (env["status"], env["error"]["code"]) == ("failed", "venue_unavailable")


def test_deterministic_and_isolation_matches_in_process(envelope):
    for _ in range(10):
        assert run()["payload"] == envelope["payload"]
    assert run(isolate=True)["payload"] == envelope["payload"]


# ---- evaluate (§6.2) on the produced run --------------------------------------------------

CONFIG = SLOT.evaluation.model_dump()


@pytest.fixture(scope="module")
def stored(envelope):
    return surface_from_payload(
        envelope["run_id"], envelope["as_of"], envelope["scope"], envelope["payload"]
    )


def test_evaluate_on_listed_contract_matches_chain(envelope, stored):
    chain = records(get_section(envelope["payload"], "chain"))
    row = next(
        r
        for r in chain
        if r["is_node"] and r["option_type"] == "call" and r["expiry_ts"].startswith("2026-11-13")
    )
    req = Request("call", syn.expiry_at(31), row["strike"], size=2)
    out = evaluate(stored, req, CONFIG, run_stale=False)
    assert out["market"]["interpolation"] == "exact"
    assert out["fair"]["per_unit_usd"] == pytest.approx(row["fair_usd"], rel=1e-12)
    assert out["fair"]["total_usd"] == pytest.approx(2 * row["fair_usd"])
    assert out["quote"] is None and out["flags"] == []
    assert len(out["distribution"]["x"]) == 200


def test_evaluate_quote_at_fair_has_zero_edge(stored):
    when = syn.expiry_at(20).replace(hour=23, minute=59)
    base = evaluate(stored, Request("put", when, 95.0), CONFIG, run_stale=False)
    assert base["market"]["interpolation"] == "interpolated"
    fair = base["fair"]["per_unit_usd"]
    for side in ("sell", "buy"):
        out = evaluate(
            stored,
            Request("put", when, 95.0, premium=fair, premium_unit="usd_per_unit", side=side),
            CONFIG,
            run_stale=False,
        )
        q = out["quote"]
        assert q["edge_per_unit_usd"] == pytest.approx(0, abs=1e-9)
        assert q["vol_edge_vp"] == pytest.approx(0, abs=1e-6)
        assert q["pct_of_fair"] == pytest.approx(1)
        assert q["screen_position"] == "inside"


def test_evaluate_quote_edges_are_good_for_us(stored):
    when = syn.expiry_at(31)
    rich = Request(
        "call",
        when,
        110.0,
        size=10,
        premium=50.0,
        premium_unit="usd_total",
        side="sell",
        spot_ref=101.5,
    )
    q = evaluate(stored, rich, CONFIG, run_stale=True)
    assert q["quote"]["edge_total_usd"] > 0 and q["quote"]["vol_edge_vp"] > 0
    assert {f["code"] for f in q["flags"]} >= {"spot_moved", "run_stale"}


def test_evaluate_unsolvable_quote_reports_reason(stored):
    req = Request(
        "call", syn.expiry_at(31), 80.0, premium=1.0, premium_unit="usd_per_unit", side="buy"
    )
    q = evaluate(stored, req, CONFIG, run_stale=False)["quote"]
    assert (q["iv_quote"], q["iv_quote_status"], q["vol_edge_vp"]) == (
        None,
        "below_intrinsic",
        None,
    )


@pytest.mark.parametrize(
    "req, code",
    [
        (Request("call", syn.AS_OF, 100.0), "expiry_in_past"),
        (Request("call", syn.expiry_at(400), 100.0), "tenor_out_of_range"),
        (Request("call", syn.expiry_at(31), 10.0), "strike_out_of_range"),
        (
            Request("call", syn.expiry_at(31), 100.0, premium=1.0, premium_unit="usd_per_unit"),
            "premium_requires_side",
        ),
        (
            Request(
                "call", syn.expiry_at(31), 100.0, premium=1.0, side="sell", premium_unit="pct_spot"
            ),
            "pct_spot_requires_spot_ref",
        ),
        (
            Request(
                "call",
                syn.expiry_at(31),
                100.0,
                premium=-1.0,
                side="sell",
                premium_unit="usd_per_unit",
            ),
            "invalid_premium",
        ),
    ],
)
def test_evaluate_errors(stored, req, code):
    with pytest.raises(EvaluateError) as exc:
        evaluate(stored, req, CONFIG, run_stale=False)
    assert exc.value.code == code


def test_evaluate_off_grid_strike_is_flagged(stored):
    out = evaluate(stored, Request("call", syn.expiry_at(31), 200.0), CONFIG, run_stale=False)
    assert "strike_extrapolated" in {f["code"] for f in out["flags"]}
    assert out["market"]["interpolation"] == "extrapolated"


def test_black76_reference_used_for_screen(stored):
    out = evaluate(stored, Request("call", syn.expiry_at(31), 110.0), CONFIG, run_stale=False)
    m = out["market"]
    contract = out["contract"]
    expected = black76(m["forward"], 110.0, contract["T"], m["iv_bid"], 0.0, True, out["spot"])
    assert m["screen_bid_usd"] == pytest.approx(expected["price"])
