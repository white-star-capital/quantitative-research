"""parity_deribit on TEST-ONLY synthetic inputs (see synthetic.py) and failure paths
(PARITY_MONITOR_IMPLEMENTATION_GUIDE.md §5 tests). Golden tests on recordings: test_golden.py."""

import json
import math

import pytest

from terminal_models.core.manifest import load_manifest
from terminal_models.core.paths import library_dir
from terminal_models.core.payloads import get_section, records, scalars
from terminal_models.core.registry import null_fees
from terminal_models.core.slots import load_slots
from terminal_models.runner.cli import main as runner
from terminal_models.runner.execute import RunSpec, execute, load_model
from terminal_models.runner.snapshots import SnapshotStore

from . import synthetic as syn

SHIPPED = load_manifest(library_dir() / "parity_deribit" / "manifest.yaml")
MANIFEST = SHIPPED.model_copy(update={"params": SHIPPED.params | {"fees": syn.TEST_FEES}})
SLOT = load_slots()["parity_monitor"]


def spec(asset="BTC", manifest=MANIFEST):
    return RunSpec(SLOT, manifest, "shadow", {"underlying": asset}, syn.AS_OF)


def run(inputs, asset="BTC", manifest=MANIFEST, adapters=None, store=None):
    return execute(spec(asset, manifest), adapters or {}, store, "a1b2c3d", inputs, isolate=False)


def table(env, key):
    return records(get_section(env["payload"], key))


def planted(asset="BTC", bump=0.005, days=31, m=0.0, futures=None, **kw):
    """Inputs with one conversion edge, plus the order books the model asks for."""
    chain = syn.chain_rows(asset)
    futures = futures if futures is not None else syn.futures_rows(asset)
    syn.plant(chain, days, m, asset, bump)
    inputs = syn.inputs(asset, chain=chain, futures=futures, **kw)
    followups = load_model(MANIFEST).followup_inputs({"underlying": asset}, inputs)
    names = [s.params["instrument_name"] for s in followups]
    return inputs, followups, syn.order_books(chain, futures, names)


def test_shipped_manifest_has_null_fees_and_stays_unregistered():
    """§5 M1 blocker: the shipped fee table is null, so the registry would refuse it."""
    assert null_fees(SHIPPED) == sorted(syn.TEST_FEES.keys() - {"schedule_version"})
    registry = (library_dir().parent.parent / "registry.yaml").read_text()
    assert "parity_deribit" not in registry


def test_null_fees_fail_the_run():
    env = run(syn.inputs(), manifest=SHIPPED)
    assert env["status"] == "failed" and env["error"]["code"] == "fees_unavailable"


@pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL", "HYPE"])
def test_clean_chain_has_tight_parity_and_no_opportunities(asset):
    env = run(syn.inputs(asset), asset)
    assert env["status"] == "success", env.get("error")
    assert env["warnings"] == []
    ctx = scalars(get_section(env["payload"], "context"))
    assert ctx["n_candidates"] == 0 and ctx["best_net_edge_ann"] is None
    for row in table(env, "expiries"):
        assert row["status"] == "ok"
        # options priced off the venue forward → implied forward reproduces it
        assert row["f_impl"] == pytest.approx(row["venue_forward"], rel=1e-9)
        assert row["implied_basis_ann"] == pytest.approx(syn.BASIS, abs=5e-4)
    for row in table(env, "bands"):
        assert row["f_impl_bid"] <= row["f_impl_ask"]
    assert ctx["funding_ann"] == pytest.approx(syn.FUNDING_8H * 3 * 365)


def test_hedge_matching_locked_vs_perp():
    env = run(syn.inputs("BTC"))
    kinds = {
        r["expiry_ts"][:10]: (r["hedge_type"], r["future_basis_ann"])
        for r in table(env, "expiries")
    }
    assert kinds["2026-10-17"][0] == "perp_funding" and kinds["2026-10-17"][1] is None
    assert all(k == "locked_future" and fb is not None for d, (k, fb) in kinds.items()
               if d != "2026-10-17")  # fmt: skip
    assert scalars(get_section(env["payload"], "context"))["n_locked_expiries"] == 3


def test_confirmed_locked_future_conversion():
    inputs, followups, books = planted()
    assert [s.name.split(":")[1] for s in followups] == [
        "BTC-14NOV26-60500-C", "BTC-14NOV26-60500-P", "BTC-14NOV26",
    ]  # fmt: skip
    assert all(s.optional and s.params["depth"] == 5 for s in followups)
    env = run(inputs | books)
    assert env["status"] == "success", env.get("error")
    (opp,) = table(env, "opportunities")
    assert (opp["structure"], opp["hedge_type"], opp["confirmed"]) == (
        "conversion", "locked_future", True,
    )  # fmt: skip
    assert opp["net_edge"] == pytest.approx(opp["gross_edge"] - opp["fees"])
    assert opp["net_edge_usd"] == pytest.approx(opp["net_edge"] * syn.SPOT["BTC"])
    assert opp["exec_size_units"] == 5.0  # option legs bind
    assert opp["hedge_contracts"] == round(5 * 60_500 / 10)  # K USD per coin, 10 USD contracts
    assert opp["flags"] == "coin_denominated_edge"
    assert opp["breakeven_funding_ann"] is None
    assert scalars(get_section(env["payload"], "context"))["n_confirmed"] == 1


def test_missing_order_book_leaves_the_row_unconfirmed():
    inputs, _, books = planted()
    books.pop("order_book:BTC-14NOV26")
    env = run(inputs | books)
    (opp,) = table(env, "opportunities")
    assert not opp["confirmed"] and opp["exec_size_units"] is None
    assert [w["code"] for w in env["warnings"]] == ["confirmation_unavailable"]


def test_leg_time_spread_is_flagged():
    inputs, followups, _ = planted()
    names = [s.params["instrument_name"] for s in followups]
    chain, futures = inputs["chain"].data["rows"], inputs["futures"].data["rows"]
    books = syn.order_books(chain, futures, names, spread_s=0.8)  # 1.6 s across three legs
    (opp,) = table(run(inputs | books), "opportunities")
    assert "leg_time_spread" in opp["flags"].split(",")


def test_perp_only_asset_shows_breakeven_funding():
    """SOL/HYPE list only perpetuals (guide §1): every expiry is perp_funding."""
    futures = syn.futures_rows("SOL", dated_days=())
    inputs, _, books = planted("SOL", bump=0.01, futures=futures)
    env = run(inputs | books, "SOL")
    assert {r["hedge_type"] for r in table(env, "expiries")} == {"perp_funding"}
    assert all(r["future_basis_ann"] is None and r["options_vs_future_bps"] is None
               for r in table(env, "expiries"))  # fmt: skip
    opps = table(env, "opportunities")
    # the perp trades near spot while options price off the forward: that basis shows up as
    # edge on perp-hedged rows, which is why they carry breakeven funding, not a locked edge
    assert opps and all(o["hedge_type"] == "perp_funding" for o in opps)
    assert all(o["hedge_instrument"] == "SOL_USDC-PERPETUAL" for o in opps)
    assert all(math.isfinite(o["breakeven_funding_ann"]) for o in opps)
    conv = [o for o in opps if o["structure"] == "conversion"]
    assert conv and all(o["breakeven_funding_ann"] > 0 for o in conv)
    assert not any("coin_denominated_edge" in o["flags"] for o in opps)
    for row in table(env, "expiries"):
        assert row["breakeven_funding_ann"] is not None and math.isfinite(
            row["breakeven_funding_ann"]
        )
    curve = get_section(env["payload"], "carry_curve")
    future = next(s for s in curve["series"] if s["key"] == "future_basis_ann")
    assert set(future["values"]) == {None}
    # perp edges are unlocked: never the headline best edge or confirmed count (guide §11)
    ctx = scalars(get_section(env["payload"], "context"))
    assert ctx["best_net_edge_ann"] is None and ctx["n_confirmed"] == 0
    assert ctx["best_perp_net_edge_ann"] > 0 and ctx["n_confirmed_perp"] >= 1


def test_snapshot_skew_flags_and_never_marks_unconfirmed_rows_executable():
    chain = syn.chain_rows("BTC")
    syn.plant(chain, 31, 0.0, "BTC", 0.005)
    env = run(syn.inputs("BTC", chain=chain, futures_age_s=5))  # futures book 5 s older
    assert "snapshot_skew" in [w["code"] for w in env["warnings"]]
    ctx = get_section(env["payload"], "context")["items"]
    skew = next(i for i in ctx if i["key"] == "snapshot_skew_s")
    assert skew["value"] == pytest.approx(5) and skew["flags"] == ["snapshot_skew"]
    opps = table(env, "opportunities")
    assert opps and not any(o["confirmed"] for o in opps)
    assert all("snapshot_skew" in o["flags"] for o in opps)


def test_insufficient_strikes_omits_the_carry_point():
    thin = syn.chain_rows("BTC", tenors=(31,), moneyness=[-0.3, -0.05, 0.0, 0.3])  # 2 ATM strikes
    chain = syn.chain_rows("BTC", tenors=(3, 10, 95)) + thin
    env = run(syn.inputs("BTC", chain=chain))
    rows = {r["expiry_ts"][:10]: r for r in table(env, "expiries")}
    short = rows["2026-11-14"]
    assert short["status"] == "insufficient_strikes" and short["n_strikes_used"] == 2
    assert short["f_impl"] is None and short["implied_basis_ann"] is None
    assert [w["code"] for w in env["warnings"]] == ["insufficient_strikes"]
    x = get_section(env["payload"], "carry_curve")["x"]["values"]
    assert len(x) == 3 and short["dte"] not in x


def test_every_expiry_thin_fails_the_run():
    chain = syn.chain_rows("BTC", moneyness=[-0.3, 0.0, 0.3])
    env = run(syn.inputs("BTC", chain=chain))
    assert env["error"]["code"] == "insufficient_market_data"


def test_no_pairs_is_insufficient_market_data():
    chain = [r for r in syn.chain_rows("BTC") if r["option_type"] == "call"]
    env = run(syn.inputs("BTC", chain=chain))
    assert env["error"]["code"] == "insufficient_market_data"


def test_fee_table_section_records_both_sides():
    fees = table(run(syn.inputs()), "fees")
    assert [f["side"] for f in fees] == ["taker", "maker"]
    assert fees[0]["option"] == syn.TEST_FEES["option_taker"]
    assert fees[1]["perp"] == syn.TEST_FEES["perp_maker"]
    assert {f["schedule_version"] for f in fees} == {"TEST-ONLY"}


def test_deterministic():
    inputs, _, books = planted()
    first = run(inputs | books)
    for _ in range(9):
        again = run(inputs | books)
        assert json.dumps(again["payload"], sort_keys=True) == json.dumps(
            first["payload"], sort_keys=True
        )


class StubAdapter:
    """Serves the synthetic inputs by name, so execute() runs its two fetch stages."""

    def __init__(self, data):
        self.data, self.calls = data, []

    def fetch(self, input_spec, as_of):
        self.calls.append(input_spec.name)
        return self.data[input_spec.name]


def test_followups_go_through_execute_and_replay(tmp_path, monkeypatch):
    inputs, _, books = planted()
    adapter = StubAdapter(inputs | books)
    monkeypatch.setenv("SNAPSHOT_DIR", str(tmp_path))
    env = execute(spec(), {"deribit": adapter}, SnapshotStore(), "a1b2c3d", isolate=False)
    assert env["status"] == "success", env.get("error")
    assert adapter.calls[:4] == ["chain", "index", "futures", "funding"]
    assert sorted(adapter.calls[4:]) == sorted(books)
    assert {s["name"] for s in env["inputs"]["sources"]} == set(inputs | books)
    assert table(env, "opportunities")[0]["confirmed"]
    # replay from the snapshot store: the follow-up books are in it
    stored = SnapshotStore().load(env["inputs"]["hash"])
    replayed = execute(spec(), {}, None, "a1b2c3d", stored, isolate=False)
    assert replayed["payload"] == env["payload"]


def test_runner_replay_cli(tmp_path, monkeypatch):
    """`runner replay --envelope` needs the implementation registered; it isn't yet (null
    fees), so replay must refuse rather than run it."""
    monkeypatch.setenv("SNAPSHOT_DIR", str(tmp_path))
    env = run(syn.inputs())
    path = tmp_path / "env.json"
    path.write_text(json.dumps(env))
    assert runner(["replay", "--envelope", str(path)]) == 2
