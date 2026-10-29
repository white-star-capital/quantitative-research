"""Recording → replay → fixtures → parity vectors, end to end and offline.

Responses are Deribit-shaped bodies built by inverting `chains.normalize` over the TEST-ONLY
synthetic chain (library/bs_pricer_black76/tests/synthetic.py): BTC as an inverse coin book, HYPE
as a linear USDC book. rv_har's paged candles (Deribit and Hyperliquid) and DVOL history come from
the TEST-ONLY `rv_close` / `dvol_close` paths below. They exercise the plumbing a real recording
goes through; nothing here is written to contracts/fixtures (D-10).
"""

import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest

from terminal_models.adapters.deribit.client import DeribitClient
from terminal_models.adapters.http import HttpSource
from terminal_models.adapters.hyperliquid.client import HyperliquidClient
from terminal_models.core.paths import contracts_dir
from terminal_models.core.validation import (
    ValidationFailure,
    assert_no_nan_inf,
    validate_envelope,
    validate_payload_full,
)
from terminal_models.library.bs_pricer_black76.tests import synthetic as syn
from terminal_models.runner.cli import main as runner
from terminal_models.runner.snapshots import SnapshotStore
from terminal_models.tools import build_fixtures, gen_eval_vectors
from terminal_models.tools.record_deribit import record

SERVER_TIME = syn.AS_OF - timedelta(seconds=5)
# The recording starts 150 s before the venues' clock reads SERVER_TIME: the time BTC's inputs
# take before later assets are fetched, past rv_har's 120 s max input age.
RECORDING_STARTED = SERVER_TIME - timedelta(seconds=150)


class RecorderClock(datetime):
    """TEST-ONLY: record_deribit's `datetime`, with now() pinned to RECORDING_STARTED."""

    @classmethod
    def now(cls, tz=None):
        return RECORDING_STARTED.astimezone(tz)


US_OUT = int(SERVER_TIME.timestamp() * 1_000_000) + 123_456
BOOK = {"BTC": ("BTC", "reversed"), "HYPE": ("USDC", "linear")}


def ms(iso: str) -> int:
    return int(syn.datetime.fromisoformat(iso).timestamp() * 1000)


def options(asset: str) -> tuple[list[dict], list[dict]]:
    """Synthetic rows → (get_instruments, get_book_summary_by_currency) results."""
    currency, kind = BOOK[asset]
    meta, summary = [], []
    for row in syn.rows():
        name = row["instrument_name"].replace("SYN", asset)
        to_coin = row["forward"] if kind == "reversed" else 1.0

        def price(usd, to_coin=to_coin):
            return 0.0 if usd is None else usd / to_coin  # Deribit sends 0 for an empty side

        meta.append(
            {
                "instrument_name": name,
                "kind": "option",
                "base_currency": asset,
                "settlement_currency": currency,
                "instrument_type": kind,
                "strike": row["strike"],
                "option_type": row["option_type"],
                "contract_size": row["contract_size"],
                "expiration_timestamp": ms(row["expiry_ts"]),
            }
        )
        summary.append(
            {
                "instrument_name": name,
                "underlying_price": row["forward"],
                "mark_price": price(row["mark_usd"]),
                "bid_price": price(row["bid_usd"]),
                "ask_price": price(row["ask_usd"]),
                "mark_iv": row["mark_iv"] * 100,
                "open_interest": row["open_interest"],
            }
        )
    return meta, summary


DATED = syn.expiry_at(31)
DATED_F = syn.SPOT * math.exp(syn.BASIS * (DATED - syn.AS_OF).total_seconds() / (365 * 86400))
PERP_MS = 32_503_708_800_000
FUTURES = {
    "BTC": [{"instrument_name": "BTC-PERPETUAL", "base_currency": "BTC",
             "settlement_currency": "BTC", "settlement_period": "perpetual", "is_active": True,
             "contract_size": 10.0, "expiration_timestamp": PERP_MS, "instrument_type": "reversed"},
            {"instrument_name": f"BTC-{DATED:%d%b%y}".upper(), "base_currency": "BTC",
             "settlement_currency": "BTC", "settlement_period": "month", "is_active": True,
             "contract_size": 10.0, "expiration_timestamp": int(DATED.timestamp() * 1000),
             "instrument_type": "reversed"}],
    "USDC": [{"instrument_name": "HYPE_USDC-PERPETUAL", "base_currency": "HYPE",
              "settlement_currency": "USDC", "settlement_period": "perpetual", "is_active": True,
              "contract_size": 1.0, "expiration_timestamp": PERP_MS, "instrument_type": "linear"}],
}  # fmt: skip
FUTURE_PRICES = {"BTC-PERPETUAL": syn.SPOT, "HYPE_USDC-PERPETUAL": syn.SPOT,
                 f"BTC-{DATED:%d%b%y}".upper(): DATED_F}  # fmt: skip


def futures_summary(currency: str) -> list[dict]:
    out = []
    for f in FUTURES[currency]:
        px = FUTURE_PRICES[f["instrument_name"]]
        out.append({"instrument_name": f["instrument_name"], "bid_price": px * 0.9999,
                    "ask_price": px * 1.0001, "mark_price": px})  # fmt: skip
    return out


def order_book(name: str) -> dict:
    """Top of book = the summary's bid/ask; options 5 contracts deep, futures deep."""
    rows = [s for a in BOOK for s in options(a)[1]] + [
        r for c in FUTURES for r in futures_summary(c)
    ]
    row = next(r for r in rows if r["instrument_name"] == name)
    depth = 1e9 if name in FUTURE_PRICES else 5.0
    bids = [[row["bid_price"], depth]] if row["bid_price"] else []
    return {"instrument_name": name, "timestamp": US_OUT // 1000 - 500, "bids": bids,
            "asks": [[row["ask_price"], depth]]}  # fmt: skip


INDEX = {"btc_usd": syn.SPOT, "hype_usdc": syn.SPOT, "btcdvol_usdc": 62.0}
PERPS = {"BTC-PERPETUAL", "HYPE_USDC-PERPETUAL"}


def result_for(method: str, p: httpx.QueryParams):
    if method == "get_instruments":
        if p["kind"] == "future":
            return FUTURES[p["currency"]]
        assets = [a for a, (c, _) in BOOK.items() if c == p["currency"]]
        return [m for a in assets for m in options(a)[0]]
    if method == "get_book_summary_by_currency" and p["kind"] == "future":
        return futures_summary(p["currency"])
    if method == "ticker":
        return {"instrument_name": p["instrument_name"], "current_funding": 0.0,
                "funding_8h": 0.0001, "timestamp": US_OUT // 1000}  # fmt: skip
    if method == "get_order_book":
        return order_book(p["instrument_name"])
    if method == "get_book_summary_by_currency":
        assets = [a for a, (c, _) in BOOK.items() if c == p["currency"]]
        return [s for a in assets for s in options(a)[1]]
    if method == "get_index_price":
        return {"index_price": INDEX[p["index_name"]]}
    if method == "get_tradingview_chart_data" and int(p["end_timestamp"]) % DAY_MS == DAY_MS - 1:
        assert p["instrument_name"] in PERPS  # an rv_daily page: whole UTC days (rv_har)
        step = int(p["resolution"]) * 60_000
        ticks = list(range(-(-int(p["start_timestamp"]) // step) * step,
                           int(p["end_timestamp"]) + 1, step))  # fmt: skip
        return {"status": "ok", "ticks": ticks, "close": [rv_close(t) for t in ticks]}
    if method == "get_tradingview_chart_data":
        assert p["instrument_name"] in PERPS
        closes = syn.closes()
        end = int(p["end_timestamp"]) // 3_600_000 * 3_600_000
        ticks = [end - 3_600_000 * (len(closes) - 1 - i) for i in range(len(closes))]
        return {"status": "ok", "ticks": ticks, "close": closes}
    if method == "get_volatility_index_data" and p["resolution"] == "1D":
        days = range(-(-int(p["start_timestamp"]) // DAY_MS) * DAY_MS,
                     int(p["end_timestamp"]) + 1, DAY_MS)  # fmt: skip
        return {"data": [[d, 60, 62, 58, dvol_close(d)] for d in days], "continuation": None}
    if method == "get_volatility_index_data":
        return {"data": [[ms(syn.AS_OF.strftime("%Y-%m-%dT%H:%M:%SZ")), 61, 63, 60, 62]]}
    raise AssertionError(method)


DAY_MS = 86_400_000


def rv_close(t_ms: int) -> float:
    """TEST-ONLY bar close at open time t (ms): a deterministic path whose daily variance moves
    from day to day, so rv_har has something to fit. Not a price series."""
    hour = t_ms // 3_600_000
    day = hour // 24
    amp = 0.006 * (2.0 + math.sin(day * 0.37) + 0.5 * math.sin(day * 1.91))
    return syn.SPOT * math.exp(amp * math.sin(hour * 1.3))


def dvol_close(day_ms: int) -> float:
    """TEST-ONLY DVOL close in vol points."""
    return 60 + 8 * math.sin(day_ms // DAY_MS * 0.21)


def hl_client_factory(capture=None):
    """Hyperliquid `POST /info` candleSnapshot: rv_close bars, and 1-minute clock candles in the
    request window up to the one in progress at SERVER_TIME, the Deribit mock's clock."""

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        assert body["type"] == "candleSnapshot"
        req = body["req"]
        if req["interval"] == "1m":
            now = int(SERVER_TIME.timestamp()) // 60 * 60_000
            ticks = range(-(-req["startTime"] // 60_000) * 60_000, min(req["endTime"], now) + 1,
                          60_000)  # fmt: skip
            candles = [{"t": t, "s": req["coin"], "i": "1m", "c": str(syn.SPOT)} for t in ticks]
        else:
            step = {"1h": 3_600_000, "5m": 300_000}[req["interval"]]
            ticks = range(-(-req["startTime"] // step) * step, req["endTime"] + 1, step)
            candles = [{"t": t, "s": req["coin"], "i": req["interval"], "c": str(rv_close(t))}
                       for t in ticks]  # fmt: skip
        return httpx.Response(200, text=json.dumps(candles))

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x.test")
    source = HttpSource("https://x.test", max_attempts=1, backoff_seconds=0, client=http)
    return HyperliquidClient(source, capture=capture)


def client_factory(capture=None):
    def handler(request: httpx.Request):
        method = request.url.path.rsplit("/", 1)[-1]
        body = {"jsonrpc": "2.0", "result": result_for(method, request.url.params)}
        return httpx.Response(200, text=json.dumps(body | {"usIn": US_OUT - 900, "usOut": US_OUT}))

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x.test")
    source = HttpSource("https://x.test", max_attempts=1, backoff_seconds=0, client=http)
    return DeribitClient(source, capture=capture)


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    out = tmp_path_factory.mktemp("pipeline") / "deribit_20261013T165955Z"
    os.environ.setdefault("TERMINAL_GIT_SHA", "a1b2c3d")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("terminal_models.tools.record_deribit.datetime", RecorderClock)
        assert (
            record(
                out,
                ["BTC", "HYPE"],
                False,
                client_factory=client_factory,
                hl_client_factory=hl_client_factory,
            )
            == 0
        )
    return out


@pytest.fixture(scope="module")
def built(recording):
    files, notes = build_fixtures.build(recording)
    return {p.name: (p, text) for p, text in files.items()}, notes


def check_fixture(text: str) -> None:
    envelope = json.loads(text)  # permissive: the NaN variant must reach the C-3 check
    validate_envelope(envelope)
    assert_no_nan_inf(envelope)
    if envelope["status"] == "success":
        validate_payload_full(
            envelope["payload_kind"], envelope["payload_version"], envelope["payload"]
        )


def test_fixtures_are_valid_successful_runs(built):
    files, notes = built
    fixtures = contracts_dir() / "fixtures"
    for asset in ("btc", "hype"):
        path, text = files[f"bs_pricer_{asset}.json"]
        assert path == fixtures / "valid" / "composite" / path.name
        env = json.loads(text)
        assert env["status"] == "success", (env.get("error"), notes)
        assert env["as_of"] == SERVER_TIME.strftime("%Y-%m-%dT%H:%M:%SZ")  # max usOut, floored
        assert env["started_at"] == env["finished_at"] == env["as_of"]
        assert env["code"]["git_sha"] == build_fixtures.FIXTURE_SHA
        assert env["run_id"][14] == "7"  # UUIDv7
        check_fixture(text)
    assert any(n.startswith("HYPE: success") for n in notes)
    assert any("guide §4.4 HYPE coverage" in n for n in notes)


def test_invalid_variants_each_fail(built):
    files, _ = built
    invalid = {n: v for n, v in files.items() if "__" in n and n.startswith("bs_pricer_")}
    assert set(invalid) == {
        "bs_pricer_btc__nan_iv.json",
        "bs_pricer_btc__chain_row_missing_fair_usd.json",
        "bs_pricer_btc__smile_series_wrong_length.json",
    }
    for path, text in invalid.values():
        assert path.parent == contracts_dir() / "fixtures" / "invalid" / "composite"
        with pytest.raises(ValidationFailure):
            check_fixture(text)


def test_build_is_deterministic(recording, built):
    again, _ = build_fixtures.build(recording)
    assert {p.name: t for p, t in again.items()} == {n: t for n, (_, t) in built[0].items()}


def test_run_ids_differ_per_asset_and_are_stable(built):
    ids = {json.loads(built[0][f"bs_pricer_{a}.json"][1])["run_id"] for a in ("btc", "hype")}
    assert len(ids) == 2
    assert build_fixtures.frozen_run_id(SERVER_TIME, "BTC") == build_fixtures.frozen_run_id(
        SERVER_TIME, "BTC"
    )


@pytest.fixture(scope="module")
def vectors(built, tmp_path_factory):
    out = tmp_path_factory.mktemp("vectors")
    for asset in ("btc", "hype"):
        fixture = out / f"bs_pricer_{asset}.json"
        fixture.write_text(built[0][fixture.name][1])
        for name, text in gen_eval_vectors.build_for_run(fixture, asset).items():
            (out / name).write_text(text)
    return out


@pytest.mark.parametrize("asset", ["btc", "hype"])
def test_run_vectors_cover_flags_and_errors(vectors, asset):
    doc = json.loads((vectors / f"evaluate_{asset}.json").read_text())
    assert {
        "spot_moved",
        "quote_stale",
        "run_stale",
        "strike_extrapolated",
        "tenor_extrapolated_short",
        "tenor_extrapolated_long",
        "expiry_time_differs_from_venue",
    } <= set(doc["flags"])
    assert {
        "expiry_in_past",
        "tenor_out_of_range",
        "strike_out_of_range",
        "premium_requires_side",
        "pct_spot_requires_spot_ref",
        "invalid_premium",
    } <= set(doc["error_codes"])
    priced = [c for c in doc["cases"] if "data" in c["expected"]]
    assert len(priced) >= 8
    assert json.loads((vectors / f"interpolation_{asset}.json").read_text())["cases"]


def test_vectors_refuse_failed_run(tmp_path, built):
    env = json.loads(built[0]["bs_pricer_btc.json"][1])
    env |= {"status": "failed", "payload": None}
    path = tmp_path / "bs_pricer_btc.json"
    path.write_text(json.dumps(env))
    with pytest.raises(SystemExit, match="vectors need a successful run"):
        gen_eval_vectors.build_for_run(path, "btc")


API = Path(__file__).resolve().parents[2] / "api"


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo not installed")
def test_rust_port_matches_recorded_run_vectors(vectors):
    """The Rust parity suite on vectors shaped like a recorded run (P-3), before real data."""
    pricing = contracts_dir() / "fixtures" / "pricing"
    for name in ("black76.json", "implied_vol.json"):
        shutil.copy(pricing / name, vectors / name)
    result = subprocess.run(
        ["cargo", "test", "--quiet", "--test", "pricing_parity"],
        cwd=API,
        env=os.environ | {"PRICING_FIXTURES_DIR": str(vectors)},
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


def test_runner_replay_reproduces_built_fixture(recording, built, tmp_path, monkeypatch):
    """A built fixture replays identically through `runner replay --envelope` (M-9)."""
    monkeypatch.setenv("SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    fixture = tmp_path / "bs_pricer_btc.json"
    fixture.write_text(built[0][fixture.name][1])
    inputs, _ = build_fixtures.replay_inputs(recording, "BTC")
    assert SnapshotStore().save(inputs) == json.loads(fixture.read_text())["inputs"]["hash"]
    assert runner(["replay", "--envelope", str(fixture)]) == 0


# ---- parity_deribit leg: recorded follow-up order books → parity fixtures ------------------

TEST_FEES = {  # TEST-ONLY arithmetic, never a schedule; keeps these tests off the shipped rates
    "schedule_version": "TEST-ONLY",
    "option_taker": 0.0003, "option_maker": 0.0002, "option_cap_pct_of_price": 0.125,
    "future_taker": 0.0005, "future_maker": 0.0001, "perp_taker": 0.0005, "perp_maker": 0.0001,
    "option_delivery": 0.00015, "future_delivery": 0.00025,
}  # fmt: skip


def with_test_fees(path):
    from terminal_models.core.manifest import load_manifest

    manifest = load_manifest(path)
    if manifest.implementation_id != "parity_deribit":
        return manifest
    return manifest.model_copy(update={"params": manifest.params | {"fees": TEST_FEES}})


@pytest.fixture(scope="module")
def parity_built(tmp_path_factory):
    from terminal_models.tools import record_deribit

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TERMINAL_GIT_SHA", "a1b2c3d")
        mp.setattr(record_deribit, "load_manifest", with_test_fees)
        mp.setattr(record_deribit, "datetime", RecorderClock)
        mp.setattr(build_fixtures, "load_manifest", with_test_fees)
        out = tmp_path_factory.mktemp("parity") / "deribit_20261013T165955Z"
        assert (
            record(
                out,
                ["BTC", "HYPE"],
                False,
                client_factory=client_factory,
                hl_client_factory=hl_client_factory,
            )
            == 0
        )
        files, notes = build_fixtures.build(out)
        again, _ = build_fixtures.build(out)
    manifest = json.loads((out / "manifest.json").read_text())
    return manifest, {p.name: (p, t) for p, t in files.items()}, notes, again


def test_recorder_records_parity_followup_order_books(parity_built):
    manifest, *_ = parity_built
    assert manifest["notes"] == []
    methods = [c["method"] for c in manifest["calls"]]
    assert {"ticker", "get_order_book"} <= set(methods)
    inputs = manifest["assets"]["BTC"]["inputs"]
    assert any(k.startswith("parity_deribit.order_book:") for k in inputs)


def test_shipped_fees_record_and_build_parity(recording, built):
    manifest = json.loads((recording / "manifest.json").read_text())
    assert manifest["notes"] == []
    files, _ = built
    assert {"parity_btc.json", "parity_hype.json"} <= set(files)


def with_null_fees(path):
    from terminal_models.core.manifest import load_manifest

    manifest = load_manifest(path)
    if manifest.implementation_id != "parity_deribit":
        return manifest
    nulled = {k: None for k in TEST_FEES} | {"schedule_version": "TBD"}
    return manifest.model_copy(update={"params": manifest.params | {"fees": nulled}})


def test_null_fees_skip_parity_and_say_so(tmp_path):
    from terminal_models.tools import record_deribit

    out = tmp_path / "deribit_20261013T165955Z"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TERMINAL_GIT_SHA", "a1b2c3d")
        mp.setattr(record_deribit, "load_manifest", with_null_fees)
        mp.setattr(build_fixtures, "load_manifest", with_null_fees)
        assert (
            record(
                out,
                ["BTC"],
                False,
                client_factory=client_factory,
                hl_client_factory=hl_client_factory,
            )
            == 0
        )
        files, notes = build_fixtures.build(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["notes"] == [
        "parity_deribit: follow-up inputs not recorded (params.fees null: future_delivery, "
        "future_maker, future_taker, option_cap_pct_of_price, option_delivery, option_maker, "
        "option_taker, perp_maker, perp_taker)"
    ]
    assert not any(p.name.startswith("parity_") for p in files)
    assert "parity_deribit: skipped, params.fees is null" in notes


def test_parity_fixtures_from_a_recording(parity_built):
    _, files, notes, again = parity_built
    for asset in ("btc", "hype"):
        path, text = files[f"parity_{asset}.json"]
        assert path.parent == contracts_dir() / "fixtures" / "valid" / "composite"
        env = json.loads(text)
        assert env["status"] == "success", (env.get("error"), notes)
        check_fixture(text)
    btc = json.loads(files["parity_btc.json"][1])
    from terminal_models.core.payloads import get_section, records

    opps = records(get_section(btc["payload"], "opportunities"))
    assert any(o["confirmed"] for o in opps), notes  # replayed order books confirmed rows
    hedges = {r["hedge_type"] for r in records(get_section(btc["payload"], "expiries"))}
    assert hedges == {"locked_future", "perp_funding"}
    invalid = {n: t for n, (_, t) in files.items() if n.startswith("parity_btc__")}
    assert set(invalid) == {
        "parity_btc__nan_band.json",
        "parity_btc__expiry_row_missing_f_impl.json",
        "parity_btc__carry_series_wrong_length.json",
    }
    for text in invalid.values():
        with pytest.raises(ValidationFailure):
            check_fixture(text)
    assert {p.name: t for p, t in again.items()} == {n: t for n, (_, t) in files.items()}


# ---- rv_har leg: paged RV candles, DVOL history and Hyperliquid → vrp fixtures --------------


def test_recorder_records_rv_inputs(recording):
    manifest = json.loads((recording / "manifest.json").read_text())
    assert "rv_har" in manifest["implementations"]
    methods = [c["method"] for c in manifest["calls"]]
    assert "candleSnapshot" in methods  # HYPE's RV comes from Hyperliquid
    inputs = manifest["assets"]["BTC"]["inputs"]
    assert {"rv_har.rv_daily", "rv_har.dvol_history"} <= set(inputs)
    assert "rv_har.rv_daily" in manifest["assets"]["HYPE"]["inputs"]


def test_vrp_fixtures_from_a_recording(built):
    from terminal_models.core.payloads import get_section, records

    files, notes = built
    for asset in ("btc", "hype"):
        path, text = files[f"vrp_{asset}.json"]
        assert path.parent == contracts_dir() / "fixtures" / "valid" / "composite"
        env = json.loads(text)
        assert env["status"] == "success", (env.get("error"), notes)
        assert env["as_of"] == SERVER_TIME.strftime("%Y-%m-%dT%H:%M:%SZ")
        check_fixture(text)
    btc = records(get_section(json.loads(files["vrp_btc.json"][1])["payload"], "signal"))
    assert [r["vrp_z"] is not None for r in btc] == [False, False, True, False]
    hype = records(get_section(json.loads(files["vrp_hype.json"][1])["payload"], "signal"))
    assert all(r["vrp_z"] is None and "z_unavailable" in r["flags"] for r in hype)
    invalid = {n: t for n, (_, t) in files.items() if n.startswith("vrp_btc__")}
    assert set(invalid) == {
        "vrp_btc__nan_rv_forecast.json",
        "vrp_btc__signal_row_missing_vrp_z.json",
        "vrp_btc__cone_series_wrong_length.json",
    }
    for text in invalid.values():
        with pytest.raises(ValidationFailure):
            check_fixture(text)


def test_validate_rv_on_a_recording(recording, tmp_path):
    """tools/validate_rv.py replays rv_daily and DVOL from the recording (§9). The series are the
    TEST-ONLY mock paths, so only the plumbing is checked, not the gate outcomes."""
    from terminal_models.tools import validate_rv

    out = tmp_path / "rv_validation.json"
    args = ["--recording", str(recording), "--out", str(out), "--folds", "4"]
    assert validate_rv.main(args) in (0, 1)
    report = json.loads(out.read_text())
    assert report["git_sha"] == os.environ["TERMINAL_GIT_SHA"]
    assert report["recording"]["name"] == recording.name
    assert report["params_hash"].startswith("sha256:")
    assert set(report["assets"]) == {"BTC", "HYPE"}
    btc, hype = report["assets"]["BTC"], report["assets"]["HYPE"]
    assert btc["status"] == hype["status"] == "ok", (btc.get("error"), hype.get("error"))
    assert btc["inputs"]["rv_daily"]["hash"].startswith("sha256:")
    assert btc["inputs"]["dvol_history"]["n_days"] > 0
    assert hype["inputs"]["dvol_history"] is None
    assert hype["inputs"]["rv_daily"]["venue"] == "hyperliquid"
    assert btc["gates"]["pnl_sharpe_vs_naive"]["status"] != "not_applicable"
    assert hype["gates"]["pnl_sharpe_vs_naive"]["status"] == "not_applicable"
    assert len(btc["backtest"]["folds"]) == 4
    again = tmp_path / "again.json"
    validate_rv.main(["--recording", str(recording), "--out", str(again), "--folds", "4"])
    assert again.read_text() == out.read_text()
