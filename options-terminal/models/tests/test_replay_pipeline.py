"""Recording → replay → fixtures → parity vectors, end to end and offline.

Responses are Deribit-shaped bodies built by inverting `chains.normalize` over the TEST-ONLY
synthetic chain (library/bs_pricer_black76/tests/synthetic.py): BTC as an inverse coin book, HYPE
as a linear USDC book. They exercise the plumbing a real recording goes through; nothing here is
written to contracts/fixtures (D-10).
"""

import json
import os
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from terminal_models.adapters.deribit.client import DeribitClient
from terminal_models.adapters.http import HttpSource
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


FUTURES = {
    "BTC": [{"instrument_name": "BTC-PERPETUAL", "base_currency": "BTC",
             "settlement_currency": "BTC", "settlement_period": "perpetual", "is_active": True}],
    "USDC": [{"instrument_name": "HYPE_USDC-PERPETUAL", "base_currency": "HYPE",
              "settlement_currency": "USDC", "settlement_period": "perpetual", "is_active": True}],
}  # fmt: skip
INDEX = {"btc_usd": syn.SPOT, "hype_usdc": syn.SPOT, "btcdvol_usdc": 62.0}
PERPS = {"BTC-PERPETUAL", "HYPE_USDC-PERPETUAL"}


def result_for(method: str, p: httpx.QueryParams):
    if method == "get_instruments":
        if p["kind"] == "future":
            return FUTURES[p["currency"]]
        assets = [a for a, (c, _) in BOOK.items() if c == p["currency"]]
        return [m for a in assets for m in options(a)[0]]
    if method == "get_book_summary_by_currency":
        assets = [a for a, (c, _) in BOOK.items() if c == p["currency"]]
        return [s for a in assets for s in options(a)[1]]
    if method == "get_index_price":
        return {"index_price": INDEX[p["index_name"]]}
    if method == "get_tradingview_chart_data":
        assert p["instrument_name"] in PERPS
        closes = syn.closes()
        end = int(p["end_timestamp"]) // 3_600_000 * 3_600_000
        ticks = [end - 3_600_000 * (len(closes) - 1 - i) for i in range(len(closes))]
        return {"status": "ok", "ticks": ticks, "close": closes}
    if method == "get_volatility_index_data":
        return {"data": [[ms(syn.AS_OF.strftime("%Y-%m-%dT%H:%M:%SZ")), 61, 63, 60, 62]]}
    raise AssertionError(method)


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
    assert record(out, ["BTC", "HYPE"], False, client_factory=client_factory) == 0
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
    invalid = {n: v for n, v in files.items() if "__" in n}
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
