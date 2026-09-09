"""Realized-universe recording (vgp/data/universe.py).

`UNIVERSE_30` is an intention. Two stages narrow it before any strategy is
evaluated — fetch failures, then FeatureEngine's min_obs_fraction filter — and
neither is visible in results.csv unless recorded. It matters because results
computed on 28 assets are not comparable with results computed on 12, and a
universe that varies between runs is a survivorship-bias channel that neither
DSR nor the null control can detect: every trial and every surrogate inherits
the same universe, so nothing in the statistics reveals that it moved.
"""
from __future__ import annotations

import pytest


class _FakeEngine:
    """Stands in for a fitted FeatureEngine (duck-typed by from_pipeline)."""

    def __init__(self, retained, dropped, n_dates=1000):
        import pandas as pd

        self.retained_assets_ = list(retained)
        self.dropped_assets_ = list(dropped)
        self.dates_ = pd.date_range("2024-01-01", periods=n_dates, freq="D")


def _report(requested, realized, failed=()):
    from vgp.data import FetchReport

    return FetchReport(
        requested=tuple(requested),
        realized=tuple(realized),
        failed=tuple(failed),
        start_date="2024-01-01",
        end_date="2025-12-31",
        interval="1d",
    )


# ---------------------------------------------------------------------------
# Construction from the pipeline
# ---------------------------------------------------------------------------

def test_from_pipeline_captures_both_narrowing_stages():
    """Fetch losses and feature-engine drops are distinct and both recorded."""
    from vgp.data import UniverseRecord

    report = _report(
        requested=["BTCUSDT", "ETHUSDT", "SOLUSDT", "NOPEUSDT"],
        realized=["BTC", "ETH", "SOL"],
        failed=[("NOPEUSDT", "no rows returned for the requested range")],
    )
    engine = _FakeEngine(retained=["BTC", "ETH"], dropped=["SOL"])

    rec = UniverseRecord.from_pipeline(report, engine)

    assert rec.requested == ("BTC", "ETH", "SOL", "NOPE"), "symbols must map to tickers"
    assert rec.fetched == ("BTC", "ETH", "SOL")
    assert dict(rec.fetch_failed) == {
        "NOPEUSDT": "no rows returned for the requested range"
    }
    assert rec.dropped_by_features == ("SOL",)
    assert rec.retained == ("BTC", "ETH")
    assert rec.n_requested == 4
    assert rec.n_retained == 2
    assert rec.n_dates == 1000
    assert not rec.is_complete


def test_complete_universe_is_flagged_complete():
    from vgp.data import UniverseRecord

    report = _report(["BTCUSDT", "ETHUSDT"], ["BTC", "ETH"])
    rec = UniverseRecord.from_pipeline(report, _FakeEngine(["BTC", "ETH"], []))

    assert rec.is_complete
    assert "complete" in rec.summary()
    assert "INCOMPLETE" not in rec.summary()


def test_incomplete_universe_says_results_are_not_comparable():
    from vgp.data import UniverseRecord

    report = _report(
        ["BTCUSDT", "ETHUSDT"], ["BTC"], failed=[("ETHUSDT", "ConnectionError: down")]
    )
    rec = UniverseRecord.from_pipeline(report, _FakeEngine(["BTC"], []))

    summary = rec.summary()
    assert "INCOMPLETE" in summary
    assert "ETHUSDT" in summary
    assert not rec.is_complete


# ---------------------------------------------------------------------------
# Fingerprint: the mechanism that makes drift detectable
# ---------------------------------------------------------------------------

def test_fingerprint_is_stable_and_order_independent():
    """Same composition, same id — regardless of the order it was listed in.

    Order independence matters because a reordering of UNIVERSE_30 is not a
    change of universe and must not read as one.
    """
    from vgp.data import UniverseRecord

    a = UniverseRecord((), (), (), (), ("BTC", "ETH", "SOL"))
    b = UniverseRecord((), (), (), (), ("SOL", "BTC", "ETH"))

    assert a.fingerprint == b.fingerprint
    assert a.fingerprint == UniverseRecord((), (), (), (), ("BTC", "ETH", "SOL")).fingerprint
    assert len(a.fingerprint) == 12


def test_fingerprint_changes_when_composition_changes():
    """A dropped asset must change the id — that is the whole point."""
    from vgp.data import UniverseRecord

    full = UniverseRecord((), (), (), (), ("BTC", "ETH", "SOL"))
    shrunk = UniverseRecord((), (), (), (), ("BTC", "ETH"))

    assert full.fingerprint != shrunk.fingerprint, (
        "a smaller universe produced the same fingerprint — drift would be invisible"
    )


def test_fingerprint_of_empty_universe_is_labelled():
    from vgp.data import UniverseRecord

    assert UniverseRecord((), (), (), (), ()).fingerprint == "empty"


def test_fingerprint_ignores_the_lost_assets():
    """Two runs that retained the same assets match, however they got there.

    One may have lost an asset to the fetch and another to min_obs_fraction;
    what the results were computed on is identical, so the ids must agree.
    """
    from vgp.data import UniverseRecord

    via_fetch = UniverseRecord(
        ("BTC", "ETH", "SOL"), ("BTC", "ETH"),
        (("SOLUSDT", "ConnectionError"),), (), ("BTC", "ETH"),
    )
    via_features = UniverseRecord(
        ("BTC", "ETH", "SOL"), ("BTC", "ETH", "SOL"), (), ("SOL",), ("BTC", "ETH"),
    )

    assert via_fetch.fingerprint == via_features.fingerprint


# ---------------------------------------------------------------------------
# Stamping and serialization
# ---------------------------------------------------------------------------

def test_stamp_rows_tags_every_row():
    from vgp.data import UniverseRecord

    rec = UniverseRecord((), (), (), (), ("BTC", "ETH"))
    rows = [{"seed": 0}, {"seed": 1}, {"seed": 2}]

    rec.stamp_rows(rows)

    for row in rows:
        assert row["n_assets"] == 2
        assert row["universe_fingerprint"] == rec.fingerprint


def test_stamp_rows_handles_empty_input():
    from vgp.data import UniverseRecord

    assert UniverseRecord((), (), (), (), ("BTC",)).stamp_rows([]) == []


def test_to_dict_is_json_serializable_and_complete():
    """universe.json must carry enough to reconstruct what ran."""
    import json

    from vgp.data import UniverseRecord

    rec = UniverseRecord.from_pipeline(
        _report(["BTCUSDT", "ETHUSDT", "NOPEUSDT"], ["BTC", "ETH"],
                failed=[("NOPEUSDT", "no rows")]),
        _FakeEngine(["BTC"], ["ETH"], n_dates=500),
    )

    payload = rec.to_dict()
    round_tripped = json.loads(json.dumps(payload))

    assert round_tripped["fingerprint"] == rec.fingerprint
    assert round_tripped["n_requested"] == 3
    assert round_tripped["n_retained"] == 1
    assert round_tripped["n_dates"] == 500
    assert round_tripped["is_complete"] is False
    assert round_tripped["retained"] == ["BTC"]
    assert round_tripped["dropped_by_features"] == ["ETH"]
    assert round_tripped["fetch_failed"] == {"NOPEUSDT": "no rows"}


def test_record_is_immutable():
    """A run's realized universe must not be editable after the fact."""
    from vgp.data import UniverseRecord

    rec = UniverseRecord((), (), (), (), ("BTC",))
    with pytest.raises(Exception):
        rec.retained = ("BTC", "ETH")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Against the real pipeline
# ---------------------------------------------------------------------------

def test_record_matches_the_real_pipeline(synthetic_ohlcv_cache, block_network):
    """End to end: the record must agree with what FeatureEngine actually used."""
    from vgp.data import BinanceFetcher, FeatureEngine, UniverseRecord

    fetcher = BinanceFetcher(cache_dir=synthetic_ohlcv_cache)
    ohlcv = fetcher.fetch_ohlcv(force_refresh=False)
    engine = FeatureEngine()
    arr = engine.fit_transform(ohlcv)

    rec = UniverseRecord.from_pipeline(fetcher.last_fetch_report_, engine)

    assert rec.n_retained == arr.shape[2], (
        f"record says {rec.n_retained} assets, feature matrix has {arr.shape[2]}"
    )
    assert rec.n_dates == arr.shape[0]
    assert not rec.fetch_failed, "the synthetic cache is complete"
    # The fixture gives two tickers short history, so the filter must have bitten
    assert rec.dropped_by_features, "expected min_obs_fraction to drop the short-history assets"
    assert not rec.is_complete, "assets were dropped, so the universe is not complete"
    assert set(rec.retained).isdisjoint(rec.dropped_by_features)
