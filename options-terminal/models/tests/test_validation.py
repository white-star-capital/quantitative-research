import math

import pytest

from terminal_models.core.validation import (
    SANITY_CHECKS,
    ValidationFailure,
    assert_no_nan_inf,
    check_payload_size,
    validate_envelope,
)

# Structural test values only: no market data. payload is an empty object because payload
# schemas land with M0; this exercises the envelope schema alone.
BASE = {
    "envelope_version": "1",
    "run_id": "0192f3a0-6b1e-7c3a-9a51-2f8e4d1c7b20",
    "slot_id": "vol_surface",
    "implementation_id": "vol_surface_svi",
    "implementation_version": "2.1.0",
    "scope": {"underlying": "BTC"},
    "status": "success",
    "as_of": "2026-10-13T17:00:00Z",
    "started_at": "2026-10-13T17:00:02Z",
    "finished_at": "2026-10-13T17:00:09Z",
    "inputs": {
        "sources": [{"name": "options_chain", "snapshot_at": "2026-10-13T17:00:00Z"}],
        "hash": "sha256:" + "0" * 64,
    },
    "code": {"git_sha": "a1b2c3d"},
    "warnings": [],
    "payload_kind": "surface",
    "payload_version": "1",
    "payload": {},
}


def failed(**overrides):
    return (
        BASE
        | {
            "status": "failed",
            "payload": None,
            "error": {"code": "stale_input", "message": "input too old"},
        }
        | overrides
    )


def test_success_envelope_valid():
    validate_envelope(BASE)


def test_failed_envelope_valid():
    validate_envelope(failed())


@pytest.mark.parametrize(
    "envelope",
    [
        pytest.param(failed(payload={}), id="failed-with-payload"),
        pytest.param({k: v for k, v in failed().items() if k != "error"}, id="failed-no-error"),
        pytest.param(BASE | {"payload": None}, id="success-null-payload"),
        pytest.param(BASE | {"error": {"code": "x", "message": "y"}}, id="success-with-error"),
        pytest.param(BASE | {"as_of": "2026-10-13T17:00:00+02:00"}, id="non-utc-timestamp"),
        pytest.param(BASE | {"slot_id": "Vol-Surface"}, id="bad-slot-id"),
        pytest.param(BASE | {"payload_kind": "heatmap"}, id="unknown-kind"),
        pytest.param(BASE | {"extra": 1}, id="unknown-field"),
    ],
)
def test_invalid_envelopes_rejected(envelope):
    with pytest.raises(ValidationFailure) as exc:
        validate_envelope(envelope)
    assert exc.value.code == "invalid_envelope"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_rejected_anywhere(bad):
    with pytest.raises(ValidationFailure) as exc:
        assert_no_nan_inf({"z": {"values": [[None, bad]]}})
    assert exc.value.code == "non_finite"
    assert "/z/values/0/1" in exc.value.message


def test_null_is_allowed():
    assert_no_nan_inf({"z": {"values": [[None, None]]}})


def test_payload_size_limits():
    assert check_payload_size({"a": "x"}) == []
    assert check_payload_size({"a": "x" * (3 * 1024 * 1024)})[0]["code"] == "payload_large"
    with pytest.raises(ValidationFailure):
        check_payload_size({"a": "x" * (6 * 1024 * 1024)})


def test_sanity_checks_registered():
    assert set(SANITY_CHECKS) == {
        "calendar_arbitrage",
        "butterfly_arbitrage",
        "put_call_parity",
        "mark_reproduction",
        "spot_vs_perp",
        "monotonic_tenor",
        "greeks_signs",
        "parity_band_ordering",
        "implied_vs_venue_forward",
    }


def _parity_payload(bands, expiries):
    from terminal_models.core.payloads import column, section, unit

    usd = unit("price_usd")
    band_cols = [column("expiry_ts", "Expiry", "timestamp"), column("strike", "K", "number", usd),
                 column("f_impl_bid", "Bid", "number", usd),
                 column("f_impl_ask", "Ask", "number", usd)]  # fmt: skip
    exp_cols = [column("expiry_ts", "Expiry", "timestamp"), column("f_impl", "F", "number", usd),
                column("venue_forward", "Venue", "number", usd)]  # fmt: skip
    return {
        "sections": [
            section("bands", "Bands", "table", {"columns": band_cols, "rows": bands}),
            section("expiries", "Expiries", "table", {"columns": exp_cols, "rows": expiries}),
        ]
    }


def test_parity_band_ordering_fails_on_a_crossed_band():
    ok = _parity_payload([["2026-11-27T08:00:00Z", 100.0, 99.0, 101.0]], [])
    assert SANITY_CHECKS["parity_band_ordering"](ok, {}) == []
    crossed = _parity_payload([["2026-11-27T08:00:00Z", 100.0, 102.0, 101.0]], [])
    with pytest.raises(ValidationFailure, match="1 crossed parity bands"):
        SANITY_CHECKS["parity_band_ordering"](crossed, {})


def test_implied_vs_venue_forward_warns_beyond_one_percent():
    rows = [["2026-11-27T08:00:00Z", 100.5, 100.0], ["2026-12-25T08:00:00Z", 103.0, 100.0],
            ["2027-03-26T08:00:00Z", None, 100.0]]  # fmt: skip
    warnings = SANITY_CHECKS["implied_vs_venue_forward"](_parity_payload([], rows), {})
    assert [w["code"] for w in warnings] == ["implied_vs_venue_forward"]
    assert "2026-12-25T08:00:00Z 3.00%" in warnings[0]["message"]
