"""CONTRACTS §7 schema self-check: schemas are valid, valid fixtures pass, invalid ones fail,
pricing vectors are present and reproducible."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from terminal_models.core.paths import contracts_dir
from terminal_models.core.slots import load_slots
from terminal_models.core.validation import (
    ValidationFailure,
    assert_no_nan_inf,
    validate_envelope,
    validate_payload_full,
)
from terminal_models.tools.gen_pricing_vectors import main as gen_vectors

ROOT = contracts_dir()
VALID = sorted((ROOT / "fixtures" / "valid").rglob("*.json"))
INVALID = sorted((ROOT / "fixtures" / "invalid").rglob("*.json"))


def _check_fixture(path: Path) -> None:
    # NaN is not valid JSON; parse permissively so the invalid-NaN fixture reaches the C-3 check.
    envelope = json.loads(path.read_text())
    validate_envelope(envelope)
    assert_no_nan_inf(envelope)
    if envelope["status"] == "success":
        validate_payload_full(
            envelope["payload_kind"], envelope["payload_version"], envelope["payload"]
        )


@pytest.mark.parametrize(
    "path", sorted((ROOT / "schemas").rglob("*.json")), ids=lambda p: p.parent.name
)
def test_schemas_are_valid_draft_2020_12(path):
    Draft202012Validator.check_schema(json.loads(path.read_text()))


def test_every_slot_definition_loads():
    slots = load_slots()
    assert {"vol_surface", "bs_pricer"} <= set(slots)
    for slot in slots.values():
        assert (
            ROOT / "schemas" / "payloads" / slot.payload_kind / f"v{slot.payload_version}.json"
        ).exists()


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.name)
def test_valid_fixtures_pass(path):
    _check_fixture(path)


@pytest.mark.parametrize("path", INVALID, ids=lambda p: p.name)
def test_invalid_fixtures_fail(path):
    with pytest.raises(ValidationFailure):
        _check_fixture(path)


def test_pricing_vectors_are_reproducible():
    assert gen_vectors(["--check"]) == 0


def test_pricing_vectors_cover_guide_minimums():
    b76 = json.loads((ROOT / "fixtures" / "pricing" / "black76.json").read_text())["cases"]
    iv = json.loads((ROOT / "fixtures" / "pricing" / "implied_vol.json").read_text())["cases"]
    assert len(b76) >= 500
    reasons = {c["expected"]["reason"] for c in iv}
    assert {None, "below_intrinsic", "above_max_price", "below_min_vol", "above_max_vol"} <= reasons


def test_evaluation_vectors_are_reproducible_and_cover_every_flag_and_error():
    from terminal_models.tools.gen_eval_vectors import main as gen_eval

    assert gen_eval(["--check"]) == 0
    doc = json.loads((ROOT / "fixtures" / "pricing" / "evaluate_synthetic.json").read_text())
    assert set(doc["flags"]) == {
        "strike_extrapolated",
        "tenor_extrapolated_short",
        "tenor_extrapolated_long",
        "low_quality_nodes",
        "expiry_time_differs_from_venue",
        "quote_stale",
        "spot_moved",
        "run_stale",
    }
    assert {
        "expiry_in_past",
        "tenor_out_of_range",
        "strike_out_of_range",
        "premium_requires_side",
        "pct_spot_requires_spot_ref",
        "invalid_premium",
    } <= set(doc["error_codes"])


def test_parity_vectors_are_reproducible_and_cover_guards():
    from terminal_models.tools.gen_parity_vectors import main as gen_parity

    assert gen_parity(["--check"]) == 0
    doc = json.loads((ROOT / "fixtures" / "parity" / "vectors.json").read_text())
    kinds = {(c["inputs"]["inverse"], c["inputs"]["side"], c["inputs"]["hedge_is_perp"])
             for c in doc["cases"]}  # fmt: skip
    assert len(kinds) == 8  # inverse/linear × taker/maker × future/perp
    reasons = {g["expected"][-1] for g in doc["guards"]}
    assert {"inverse_denominator_non_positive", "non_positive_input", "tenor_below_min"} <= reasons
