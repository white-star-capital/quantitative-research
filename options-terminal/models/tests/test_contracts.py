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
