import textwrap

import pytest

from terminal_models.core.paths import contracts_dir, library_dir, registry_path
from terminal_models.core.registry import (
    Registry,
    RegistryError,
    load_registry,
    validate_registry,
)
from terminal_models.core.slots import load_slots

MANIFEST = """\
implementation_id: {impl}
version: {version}
fills_slot: {slot}
owner: test
description: registry test double; never run
schedule: "*/15 * * * *"
timeout_seconds: 60
max_input_age_seconds: 120
"""


@pytest.fixture
def slots():
    return load_slots(contracts_dir())


@pytest.fixture
def library(tmp_path):
    def add(impl, version="1.0.0", slot="vol_surface"):
        d = tmp_path / impl
        d.mkdir()
        (d / "manifest.yaml").write_text(MANIFEST.format(impl=impl, version=version, slot=slot))

    add("vol_surface_svi")
    add("vol_surface_sabr", version="0.3.0")
    add("smile_basic", slot="smile")
    return tmp_path


def registry(**entry):
    base = {
        "production": {"implementation": "vol_surface_svi", "version": "1.0.0"},
        "scopes": [{"underlying": "BTC"}],
    }
    return Registry.model_validate({"slots": {"vol_surface": base | entry}})


def problems(reg, slots, library):
    with pytest.raises(RegistryError) as exc:
        validate_registry(reg, slots, library)
    return exc.value.problems


def test_committed_registry_is_valid(slots):
    validate_registry(load_registry(registry_path()), slots, library_dir())


def test_valid_registry_returns_manifests(slots, library):
    reg = registry(shadow=[{"implementation": "vol_surface_sabr", "version": "0.3.0"}])
    assert set(validate_registry(reg, slots, library)) == {"vol_surface_svi", "vol_surface_sabr"}


def test_unknown_slot(slots, library):
    reg = Registry.model_validate(
        {
            "slots": {
                "nope_slot": {
                    "production": {"implementation": "vol_surface_svi", "version": "1.0.0"},
                    "scopes": [{"underlying": "BTC"}],
                }
            }
        }
    )
    assert "no slot definition" in problems(reg, slots, library)[0]


def test_missing_implementation(slots, library):
    reg = registry(production={"implementation": "vol_surface_ghost", "version": "1.0.0"})
    assert "not found in library" in problems(reg, slots, library)[0]


def test_version_mismatch(slots, library):
    reg = registry(production={"implementation": "vol_surface_svi", "version": "2.0.0"})
    assert "manifest is 1.0.0" in problems(reg, slots, library)[0]


def test_implementation_fills_other_slot(slots, library):
    reg = registry(production={"implementation": "smile_basic", "version": "1.0.0"})
    assert "fills slot 'smile'" in problems(reg, slots, library)[0]


def test_scope_with_wrong_dimensions(slots, library):
    reg = registry(scopes=[{"underlying": "BTC", "expiry": "2026-12-25"}])
    assert "exactly the keys" in problems(reg, slots, library)[0]


def test_scope_value_not_allowed(slots, library):
    reg = registry(scopes=[{"underlying": "DOGE"}])
    assert "not in allowed" in problems(reg, slots, library)[0]


def test_duplicate_implementation(slots, library):
    reg = registry(shadow=[{"implementation": "vol_surface_svi", "version": "1.0.0"}])
    assert "more than once" in problems(reg, slots, library)[0]


def test_reports_every_problem(slots, library):
    reg = registry(
        production={"implementation": "vol_surface_svi", "version": "9.9.9"},
        scopes=[{"underlying": "DOGE"}],
    )
    assert len(problems(reg, slots, library)) == 2


def test_production_is_required(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        textwrap.dedent("""\
        slots:
          vol_surface:
            scopes: [{underlying: BTC}]
    """)
    )
    with pytest.raises(ValueError):
        load_registry(path)


def test_shadow_only_slot_is_allowed(slots, library):
    reg = registry(
        production=None, shadow=[{"implementation": "vol_surface_sabr", "version": "0.3.0"}]
    )
    assert set(validate_registry(reg, slots, library)) == {"vol_surface_sabr"}


def test_null_production_needs_a_shadow():
    with pytest.raises(ValueError, match="at least one shadow"):
        registry(production=None)
