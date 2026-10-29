"""Golden tests on a recorded snapshot (RV_FORECAST_IMPLEMENTATION_GUIDE.md §11 P3, M-24).

Recordings are shared with bs_pricer_black76 and parity_deribit (one recorder run covers every
model) and live under library/bs_pricer_black76/tests/fixtures/deribit_<UTC stamp>/. The latest
one that includes rv_har is replayed through the production adapters (Deribit and Hyperliquid),
the runner's fetch path and the model; the result must equal
contracts/fixtures/valid/composite/vrp_<asset>.json.

Skips while no such recording is committed: there is no synthetic stand-in for real market data
(D-10).
"""

import json
from pathlib import Path

import pytest

from terminal_models.core.paths import contracts_dir, library_dir
from terminal_models.tools.build_fixtures import run_asset

IMPLEMENTATION = "rv_har"
RECORDINGS_DIR = library_dir() / "bs_pricer_black76" / "tests" / "fixtures"
RECORDINGS = sorted(
    p
    for p in RECORDINGS_DIR.glob("deribit_*")
    if (p / "manifest.json").exists()
    and IMPLEMENTATION in json.loads((p / "manifest.json").read_text())["implementations"]
)
RECORDING = RECORDINGS[-1] if RECORDINGS else None
ASSETS = (
    [
        a
        for a, s in json.loads((RECORDING / "manifest.json").read_text())["assets"].items()
        if s["ok"]
    ]
    if RECORDING
    else []
)

pytestmark = pytest.mark.skipif(
    RECORDING is None,
    reason="no recording with rv_har committed under bs_pricer_black76/tests/fixtures/deribit_*/",
)


def fixture_path(asset: str) -> Path:
    path = contracts_dir() / "fixtures" / "valid" / "composite" / f"vrp_{asset.lower()}.json"
    if not path.exists():
        pytest.fail(f"{path.name} missing: run tools.build_fixtures --recording {RECORDING}")
    return path


@pytest.mark.parametrize("asset", ASSETS)
def test_replay_equals_committed_fixture(asset):
    assert run_asset(RECORDING, asset, IMPLEMENTATION) == json.loads(
        fixture_path(asset).read_text()
    )


@pytest.mark.parametrize("asset", ASSETS)
def test_deterministic(asset):
    first = json.dumps(run_asset(RECORDING, asset, IMPLEMENTATION), sort_keys=True)
    for _ in range(9):
        assert json.dumps(run_asset(RECORDING, asset, IMPLEMENTATION), sort_keys=True) == first
