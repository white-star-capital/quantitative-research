"""Golden tests on a recorded Deribit snapshot (PARITY_MONITOR_IMPLEMENTATION_GUIDE.md §5, M-24).

Recordings are shared with bs_pricer_black76 (one recorder run covers both models) and live
under library/bs_pricer_black76/tests/fixtures/deribit_<UTC stamp>/. The latest one is replayed
through the production adapter, the runner's fetch path (with the recorded follow-up order
books) and the model; the result must equal contracts/fixtures/valid/composite/parity_<asset>.json.

Skips while no recording is committed or the fee table is null: there is no synthetic stand-in
for real market data (D-10), and no run on placeholder fees.
"""

import json
from pathlib import Path

import pytest

from terminal_models.core.manifest import load_manifest
from terminal_models.core.paths import contracts_dir, library_dir
from terminal_models.core.registry import null_fees
from terminal_models.tools.build_fixtures import run_asset

IMPLEMENTATION = "parity_deribit"
RECORDINGS_DIR = library_dir() / "bs_pricer_black76" / "tests" / "fixtures"
RECORDINGS = sorted(p for p in RECORDINGS_DIR.glob("deribit_*") if (p / "manifest.json").exists())
RECORDING = RECORDINGS[-1] if RECORDINGS else None
FEES_NULL = bool(null_fees(load_manifest(library_dir() / IMPLEMENTATION / "manifest.yaml")))
ASSETS = (
    [
        a
        for a, s in json.loads((RECORDING / "manifest.json").read_text())["assets"].items()
        if s["ok"]
    ]
    if RECORDING
    else []
)

pytestmark = [
    pytest.mark.skipif(
        RECORDING is None,
        reason="no Deribit recording committed under bs_pricer_black76/tests/fixtures/deribit_*/",
    ),
    pytest.mark.skipif(FEES_NULL, reason="parity_deribit params.fees is null (guide §9 step 2)"),
]


def fixture_path(asset: str) -> Path:
    path = contracts_dir() / "fixtures" / "valid" / "composite" / f"parity_{asset.lower()}.json"
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
