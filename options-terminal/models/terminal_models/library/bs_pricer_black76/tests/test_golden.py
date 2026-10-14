"""Golden tests on a recorded Deribit snapshot (guide §4.4, §5.1, M-24).

The latest `fixtures/deribit_<UTC stamp>/` recording (tools/record_deribit.py) is replayed through
the production adapter and the model; the result must equal the committed contract fixture built
from it (tools/build_fixtures.py). Skips while no recording is committed: there is no synthetic
stand-in for real market data (D-10).
"""

import json
from pathlib import Path

import pytest

from terminal_models.core.paths import contracts_dir
from terminal_models.tools.build_fixtures import run_asset

FIXTURES = Path(__file__).parent / "fixtures"
RECORDINGS = sorted(p for p in FIXTURES.glob("deribit_*") if (p / "manifest.json").exists())
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
    reason="no Deribit recording committed under tests/fixtures/deribit_*/ "
    "(run terminal_models.tools.record_deribit where Deribit is reachable)",
)


def committed(asset: str) -> dict:
    path = contracts_dir() / "fixtures" / "valid" / "composite" / f"bs_pricer_{asset.lower()}.json"
    if not path.exists():
        pytest.fail(f"{path.name} missing: run tools.build_fixtures --recording {RECORDING}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("asset", ASSETS)
def test_replay_equals_committed_fixture(asset):
    assert run_asset(RECORDING, asset) == committed(asset)


@pytest.mark.parametrize("asset", ASSETS)
def test_deterministic(asset):
    first = json.dumps(run_asset(RECORDING, asset), sort_keys=True)
    for _ in range(9):
        assert json.dumps(run_asset(RECORDING, asset), sort_keys=True) == first


@pytest.mark.parametrize("asset", [a for a in ("BTC", "ETH") if a in ASSETS])
def test_mark_reproduction_on_liquid_books(asset):
    """r = 0 Black-76 at the venue mark IV reproduces the venue mark on BTC/ETH (guide §9.6)."""
    env = committed(asset)
    assert env["status"] == "success", env.get("error")
    assert not [w for w in env["warnings"] if w["code"] == "mark_reproduction"]
