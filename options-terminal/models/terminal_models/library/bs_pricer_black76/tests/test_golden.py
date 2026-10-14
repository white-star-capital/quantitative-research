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
from terminal_models.runner.cli import main as runner
from terminal_models.runner.snapshots import SnapshotStore
from terminal_models.tools.build_fixtures import replay_inputs, run_asset
from terminal_models.tools.gen_eval_vectors import build_for_run

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


def fixture_path(asset: str) -> Path:
    path = contracts_dir() / "fixtures" / "valid" / "composite" / f"bs_pricer_{asset.lower()}.json"
    if not path.exists():
        pytest.fail(f"{path.name} missing: run tools.build_fixtures --recording {RECORDING}")
    return path


def committed(asset: str) -> dict:
    return json.loads(fixture_path(asset).read_text())


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


@pytest.mark.parametrize("asset", ASSETS)
def test_runner_replay_reproduces_fixture(asset, tmp_path, monkeypatch):
    """`runner replay --envelope` on the fixture, from the recording's inputs (M-9)."""
    monkeypatch.setenv("SNAPSHOT_DIR", str(tmp_path))
    inputs, _ = replay_inputs(RECORDING, asset)
    assert SnapshotStore().save(inputs) == committed(asset)["inputs"]["hash"]
    assert runner(["replay", "--envelope", str(fixture_path(asset))]) == 0


@pytest.mark.parametrize("asset", [a for a in ("BTC", "HYPE") if a in ASSETS])
def test_parity_vectors_are_fresh(asset):
    """contracts/fixtures/pricing/*_<asset>.json were generated from the committed fixture."""
    pricing = contracts_dir() / "fixtures" / "pricing"
    for name, text in build_for_run(fixture_path(asset), asset.lower()).items():
        path = pricing / name
        assert path.exists() and path.read_text() == text, (
            f"{name} stale: run tools.gen_eval_vectors --run {fixture_path(asset)}"
        )
