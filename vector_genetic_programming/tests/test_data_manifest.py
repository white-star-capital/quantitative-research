"""The committed data must stay identical to what produced results/.

Code and seeds are in git by nature; data is only trustworthy if a reader can
check it. Without this, silent drift in data/ turns "the result does not
reproduce" into an argument instead of a diagnosis.
"""

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_DIR / "data" / "MANIFEST.json"


@pytest.fixture(scope="module")
def verifier():
    path = PROJECT_DIR / "scripts" / "verify_data.py"
    spec = importlib.util.spec_from_file_location("verify_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


def test_every_committed_file_matches_its_checksum(verifier):
    """The whole point: byte-identical data, or a readable failure."""
    problems = verifier.check()
    assert problems == [], "data/ has drifted from MANIFEST.json:\n" + "\n".join(problems)


def test_the_manifest_covers_the_universe_it_claims(manifest):
    """n_committed + missing must account for every requested symbol.

    Guards the arithmetic rather than the files: a manifest that silently
    dropped symbols would still pass the checksum test above.
    """
    assert manifest["n_committed"] + len(manifest["missing"]) == manifest["n_requested"]
    assert manifest["n_committed"] == len(manifest["files"])


def test_the_absent_symbols_are_named_not_merely_counted(manifest):
    """A partial universe changes the experiment, so it is recorded explicitly."""
    missing = manifest["missing"]
    assert missing, "if the universe is complete, remove this test and the carve-out"
    assert all(isinstance(s, str) and s for s in missing)
    assert set(missing).issubset(set(manifest["universe_requested"]))
    assert manifest["missing_reason"].strip(), "an absent symbol needs a stated reason"


def test_results_provenance_is_recorded(manifest):
    """The fingerprint ties data/ to results/ — compare fingerprints, not counts."""
    assert manifest["results_fingerprint"] == "af8fe4ee067f"
    assert manifest["n_retained_in_results"] == 21
    assert manifest["n_dates_in_results"] == 701


def test_no_file_is_a_git_lfs_pointer(manifest):
    """An LFS pointer is ~130 bytes and will not parse as parquet.

    The sibling risk_premium_pca/ uses LFS; this directory deliberately does
    not, and a stray pointer here would fail far from its cause.
    """
    for name, meta in manifest["files"].items():
        assert meta["bytes"] > 1000, f"{name} is {meta['bytes']} bytes — an LFS pointer?"
        raw = (PROJECT_DIR / "data" / name).read_bytes()[:40]
        assert not raw.startswith(b"version https://git-lfs"), name
