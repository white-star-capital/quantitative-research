"""Verify the committed market data is byte-identical to what produced results/.

Reproducing a result needs the code, the seed AND the data. The first two are in
git by nature; the third is only trustworthy if a reader can check it. This
compares every file in data/ against data/MANIFEST.json and fails loudly on any
drift, so "I re-ran it and got something else" can be diagnosed rather than
argued about.

    python scripts/verify_data.py        # or: make verify-data

Exit status is 0 when the data matches, 1 when it does not.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MANIFEST = DATA_DIR / "MANIFEST.json"


def check() -> list[str]:
    """Return a list of problems; empty means the data is intact."""
    problems: list[str] = []
    if not MANIFEST.exists():
        return [f"missing manifest: {MANIFEST}"]

    manifest = json.loads(MANIFEST.read_text())
    expected = manifest["files"]

    on_disk = {p.name for p in DATA_DIR.glob("*.parquet")}
    for name in sorted(set(expected) - on_disk):
        problems.append(f"MISSING   {name}")
    for name in sorted(on_disk - set(expected)):
        problems.append(f"UNTRACKED {name} — present but not in the manifest")

    for name in sorted(set(expected) & on_disk):
        path = DATA_DIR / name
        raw = path.read_bytes()
        # A Git LFS pointer is ~130 bytes of text and will not parse as parquet.
        # Catching it here gives a readable error instead of an arrow traceback.
        if raw.startswith(b"version https://git-lfs"):
            problems.append(f"LFS PTR   {name} — run `git lfs pull`, this is not real data")
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected[name]["sha256"]:
            problems.append(
                f"CHANGED   {name}\n"
                f"            expected {expected[name]['sha256'][:16]}... "
                f"({expected[name]['bytes']} bytes)\n"
                f"            found    {digest[:16]}... ({len(raw)} bytes)"
            )
    return problems


def main() -> int:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    problems = check()

    if problems:
        print("DATA DOES NOT MATCH THE COMMITTED RESULTS\n")
        for p in problems:
            print(f"  {p}")
        print(
            "\nA re-run on this data will not reproduce results/. Restore the files "
            "with `git checkout -- data/` or re-clone."
        )
        return 1

    n = manifest.get("n_committed", 0)
    print(
        f"data OK — {n} files, {manifest.get('total_bytes', 0) / 1024:.0f} KB, "
        f"all sha256 match the manifest"
    )
    print(
        f"results were computed on universe fingerprint "
        f"{manifest.get('results_fingerprint')}: {manifest.get('n_requested')} requested, "
        f"{n} fetched, {manifest.get('n_retained_in_results')} retained, "
        f"{manifest.get('n_dates_in_results')} dates"
    )
    missing = manifest.get("missing", [])
    if missing:
        print(
            f"\n{len(missing)} symbol(s) of UNIVERSE_30 are deliberately absent: "
            f"{', '.join(missing)}.\nLoaders must pass allow_partial=True with a "
            f"min_assets floor; scripts/run.py already does."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
