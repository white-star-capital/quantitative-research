"""Render dashboard/index.html from whatever is currently in results/.

The dashboard is generated, never hand-edited. Every number on the page is read
out of results/results.csv and results/null_control.txt at build time, so the
page cannot drift from the run it claims to show — the failure mode that put a
two-run-stale result in README.md for a month.

    python scripts/build_dashboard.py            # -> dashboard/index.html
    python scripts/build_dashboard.py --check    # verify only, write nothing

`--check` exits non-zero when the committed page does not match what the current
results/ would produce. CI uses it so a new run cannot be merged with a stale
dashboard beside it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
DASH_DIR = PROJECT_DIR / "dashboard"
TEMPLATE = DASH_DIR / "template.html"
OUTPUT = DASH_DIR / "index.html"

# The four statistics the null control reports, as they are labelled in
# null_control.txt. Order is the order they appear on the page.
NULL_SERIES = {
    "max_is": "best IS Sharpe",
    "max_oos": "best OOS Sharpe",
    "typical_is": "typical IS Sharpe",
    "typical_oos": "typical OOS Sharpe",
}


class BuildError(RuntimeError):
    """Something in results/ is missing or unparseable — never guess past it."""


def _null_runs(text: str) -> dict[str, list[float]]:
    """The per-run Sharpe arrays, one per statistic."""
    out: dict[str, list[float]] = {}
    for key, label in NULL_SERIES.items():
        m = re.search(rf"null {re.escape(label)} per run:\s*(.+)", text)
        if not m:
            raise BuildError(f"null_control.txt has no '{label} per run' line")
        out[key] = [float(x) for x in m.group(1).split(",")]
    lengths = {len(v) for v in out.values()}
    if len(lengths) != 1:
        raise BuildError(f"null series have different lengths: {lengths}")
    return out


def _summary_row(text: str, label: str) -> tuple[float, float, float]:
    """(observed, null median, p) from a summary line of null_control.txt."""
    m = re.search(
        rf"{re.escape(label)}\s+([+-][\d.]+)\s+null median\s+([+-][\d.]+)"
        rf"\s+null p95\s+[+-][\d.]+\s+p = ([\d.]+)",
        text,
    )
    if not m:
        raise BuildError(f"null_control.txt has no summary line for '{label}'")
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def collect() -> dict:
    """Everything the page needs, read from results/."""
    for path in (RESULTS_DIR / "results.csv", RESULTS_DIR / "null_control.txt", TEMPLATE):
        if not path.exists():
            raise BuildError(f"missing input: {path}")

    text = (RESULTS_DIR / "null_control.txt").read_text()
    rows = list(csv.DictReader((RESULTS_DIR / "results.csv").open()))
    if not rows:
        raise BuildError("results.csv has no rows")

    observed, nullmed, pvals = {}, {}, {}
    for key, label in NULL_SERIES.items():
        observed[key], nullmed[key], pvals[key] = _summary_row(text, label)

    runs = [
        {
            "w": int(r["window_id"]),
            "seed": int(r["seed"]),
            "start": r["test_start"],
            "end": r["test_end"],
            "is_": round(float(r["is_sharpe"]), 3),
            "val": round(float(r["val_sharpe"]), 3),
            "oos": round(float(r["oos_sharpe"]), 3),
            "trades": int(r["oos_n_trades"]),
            "nodes": int(r["n_nodes_best"]),
            "front": int(r["n_pareto_front"]),
        }
        for r in rows
        # A row whose OOS was never measured carries NaN plus a status; it is not
        # a result and must not be drawn as one.
        if r.get("oos_status") == "ok"
    ]
    if not runs:
        raise BuildError("no rows with oos_status == 'ok' — nothing measured to show")

    return {
        "nulls": _null_runs(text),
        "runs": runs,
        "observed": observed,
        "nullmed": nullmed,
        "p": pvals,
        "n_evaluations": sum(int(r["n_evaluations"]) for r in rows),
        "dsr_n_trials": int(rows[0]["dsr_n_trials"]),
        "fingerprint": rows[0]["universe_fingerprint"],
        "n_assets": int(rows[0]["n_assets"]),
    }


def render(data: dict) -> str:
    template = TEMPLATE.read_text()
    if "__VGP_DATA__" not in template:
        raise BuildError(f"{TEMPLATE} has no __VGP_DATA__ placeholder")
    blob = json.dumps(data, separators=(",", ":"))
    # The blob lands inside <script type="application/json">, so the only
    # sequence that can break out of it is a literal "</script>".
    if "</script" in blob.lower():
        raise BuildError("data contains a script-closing sequence")
    return template.replace("__VGP_DATA__", blob)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify dashboard/index.html is current; write nothing",
    )
    args = ap.parse_args()

    try:
        page = render(collect())
    except BuildError as exc:
        print(f"build_dashboard: {exc}", file=sys.stderr)
        return 2

    if args.check:
        if not OUTPUT.exists():
            print(
                f"build_dashboard: {OUTPUT} does not exist — run without --check",
                file=sys.stderr,
            )
            return 1
        if OUTPUT.read_text() != page:
            print(
                "build_dashboard: dashboard/index.html is STALE — results/ has changed "
                "since it was built. Run `make dashboard` and commit the result.",
                file=sys.stderr,
            )
            return 1
        print("dashboard/index.html is current")
        return 0

    DASH_DIR.mkdir(exist_ok=True)
    OUTPUT.write_text(page)
    print(f"wrote {OUTPUT.relative_to(PROJECT_DIR)} ({len(page) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
