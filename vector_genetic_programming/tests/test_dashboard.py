"""The dashboard must reflect the run in results/, not a snapshot of one.

This is the defect that put a two-run-stale result in README.md for a month: a
hand-maintained page drifts from the data silently, and nothing fails. The page
is generated, and these tests fail when the committed copy no longer matches
what results/ would produce.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX = PROJECT_DIR / "dashboard" / "index.html"
TEMPLATE = PROJECT_DIR / "dashboard" / "template.html"


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location(
        "build_dashboard", PROJECT_DIR / "scripts" / "build_dashboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def served():
    """The data blob the committed page actually carries."""
    html = INDEX.read_text()
    m = re.search(r'<script id="vgp-data"[^>]*>(.*?)</script>', html, re.S)
    assert m, "dashboard/index.html has no data blob"
    return json.loads(m.group(1))


def test_committed_page_matches_current_results(builder):
    """The whole point. `make dashboard` after a run, or this fails."""
    assert INDEX.exists(), "dashboard/index.html missing — run `make dashboard`"
    assert INDEX.read_text() == builder.render(builder.collect()), (
        "dashboard/index.html is stale: results/ has changed since it was built. "
        "Run `make dashboard` and commit the result."
    )


def test_the_page_carries_the_numbers_results_csv_carries(served):
    """Guards against a page that renders but shows something else."""
    import csv

    rows = [
        r
        for r in csv.DictReader((PROJECT_DIR / "results" / "results.csv").open())
        if r["oos_status"] == "ok"
    ]
    assert len(served["runs"]) == len(rows)
    for page_row, csv_row in zip(served["runs"], rows):
        assert page_row["oos"] == pytest.approx(float(csv_row["oos_sharpe"]), abs=5e-4)
        assert page_row["seed"] == int(csv_row["seed"])


def test_every_null_series_is_complete(served):
    """A truncated null array would silently shift every histogram."""
    lengths = {k: len(v) for k, v in served["nulls"].items()}
    assert len(set(lengths.values())) == 1, f"ragged null series: {lengths}"
    assert all(n > 0 for n in lengths.values())


def test_unmeasured_rows_are_not_drawn_as_results(builder, tmp_path, monkeypatch):
    """A `-inf` ranking sentinel is not a Sharpe, and must never reach the page.

    Statistical invariant 4: NaN means "not measured", never "bad result".
    """
    src = (PROJECT_DIR / "results" / "results.csv").read_text().splitlines()
    header, first = src[0], src[1]
    cols = header.split(",")
    vals = first.split(",")
    vals[cols.index("oos_status")] = "below_min_trades"
    vals[cols.index("oos_sharpe")] = "nan"

    fake = tmp_path / "results"
    fake.mkdir()
    (fake / "results.csv").write_text("\n".join([header, ",".join(vals)] + src[2:]) + "\n")
    (fake / "null_control.txt").write_text(
        (PROJECT_DIR / "results" / "null_control.txt").read_text()
    )
    monkeypatch.setattr(builder, "RESULTS_DIR", fake)

    data = builder.collect()
    assert len(data["runs"]) == len(src) - 2, "the unmeasured row should be dropped"


def test_template_still_has_its_placeholder():
    """Without it the builder would emit a page with no data and no error."""
    assert "__VGP_DATA__" in TEMPLATE.read_text()
