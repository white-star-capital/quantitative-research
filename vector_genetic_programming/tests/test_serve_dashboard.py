"""The run endpoint is a command-execution surface. These pin its refusals.

A regression here is not a cosmetic bug: it is an unauthenticated port that
starts processes. The tests check the two things that stop that being dangerous
— launching is off unless asked for, and nothing from a request reaches a shell
or a path.
"""

import importlib.util
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load(monkeypatch, allow_run: str | None):
    """Import the server fresh, since ALLOW_RUN is read at import time."""
    if allow_run is None:
        monkeypatch.delenv("VGP_ALLOW_RUN", raising=False)
    else:
        monkeypatch.setenv("VGP_ALLOW_RUN", allow_run)
    spec = importlib.util.spec_from_file_location(
        "serve_dashboard", PROJECT_DIR / "scripts" / "serve_dashboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launching_is_off_unless_explicitly_enabled(monkeypatch):
    assert _load(monkeypatch, None).ALLOW_RUN is False
    assert _load(monkeypatch, "0").ALLOW_RUN is False
    assert _load(monkeypatch, "true").ALLOW_RUN is False, "only the literal '1' enables it"
    assert _load(monkeypatch, "1").ALLOW_RUN is True


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location(
        "serve_dashboard_ro", PROJECT_DIR / "scripts" / "serve_dashboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_defaults_round_trip(srv):
    out = srv.validate({})
    assert out["fee_bps"] == 10.0
    assert out["n_null_runs"] == 99
    assert out["n_seeds"] == 3
    assert all(isinstance(out[k], int) for k in ("n_null_runs", "n_seeds", "pop_size", "n_gen"))


@pytest.mark.parametrize(
    "payload, needle",
    [
        ({"fee_bps": -1}, "outside"),
        ({"fee_bps": 1e9}, "outside"),
        ({"n_null_runs": -1}, "outside"),
        ({"n_gen": 0}, "outside"),
        ({"pop_size": 10**9}, "outside"),
        ({"pop_size": "; rm -rf /"}, "not a number"),
        ({"fee_bps": "$(whoami)"}, "not a number"),
        ({"n_seeds": None}, "not a number"),
        ({"fee_bps": float("nan")}, "not finite"),
        ({"fee_bps": float("inf")}, "not finite"),
        ({"VGP_RESULTS_DIR": "/etc"}, "unknown parameter"),
        ({"__proto__": 1}, "unknown parameter"),
    ],
)
def test_every_bad_input_is_refused(srv, payload, needle):
    with pytest.raises(ValueError) as exc:
        srv.validate(payload)
    assert needle in str(exc.value)


def test_validated_values_are_numbers_only(srv):
    """Nothing a caller sends can survive as a string.

    This is what keeps the request out of argv and out of any path: the launcher
    stringifies numbers into environment variables, and a number cannot carry a
    shell metacharacter or a directory traversal.
    """
    out = srv.validate({"fee_bps": 2.5, "n_gen": 7})
    assert all(isinstance(v, (int, float)) for v in out.values())
    assert set(out) == set(srv.PARAM_SPEC)


def test_a_run_never_writes_into_the_committed_results(srv):
    """Variant runs go to runs/, so results/ stays the record of the published run."""
    assert srv.RUNS_DIR.name == "runs"
    assert srv.RUNS_DIR != PROJECT_DIR / "results"


def test_capability_is_reported_not_assumed(srv):
    """The viewing image has no vectorbt; the UI must be told, not shown a dead button."""
    ok, reason = srv.experiment_available()
    assert isinstance(ok, bool)
    assert ok or reason, "an unavailable runner must explain itself"
