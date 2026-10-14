"""Generate evaluation parity vectors for the Rust port (API A-33, guide §4.4).

    python -m terminal_models.tools.gen_eval_vectors [--check]

Writes contracts/fixtures/pricing/{interpolation,evaluate}_synthetic.json from a frozen run of
bs_pricer_black76 on the TEST-ONLY synthetic chain (library/bs_pricer_black76/tests/synthetic.py).
They are math vectors, not market data: they pin the surface/evaluate semantics in Python so the
Rust port can be held to them today. Once Deribit snapshots are recorded, the same cases run on
the frozen BTC and HYPE runs and land as interpolation.json / evaluate.json (guide §4.4).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from terminal_models.core.manifest import load_manifest
from terminal_models.core.paths import contracts_dir, library_dir
from terminal_models.core.quant.evaluate import (
    EvaluateError,
    Request,
    evaluate,
    parse_utc,
    surface_from_payload,
    utc,
)
from terminal_models.core.slots import load_slots
from terminal_models.library.bs_pricer_black76.tests import synthetic as syn
from terminal_models.runner.execute import RunSpec, execute

RUN_ID = "0192f3a0-6b1e-7c3a-9a51-000000000001"


def frozen_run() -> dict[str, Any]:
    slot = load_slots()["bs_pricer"]
    manifest = load_manifest(library_dir() / "bs_pricer_black76" / "manifest.yaml")
    spec = RunSpec(slot, manifest, "shadow", {"underlying": "BTC"}, syn.AS_OF)
    env = execute(spec, {}, None, "a1b2c3d", syn.inputs(), isolate=False)
    if env["status"] != "success":
        raise RuntimeError(f"synthetic run failed: {env['error']}")
    # Freeze the parts that vary per execution.
    env |= {"run_id": RUN_ID, "started_at": env["as_of"], "finished_at": env["as_of"]}
    return env


def interpolation_cases(env: dict[str, Any]) -> list[dict]:
    run = surface_from_payload(env["run_id"], env["as_of"], env["scope"], env["payload"])
    surface = run.surface
    cases = []
    tenors = [e.T for e in surface.expiries]
    probe_t = tenors + [
        0.5 * (tenors[0] + tenors[1]),
        0.5 * (tenors[2] + tenors[3]),
        tenors[0] * 0.5,  # short
        tenors[-1] * 1.5,  # long
    ]
    for T in probe_t:  # noqa: N806
        forward, f_flags = surface.forward(T)
        for k in (-0.6, -0.3, -0.1, 0.0, 0.05, 0.2, 0.45):
            out: dict[str, Any] = {"forward": forward, "forward_flags": list(f_flags)}
            for attr in ("iv", "iv_bid", "iv_ask"):
                p = surface.sigma(k, T, attr)
                out[attr] = (
                    None
                    if p is None
                    else {
                        "sigma": p.sigma,
                        "flags": list(p.flags),
                        "interpolation": p.interpolation,
                        "expiries": [utc(e.expiry) for e in p.expiries],
                    }
                )
            cases.append({"inputs": {"k": k, "T": T}, "expected": out})
    return cases


def evaluate_cases(env: dict[str, Any]) -> list[dict]:
    as_of = parse_utc(env["as_of"], "as_of")
    exp31 = syn.expiry_at(31)
    later = utc(syn.expiry_at(20).replace(hour=23, minute=59))
    stale_quote = utc(as_of - timedelta(days=5))
    requests: list[tuple[dict[str, Any], bool]] = [
        ({"option_type": "call", "expiry": utc(exp31), "strike": 105.0, "size": 2.0}, False),
        ({"option_type": "put", "expiry": later, "strike": 95.0}, False),
        (
            {
                "option_type": "call",
                "expiry": utc(exp31),
                "strike": 110.0,
                "size": 10.0,
                "premium": 50.0,
                "premium_unit": "usd_total",
                "side": "sell",
                "spot_ref": 101.5,
                "quoted_at": stale_quote,
            },
            True,
        ),
        (
            {
                "option_type": "put",
                "expiry": later,
                "strike": 92.5,
                "premium": 1.1,
                "premium_unit": "pct_spot",
                "side": "buy",
                "spot_ref": 99.9,
            },
            False,
        ),
        (
            {
                "option_type": "call",
                "expiry": utc(exp31),
                "strike": 100.0,
                "premium": 0.06,
                "premium_unit": "coin_per_unit",
                "side": "sell",
            },
            False,
        ),
        (
            {
                "option_type": "call",
                "expiry": utc(exp31),
                "strike": 80.0,
                "premium": 1.0,
                "premium_unit": "usd_per_unit",
                "side": "buy",
            },
            False,
        ),
        ({"option_type": "call", "expiry": utc(exp31), "strike": 200.0}, False),
        ({"option_type": "put", "expiry": utc(as_of + timedelta(days=1)), "strike": 99.0}, False),
        ({"option_type": "call", "expiry": utc(syn.expiry_at(150)), "strike": 120.0}, False),
        (
            {
                "option_type": "call",
                "expiry": utc(exp31.replace(hour=23, minute=59)),
                "strike": 102.5,
            },
            False,
        ),
        ({"option_type": "call", "expiry": utc(syn.expiry_at(3)), "strike": 136.0}, False),
        # errors
        ({"option_type": "call", "expiry": utc(as_of), "strike": 100.0}, False),
        ({"option_type": "call", "expiry": utc(syn.expiry_at(400)), "strike": 100.0}, False),
        ({"option_type": "call", "expiry": utc(exp31), "strike": 10.0}, False),
        (
            {
                "option_type": "call",
                "expiry": utc(exp31),
                "strike": 100.0,
                "premium": 1.0,
                "premium_unit": "usd_per_unit",
            },
            False,
        ),
        (
            {
                "option_type": "call",
                "expiry": utc(exp31),
                "strike": 100.0,
                "premium": 1.0,
                "side": "sell",
            },
            False,
        ),
        (
            {
                "option_type": "call",
                "expiry": utc(exp31),
                "strike": 100.0,
                "premium": 1.0,
                "side": "sell",
                "premium_unit": "pct_spot",
            },
            False,
        ),
        (
            {
                "option_type": "call",
                "expiry": utc(exp31),
                "strike": 100.0,
                "premium": -1.0,
                "side": "sell",
                "premium_unit": "usd_per_unit",
            },
            False,
        ),
        (
            {
                "option_type": "call",
                "expiry": utc(exp31),
                "strike": 100.0,
                "premium": 1.0,
                "side": "sell",
                "premium_unit": "usd_per_unit",
                "spot_ref": -5.0,
            },
            False,
        ),
        ({"option_type": "straddle", "expiry": utc(exp31), "strike": 100.0}, False),
        ({"option_type": "call", "expiry": utc(exp31), "strike": 100.0, "size": 0.0}, False),
    ]
    return expected_cases(env, requests)


def relative_requests(env: dict[str, Any]) -> list[tuple[dict[str, Any], bool]]:
    """Cases for a recorded run, placed relative to its own expiries, forwards and nodes so every
    flag and error path is exercised at any price level. Premiums are derived from the run's own
    fair values (no market numbers are entered here)."""
    run = surface_from_payload(env["run_id"], env["as_of"], env["scope"], env["payload"])
    surf, as_of, spot = run.surface, run.as_of, run.spot
    exps = surf.expiries
    mid = exps[len(exps) // 2]
    nxt = exps[min(len(exps) - 1, len(exps) // 2 + 1)]
    f_mid = mid.forward
    node_call = min((n.strike for n in mid.nodes if n.strike >= f_mid), default=f_mid)
    between = as_of + (mid.expiry - as_of + nxt.expiry - as_of) / 2
    between = between.replace(second=0, microsecond=0)
    config = load_slots()["bs_pricer"].evaluation.model_dump()

    def fair(option_type: str, expiry, strike: float) -> float:
        req = Request(option_type=option_type, expiry=expiry, strike=strike)
        return evaluate(run, req, config, False)["fair"]["per_unit_usd"]

    def r6(x: float) -> float:
        return float(f"{x:.6g}")

    k_sell, k_put, k_coin = (
        r6(1.1 * f_mid),
        r6(0.92 * surf.forward((between - as_of).total_seconds() / (365 * 86400))[0]),
        r6(f_mid),
    )
    weak = next((n for e in exps for n in e.nodes if n.quality != "good"), None)
    weak_exp = next((e for e in exps if weak in e.nodes), None)
    reqs: list[tuple[dict[str, Any], bool]] = [
        (
            {"option_type": "call", "expiry": utc(mid.expiry), "strike": node_call, "size": 2.0},
            False,
        ),
        ({"option_type": "put", "expiry": utc(between), "strike": k_put}, False),
        (
            {
                "option_type": "call",
                "expiry": utc(mid.expiry),
                "strike": k_sell,
                "size": 10.0,
                "premium": r6(fair("call", mid.expiry, k_sell) * 10 * 1.05),
                "premium_unit": "usd_total",
                "side": "sell",
                "spot_ref": r6(spot * 1.015),
                "quoted_at": utc(as_of - timedelta(days=5)),
            },
            True,
        ),
        (
            {
                "option_type": "put",
                "expiry": utc(between),
                "strike": k_put,
                "premium": r6(fair("put", between, k_put) / (spot * 0.999) * 100 * 0.97),
                "premium_unit": "pct_spot",
                "side": "buy",
                "spot_ref": r6(spot * 0.999),
            },
            False,
        ),
        (
            {
                "option_type": "call",
                "expiry": utc(mid.expiry),
                "strike": k_coin,
                "premium": r6(fair("call", mid.expiry, k_coin) / spot * 1.02),
                "premium_unit": "coin_per_unit",
                "side": "sell",
            },
            False,
        ),
        (
            {
                "option_type": "call",
                "expiry": utc(mid.expiry),
                "strike": r6(0.8 * f_mid),
                "premium": r6(0.1 * f_mid),
                "premium_unit": "usd_per_unit",
                "side": "buy",
            },
            False,
        ),
        (
            {
                "option_type": "call",
                "expiry": utc(mid.expiry),
                "strike": r6(min(4.5, math.exp(mid.k_max + 0.3)) * f_mid),
            },
            False,
        ),
        (
            {
                "option_type": "put",
                "expiry": utc(as_of + (exps[0].expiry - as_of) / 2),
                "strike": r6(spot),
            },
            False,
        ),
        (
            {
                "option_type": "call",
                "expiry": utc(as_of + (exps[-1].expiry - as_of) * 1.5),
                "strike": r6(1.05 * exps[-1].forward),
            },
            False,
        ),
        (
            {
                "option_type": "call",
                "expiry": utc(
                    mid.expiry.replace(hour=23 if mid.expiry.hour != 23 else 0, minute=59)
                ),
                "strike": node_call,
            },
            False,
        ),
    ]
    if weak is not None and weak_exp is not None:
        reqs.append(
            ({"option_type": "call", "expiry": utc(weak_exp.expiry), "strike": weak.strike}, False)
        )
    reqs += [
        ({"option_type": "call", "expiry": utc(as_of), "strike": r6(f_mid)}, False),
        (
            {
                "option_type": "call",
                "expiry": utc(as_of + (exps[-1].expiry - as_of) * 3),
                "strike": r6(f_mid),
            },
            False,
        ),
        ({"option_type": "call", "expiry": utc(mid.expiry), "strike": r6(0.1 * f_mid)}, False),
        (
            {
                "option_type": "call",
                "expiry": utc(mid.expiry),
                "strike": r6(f_mid),
                "premium": 1.0,
                "premium_unit": "usd_per_unit",
            },
            False,
        ),
        (
            {
                "option_type": "call",
                "expiry": utc(mid.expiry),
                "strike": r6(f_mid),
                "premium": 1.0,
                "side": "sell",
                "premium_unit": "pct_spot",
            },
            False,
        ),
        (
            {
                "option_type": "call",
                "expiry": utc(mid.expiry),
                "strike": r6(f_mid),
                "premium": -1.0,
                "side": "sell",
                "premium_unit": "usd_per_unit",
            },
            False,
        ),
    ]
    return reqs


def expected_cases(env: dict[str, Any], requests: list[tuple[dict[str, Any], bool]]):
    run = surface_from_payload(env["run_id"], env["as_of"], env["scope"], env["payload"])
    config = load_slots()["bs_pricer"].evaluation.model_dump()
    cases = []
    for params, run_stale in requests:
        req = Request(
            option_type=params["option_type"],
            expiry=parse_utc(params["expiry"], "expiry"),
            strike=params["strike"],
            size=params.get("size", 1.0),
            premium=params.get("premium"),
            premium_unit=params.get("premium_unit"),
            side=params.get("side"),
            spot_ref=params.get("spot_ref"),
            quoted_at=parse_utc(params["quoted_at"], "quoted_at")
            if "quoted_at" in params
            else None,
        )
        try:
            expected: dict[str, Any] = {"data": evaluate(run, req, config, run_stale)}
        except EvaluateError as exc:
            expected = {"error": exc.code}
        cases.append({"request": params, "run_stale": run_stale, "expected": expected})
    codes = {c["expected"].get("error") for c in cases} - {None}
    flags = {f["code"] for c in cases for f in c["expected"].get("data", {}).get("flags", [])}
    return cases, codes, flags, config


def render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=1, allow_nan=False) + "\n"


def build_for_run(path: Path, name: str) -> dict[str, str]:
    """Vectors from a recorded-run fixture (guide §4.4): interpolation_<name>.json and
    evaluate_<name>.json. The run must be a successful bs_pricer envelope."""
    env = json.loads(path.read_text())
    if env.get("status") != "success":
        raise SystemExit(f"{path}: run status {env.get('status')}; vectors need a successful run")
    cases, codes, flags, config = expected_cases(env, relative_requests(env))
    header = {
        "generated_by": "python -m terminal_models.tools.gen_eval_vectors --run",
        "data": f"frozen recorded run {path.name}",
    }
    return {
        f"interpolation_{name}.json": render(
            header | {"run": env, "cases": interpolation_cases(env)}
        ),
        f"evaluate_{name}.json": render(
            header
            | {
                "run": env,
                "config": config,
                "error_codes": sorted(codes),
                "flags": sorted(flags),
                "cases": cases,
            }
        ),
    }


def build() -> dict[str, str]:
    env = frozen_run()
    cases, codes, flags, config = evaluate_cases(env)
    header = {
        "generated_by": "python -m terminal_models.tools.gen_eval_vectors",
        "data": "SYNTHETIC test chain (math vectors, not market data)",
    }
    return {
        "interpolation_synthetic.json": render(
            header | {"run": env, "cases": interpolation_cases(env)}
        ),
        "evaluate_synthetic.json": render(
            header
            | {
                "run": env,
                "config": config,
                "error_codes": sorted(codes),
                "flags": sorted(flags),
                "cases": cases,
            }
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if committed files differ")
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        default=[],
        help="recorded-run fixture (repeatable); writes *_<asset>.json vectors",
    )
    args = parser.parse_args(argv)
    out = contracts_dir() / "fixtures" / "pricing"
    stale = []
    files = build() if not args.run else {}
    for run in args.run:
        files |= build_for_run(run, run.stem.removeprefix("bs_pricer_"))
    for name, content in files.items():
        path = out / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(name)
        else:
            path.write_text(content)
            print(f"wrote {path}")
    if stale:
        print(f"stale evaluation vectors: {', '.join(stale)}; regenerate them", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
