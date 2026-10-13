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
import sys
from datetime import timedelta
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
    run = surface_from_payload(env["run_id"], env["as_of"], env["scope"], env["payload"])
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
    args = parser.parse_args(argv)
    out = contracts_dir() / "fixtures" / "pricing"
    stale = []
    for name, content in build().items():
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
