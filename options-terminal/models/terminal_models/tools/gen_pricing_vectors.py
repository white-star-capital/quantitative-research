"""Generate contracts/fixtures/pricing/{black76,implied_vol}.json from the Python reference.

    python -m terminal_models.tools.gen_pricing_vectors [--check]

Inputs are a parameter grid (pure math, not market data). Output is deterministic: rerunning
must produce byte-identical files; `--check` fails if the committed files differ (CI).
`interpolation.json` and `evaluate.json` need frozen recorded runs and are generated separately.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from terminal_models.core.paths import contracts_dir
from terminal_models.core.quant.black76 import black76, implied_vol

F = 100.0
S = 99.8  # a small basis, so spot- and forward-based greeks differ
MONEYNESS = [0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.2, 1.5, 2.5]  # K / F
TENORS = [1 / 8760, 1 / 365, 7 / 365, 30 / 365, 0.5, 2.0]  # 1 h .. 2 y
VOLS = [0.05, 0.2, 0.55, 1.0, 1.8, 3.0]
RATES = [0.0, 0.05]


def black76_cases() -> list[dict]:
    cases = []
    grid = itertools.product(MONEYNESS, TENORS, VOLS, RATES, [True, False])
    for m, t, v, r, call in grid:
        inputs = {"F": F, "K": F * m, "T": t, "sigma": v, "r": r, "is_call": call, "S": S}
        cases.append({"inputs": inputs, "expected": black76(**inputs)})
    # Limits: expired and zero-vol return discounted intrinsic only.
    for t, v, call in [(0.0, 0.5, True), (0.0, 0.5, False), (0.25, 0.0, True), (0.25, 0.0, False)]:
        inputs = {"F": F, "K": 90.0, "T": t, "sigma": v, "r": 0.05, "is_call": call, "S": S}
        cases.append({"inputs": inputs, "expected": black76(**inputs)})
    return cases


def implied_vol_cases() -> list[dict]:
    cases = []
    for m, t, v, r, call in itertools.product(
        [0.7, 0.95, 1.0, 1.05, 1.5],
        [1 / 365, 30 / 365, 1.0],
        [0.05, 0.55, 2.0],
        RATES,
        [True, False],
    ):
        k, price = F * m, None
        price = black76(F, k, t, v, r, call, S)["price"]
        iv, reason = implied_vol(F, k, t, price, r, call, S)
        cases.append(
            {
                "inputs": {"F": F, "K": k, "T": t, "price": price, "r": r, "is_call": call, "S": S},
                "source_sigma": v,
                "expected": {"iv": iv, "reason": reason},
            }
        )
    t = 30 / 365
    errors = [
        ("below_intrinsic", 80.0, True, 20.0),  # deep ITM call priced at intrinsic
        ("above_max_price", 120.0, True, 100.0),  # call priced at the forward
        ("below_min_vol", 100.0, True, black76(F, 100.0, t, 0.005, 0.0, True, S)["price"]),
        ("above_max_vol", 100.0, True, black76(F, 100.0, t, 6.0, 0.0, True, S)["price"]),
        ("below_intrinsic", 120.0, False, 20.0),
        ("above_max_price", 80.0, False, 80.0),
    ]
    for expected_reason, k, call, price in errors:
        iv, reason = implied_vol(F, k, t, price, 0.0, call, S)
        assert reason == expected_reason, (expected_reason, reason)
        cases.append(
            {
                "inputs": {
                    "F": F,
                    "K": k,
                    "T": t,
                    "price": price,
                    "r": 0.0,
                    "is_call": call,
                    "S": S,
                },
                "expected": {"iv": iv, "reason": reason},
            }
        )
    return cases


def render(cases: list[dict], description: str) -> str:
    doc = {
        "generated_by": "python -m terminal_models.tools.gen_pricing_vectors",
        "description": description,
        "cases": cases,
    }
    return json.dumps(doc, indent=1, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if committed files differ")
    args = parser.parse_args(argv)
    out = contracts_dir() / "fixtures" / "pricing"
    files = {
        "black76.json": render(black76_cases(), "black76(F, K, T, sigma, r, is_call, S) outputs"),
        "implied_vol.json": render(
            implied_vol_cases(), "implied_vol round-trips (solver output) and every error reason"
        ),
    }
    stale = []
    for name, content in files.items():
        path: Path = out / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(name)
        else:
            out.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            print(f"wrote {path} ({len(json.loads(content)['cases'])} cases)")
    if stale:
        print(f"stale pricing vectors: {', '.join(stale)}; regenerate them", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
