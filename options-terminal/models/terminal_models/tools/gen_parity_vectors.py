"""Generate contracts/fixtures/parity/vectors.json from core/quant/parity.py.

    python -m terminal_models.tools.gen_parity_vectors [--check]

Pure math on a seeded grid (not market data), consumed by the frontend tests: the Parity tab
recomputes maker/taker fees, net edges and inventory pickup client-side and must agree with
the model. The fee table here is TEST-ONLY and labelled as such in the file.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict

from terminal_models.core.paths import contracts_dir
from terminal_models.core.quant import parity as P

TEST_FEES = {  # arithmetic only; never a real schedule
    "schedule_version": "TEST-ONLY",
    "option_taker": 0.0003, "option_maker": 0.0002, "option_cap_pct_of_price": 0.125,
    "future_taker": 0.0005, "future_maker": 0.0001, "perp_taker": 0.0005, "perp_maker": 0.0001,
    "option_delivery": 0.00015, "future_delivery": 0.00025,
}  # fmt: skip


def r6(x: float) -> float:
    return float(f"{x:.6g}")


def quotes(rng: random.Random, inverse: bool) -> dict:
    F = r6(rng.uniform(40_000, 120_000) if inverse else rng.uniform(5, 400))  # noqa: N806
    K = r6(F * rng.uniform(0.85, 1.15))  # noqa: N806
    scale = 1.0 if inverse else F
    c_b = r6(rng.uniform(0.005, 0.12) * scale)
    p_a = r6(rng.uniform(0.005, 0.12) * scale)
    return {
        "inverse": inverse, "K": K, "S": r6(F * rng.uniform(0.98, 1.0)),
        "c_bid": c_b, "c_ask": r6(c_b * 1.03), "p_bid": r6(p_a * 0.97), "p_ask": p_a,
        "F_bid": r6(F * 0.9995), "F_ask": r6(F * 1.0005), "T": r6(rng.uniform(2 / 365, 1.0)),
    }  # fmt: skip


def case(q: dict, side: str, hedge_is_perp: bool) -> dict:
    inv, K, S, T = q["inverse"], q["K"], q["S"], q["T"]  # noqa: N806
    rates = P.FeeRates.from_table(TEST_FEES, side)
    lo, hi, band_reason = P.band(inv, K, q["c_bid"], q["c_ask"], q["p_bid"], q["p_ask"])
    out: dict = {"band": [lo, hi, band_reason]}
    for structure, (gross, why), legs in [
        ("conversion", P.conversion(inv, K, q["c_bid"], q["p_ask"], q["F_ask"]),
         (q["c_bid"], q["p_ask"], q["F_ask"])),
        ("reversal", P.reversal(inv, K, q["c_ask"], q["p_bid"], q["F_bid"]),
         (q["c_ask"], q["p_bid"], q["F_bid"])),
    ]:  # fmt: skip
        fees = P.structure_fees(inv, rates, K, *legs, S, hedge_is_perp)
        out[structure] = {"gross_edge": gross, "reason": why} | P.net_metrics(
            inv, gross, fees, K, S, T
        )
        if hedge_is_perp:
            usd = out[structure]["net_edge_usd"]
            out[structure]["breakeven_funding_ann"] = P.breakeven_funding(structure, usd, K, T)
    inv_fees = P.inventory_fees(inv, rates, q["c_bid"], q["p_ask"], S)
    out["inventory"] = {
        "fees": inv_fees,
        "r_lock": P.inventory_rate(inv, K, S, q["c_bid"], q["p_ask"], T, inv_fees)[0],
        "r_lock_gross": P.inventory_rate(inv, K, S, q["c_bid"], q["p_ask"], T)[0],
    }
    return {"inputs": q | {"side": side, "hedge_is_perp": hedge_is_perp}, "expected": out}


def guard_cases() -> list[dict]:
    return [
        {"fn": "inv_band", "args": [100, 1.1, 1.2, 0.05, 0.06],
         "expected": list(P.inv_band(100, 1.1, 1.2, 0.05, 0.06))},
        {"fn": "lin_conversion", "args": [100, 0.0, 5, 101],
         "expected": list(P.lin_conversion(100, 0.0, 5, 101))},
        {"fn": "inv_inventory_rate", "args": [100, 100, 1.5, 0.2, 0.1],
         "expected": list(P.inv_inventory_rate(100, 100, 1.5, 0.2, 0.1))},
        {"fn": "lin_inventory_rate", "args": [100, 100, 5, 5, 0.00005, 0.0, 1],
         "expected": list(P.lin_inventory_rate(100, 100, 5, 5, 0.00005, 0.0, 1))},
    ]  # fmt: skip


def build() -> str:
    rng = random.Random(20261014)
    cases = [
        case(quotes(rng, inverse), side, perp)
        for inverse in (True, False)
        for side in ("taker", "maker")
        for perp in (False, True)
        for _ in range(6)
    ]
    doc = {
        "generated_by": "python -m terminal_models.tools.gen_parity_vectors",
        "data": "Seeded math grid, not market data. fee_table is TEST-ONLY.",
        "fee_table": TEST_FEES,
        "fee_rates": {s: asdict(P.FeeRates.from_table(TEST_FEES, s)) for s in ("taker", "maker")},
        "cases": cases,
        "guards": guard_cases(),
    }
    return json.dumps(doc, indent=1, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the committed file differs")
    args = parser.parse_args(argv)
    path = contracts_dir() / "fixtures" / "parity" / "vectors.json"
    content = build()
    if args.check:
        if not path.exists() or path.read_text() != content:
            print(f"stale parity vectors: {path.name}; regenerate them", file=sys.stderr)
            return 1
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"wrote {path} ({len(json.loads(content)['cases'])} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
