"""Turn a Deribit recording into contract fixtures (guide §4.4).

    uv run python -m terminal_models.tools.build_fixtures --recording <dir> [--check]

For each recorded asset and each implementation in the recording (bs_pricer_black76,
parity_deribit, rv_har): replay the raw bodies through the production Deribit and Hyperliquid
adapters and the runner's fetch path (declared and follow-up inputs), run the model via the runner
pipeline (full validation, sanity checks), freeze the parts that vary per execution (run_id,
started/finished, git SHA), and write contracts/fixtures/valid/composite/<prefix>_<asset>.json.
A run that fails on real data is written as the failed envelope and reported: real-data failures
are findings, not things to tune away. Invalid fixtures are derived from the BTC fixture, one
broken rule each.

parity_deribit is skipped while its manifest's fee table is null (it can't run without fees).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from terminal_models.adapters.deribit import DeribitAdapter
from terminal_models.adapters.deribit.replay import RecordedClient
from terminal_models.adapters.hyperliquid import HyperliquidAdapter
from terminal_models.adapters.hyperliquid.replay import RecordedHyperliquidClient
from terminal_models.core.base import InputData
from terminal_models.core.manifest import load_manifest
from terminal_models.core.paths import contracts_dir, library_dir
from terminal_models.core.payloads import get_section, records, scalars
from terminal_models.core.registry import null_fees
from terminal_models.core.slots import load_slots
from terminal_models.runner.execute import RunSpec, execute, fetch_inputs

IMPLEMENTATION = "bs_pricer_black76"
PREFIX = {  # fixture file prefix
    "bs_pricer_black76": "bs_pricer",
    "parity_deribit": "parity",
    "rv_har": "vrp",
}
FIXTURE_SHA = "0" * 40  # fixtures are code-version independent; the producer is in the report


def frozen_run_id(as_of: datetime, asset: str) -> str:
    """Deterministic UUIDv7: as_of milliseconds + bits from the asset name."""
    ms = int(as_of.timestamp() * 1000)
    rand = int.from_bytes(hashlib.sha256(asset.encode()).digest()[:10], "big")
    value = (ms & ((1 << 48) - 1)) << 80 | 0x7 << 76 | (rand >> 68 & 0xFFF) << 64
    value |= 0b10 << 62 | (rand & ((1 << 62) - 1))
    h = f"{value:032x}"
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def replay_inputs(
    recording: Path, asset: str, implementation: str = IMPLEMENTATION
) -> tuple[dict[str, InputData], datetime]:
    """Inputs for one asset from the recording (declared and follow-up), and the run's as_of:
    the latest server time among them, floored to the second, so staleness checks are real."""
    client = RecordedClient(recording)
    manifest = load_manifest(library_dir() / implementation / "manifest.yaml")
    started = datetime.fromisoformat(client.manifest["started_at"])
    spec = RunSpec(
        load_slots()[manifest.fills_slot], manifest, "shadow", {"underlying": asset}, started
    )
    adapters = {
        "deribit": DeribitAdapter(client),
        "hyperliquid": HyperliquidAdapter(RecordedHyperliquidClient(recording)),
    }
    inputs = fetch_inputs(spec, adapters)
    latest = max(d.snapshot_at for d in inputs.values())
    return inputs, latest.replace(microsecond=0)


def run_asset(recording: Path, asset: str, implementation: str = IMPLEMENTATION) -> dict[str, Any]:
    inputs, as_of = replay_inputs(recording, asset, implementation)
    manifest = load_manifest(library_dir() / implementation / "manifest.yaml")
    slot = load_slots()[manifest.fills_slot]
    spec = RunSpec(slot, manifest, "shadow", {"underlying": asset}, as_of)
    env = execute(spec, {}, None, FIXTURE_SHA, inputs, isolate=False)
    env |= {
        "run_id": frozen_run_id(as_of, asset),
        "started_at": env["as_of"],
        "finished_at": env["as_of"],
    }
    return env


def invalid_variants(env: dict[str, Any], implementation: str = IMPLEMENTATION) -> dict[str, str]:
    """One broken rule each (CONTRACTS §1: the filename says which)."""
    if implementation == "parity_deribit":
        return parity_invalid_variants(env)
    if implementation == "rv_har":
        return vrp_invalid_variants(env)
    out: dict[str, str] = {}

    nan = copy.deepcopy(env)
    chain = get_section(nan["payload"], "chain")
    col = [c["key"] for c in chain["columns"]].index("iv_mark")
    chain["rows"][0][col] = float("nan")
    out["nan_iv"] = json.dumps(nan, indent=1, allow_nan=True) + "\n"

    short_row = copy.deepcopy(env)
    chain = get_section(short_row["payload"], "chain")
    col = [c["key"] for c in chain["columns"]].index("fair_usd")
    del chain["rows"][0][col]
    out["chain_row_missing_fair_usd"] = json.dumps(short_row, indent=1) + "\n"

    smile = copy.deepcopy(env)
    get_section(smile["payload"], "smile")["series"][0]["values"].pop()
    out["smile_series_wrong_length"] = json.dumps(smile, indent=1) + "\n"
    return out


def parity_invalid_variants(env: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    nan = copy.deepcopy(env)
    bands = get_section(nan["payload"], "bands")
    col = [c["key"] for c in bands["columns"]].index("f_impl_bid")
    bands["rows"][0][col] = float("nan")
    out["nan_band"] = json.dumps(nan, indent=1, allow_nan=True) + "\n"

    short = copy.deepcopy(env)
    expiries = get_section(short["payload"], "expiries")
    col = [c["key"] for c in expiries["columns"]].index("f_impl")
    del expiries["rows"][0][col]
    out["expiry_row_missing_f_impl"] = json.dumps(short, indent=1) + "\n"

    curve = copy.deepcopy(env)
    get_section(curve["payload"], "carry_curve")["series"][0]["values"].append(0.0)
    out["carry_series_wrong_length"] = json.dumps(curve, indent=1) + "\n"
    return out


def vrp_invalid_variants(env: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    nan = copy.deepcopy(env)
    signal = get_section(nan["payload"], "signal")
    col = [c["key"] for c in signal["columns"]].index("rv_forecast")
    signal["rows"][0][col] = float("nan")
    out["nan_rv_forecast"] = json.dumps(nan, indent=1, allow_nan=True) + "\n"

    short = copy.deepcopy(env)
    signal = get_section(short["payload"], "signal")
    col = [c["key"] for c in signal["columns"]].index("vrp_z")
    del signal["rows"][0][col]
    out["signal_row_missing_vrp_z"] = json.dumps(short, indent=1) + "\n"

    cone = copy.deepcopy(env)
    get_section(cone["payload"], "cone")["series"][0]["values"].pop()
    out["cone_series_wrong_length"] = json.dumps(cone, indent=1) + "\n"
    return out


def vrp_report(asset: str, env: dict[str, Any]) -> list[str]:
    lines = [f"{asset}: {env['status']} as_of {env['as_of']}"]
    if env["status"] == "failed":
        return lines + [f"  FAILED {env['error']['code']}: {env['error']['message']}"]
    lines += [f"  warning {w['code']}: {w['message']}" for w in env["warnings"]]
    context = scalars(get_section(env["payload"], "context"))
    lines.append(
        f"  days {context['n_days']} (missing {context['n_days_missing']}), "
        f"signal_enabled {context['signal_enabled']}, dvol {context['dvol']}"
    )
    for r in records(get_section(env["payload"], "signal")):
        lines.append(
            f"  {r['tenor_days']}d iv {r['iv_impl']} rv {r['rv_forecast']} z {r['vrp_z']} "
            f"flags {r['flags'] or '-'}"
        )
    return lines


def parity_report(asset: str, env: dict[str, Any]) -> list[str]:
    lines = [f"{asset}: {env['status']} as_of {env['as_of']}"]
    if env["status"] == "failed":
        return lines + [f"  FAILED {env['error']['code']}: {env['error']['message']}"]
    lines += [f"  warning {w['code']}: {w['message']}" for w in env["warnings"]]
    expiries = records(get_section(env["payload"], "expiries"))
    opps = records(get_section(env["payload"], "opportunities"))
    hedges: dict[str, int] = {}
    for r in expiries:
        hedges[str(r["hedge_type"])] = hedges.get(str(r["hedge_type"]), 0) + 1
    lines.append(f"  expiries {len(expiries)} by hedge: {hedges}")
    lines.append(
        f"  opportunities {len(opps)}, confirmed {sum(1 for o in opps if o['confirmed'])}, "
        f"skew-flagged {sum(1 for o in opps if 'snapshot_skew' in o['flags'])}"
    )
    return lines


def report(asset: str, env: dict[str, Any]) -> list[str]:
    lines = [f"{asset}: {env['status']} as_of {env['as_of']}"]
    if env["status"] == "failed":
        lines.append(f"  FAILED {env['error']['code']}: {env['error']['message']}")
        return lines
    lines += [f"  warning {w['code']}: {w['message']}" for w in env["warnings"]]
    nodes = [r for r in records(get_section(env["payload"], "chain")) if r["is_node"]]
    counts: dict[str, int] = {}
    for r in nodes:
        counts[r["node_quality"]] = counts.get(r["node_quality"], 0) + 1
    lines.append(f"  nodes {len(nodes)}: {counts}")
    if asset == "HYPE":
        have = {"wide", "one_sided"} <= set(counts)
        dropped = any(w["code"] == "expiry_dropped_insufficient_nodes" for w in env["warnings"])
        lines.append(f"  guide §4.4 HYPE coverage: wide+one_sided={have} dropped_expiry={dropped}")
    return lines


REPORTS = {"bs_pricer": report, "parity": parity_report, "vrp": vrp_report}


def build(recording: Path) -> tuple[dict[Path, str], list[str]]:
    manifest = json.loads((recording / "manifest.json").read_text())
    assets = [a for a, s in manifest["assets"].items() if s["ok"]]
    root = contracts_dir() / "fixtures"
    files: dict[Path, str] = {}
    notes = [
        f"recording {recording.name} (started {manifest['started_at']}, "
        f"recorder {manifest['git_sha'][:12]})"
    ]
    for implementation, prefix in PREFIX.items():
        if implementation not in manifest["implementations"]:
            notes.append(f"{implementation}: skipped, not in this recording")
            continue
        impl_manifest = load_manifest(library_dir() / implementation / "manifest.yaml")
        if null_fees(impl_manifest):
            notes.append(f"{implementation}: skipped, params.fees is null")
            continue
        notes.append(f"{implementation}:")
        envs = {}
        for asset in assets:
            env = run_asset(recording, asset, implementation)
            envs[asset] = env
            files[root / "valid" / "composite" / f"{prefix}_{asset.lower()}.json"] = (
                json.dumps(env, indent=1, allow_nan=False) + "\n"
            )
            notes += REPORTS[prefix](asset, env)
        btc = envs.get("BTC")
        if btc and btc["status"] == "success":
            for name, text in invalid_variants(btc, implementation).items():
                files[root / "invalid" / "composite" / f"{prefix}_btc__{name}.json"] = text
        else:
            notes.append(f"no successful BTC {implementation} run: invalid fixtures not rebuilt")
    return files, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m terminal_models.tools.build_fixtures")
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="fail if committed fixtures differ")
    args = parser.parse_args(argv)
    files, notes = build(args.recording)
    stale = []
    for path, text in files.items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path.name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    print("\n".join(notes))
    if stale:
        print(f"stale fixtures: {', '.join(stale)}; rebuild them", file=sys.stderr)
        return 1
    if not args.check:
        print(f"wrote {len(files)} fixture files at {datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
