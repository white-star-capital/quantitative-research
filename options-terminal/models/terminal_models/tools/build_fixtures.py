"""Turn a Deribit recording into contract fixtures (guide §4.4).

    uv run python -m terminal_models.tools.build_fixtures --recording <dir> [--check]

For each recorded asset: replay the raw bodies through the production DeribitAdapter, run
bs_pricer_black76 via the runner pipeline (full validation, sanity checks), freeze the parts that
vary per execution (run_id, started/finished, git SHA), and write
contracts/fixtures/valid/composite/bs_pricer_<asset>.json. A run that fails on real data is
written as the failed envelope and reported: real-data failures are findings, not things to tune
away. Invalid fixtures are derived from the BTC fixture, one broken rule each.
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
from terminal_models.core.base import InputData
from terminal_models.core.manifest import load_manifest
from terminal_models.core.paths import contracts_dir, library_dir
from terminal_models.core.payloads import get_section, records
from terminal_models.core.slots import load_slots
from terminal_models.runner.execute import RunSpec, execute, load_model

IMPLEMENTATION = "bs_pricer_black76"
FIXTURE_SHA = "0" * 40  # fixtures are code-version independent; the producer is in the report


def frozen_run_id(as_of: datetime, asset: str) -> str:
    """Deterministic UUIDv7: as_of milliseconds + bits from the asset name."""
    ms = int(as_of.timestamp() * 1000)
    rand = int.from_bytes(hashlib.sha256(asset.encode()).digest()[:10], "big")
    value = (ms & ((1 << 48) - 1)) << 80 | 0x7 << 76 | (rand >> 68 & 0xFFF) << 64
    value |= 0b10 << 62 | (rand & ((1 << 62) - 1))
    h = f"{value:032x}"
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def replay_inputs(recording: Path, asset: str) -> tuple[dict[str, InputData], datetime]:
    """Inputs for one asset from the recording, and the run's as_of: the latest server time
    (usOut) among them, floored to the second, so staleness checks are real."""
    client = RecordedClient(recording)
    adapter = DeribitAdapter(client)
    manifest = load_manifest(library_dir() / IMPLEMENTATION / "manifest.yaml")
    model = load_model(manifest)
    started = datetime.fromisoformat(client.manifest["started_at"])
    inputs = {
        spec.name: adapter.fetch(spec, started)
        for spec in model.required_inputs({"underlying": asset})
    }
    latest = max(d.snapshot_at for d in inputs.values())
    return inputs, latest.replace(microsecond=0)


def run_asset(recording: Path, asset: str) -> dict[str, Any]:
    inputs, as_of = replay_inputs(recording, asset)
    manifest = load_manifest(library_dir() / IMPLEMENTATION / "manifest.yaml")
    spec = RunSpec(load_slots()["bs_pricer"], manifest, "shadow", {"underlying": asset}, as_of)
    env = execute(spec, {}, None, FIXTURE_SHA, inputs, isolate=False)
    env |= {
        "run_id": frozen_run_id(as_of, asset),
        "started_at": env["as_of"],
        "finished_at": env["as_of"],
    }
    return env


def invalid_variants(env: dict[str, Any]) -> dict[str, str]:
    """One broken rule each (CONTRACTS §1: the filename says which)."""
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


def build(recording: Path) -> tuple[dict[Path, str], list[str]]:
    manifest = json.loads((recording / "manifest.json").read_text())
    assets = [a for a, s in manifest["assets"].items() if s["ok"]]
    root = contracts_dir() / "fixtures"
    files: dict[Path, str] = {}
    notes = [
        f"recording {recording.name} (started {manifest['started_at']}, "
        f"recorder {manifest['git_sha'][:12]})"
    ]
    envs = {}
    for asset in assets:
        env = run_asset(recording, asset)
        envs[asset] = env
        files[root / "valid" / "composite" / f"bs_pricer_{asset.lower()}.json"] = (
            json.dumps(env, indent=1, allow_nan=False) + "\n"
        )
        notes += report(asset, env)
    btc = envs.get("BTC")
    if btc and btc["status"] == "success":
        for name, text in invalid_variants(btc).items():
            files[root / "invalid" / "composite" / f"bs_pricer_btc__{name}.json"] = text
    else:
        notes.append("no successful BTC run: invalid fixtures not regenerated")
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
