"""Record raw Deribit responses for fixtures, golden tests and replay (guide §4.4, §5.5).

    uv run python -m terminal_models.tools.record_deribit --out <new dir> \\
        [--assets BTC,ETH,SOL,HYPE] [--allow-partial]

Run it anywhere www.deribit.com is reachable. It fetches exactly what a bs_pricer_black76 run
fetches (the model's own required_inputs, through the production DeribitAdapter), plus BTC/ETH
DVOL history, and saves every response body byte-for-byte with its request and local receive
time. Nothing is parsed into new numbers here; tools/build_fixtures.py replays the recording.

A required call that fails after retries fails the recording: no manifest.json is written and
FAILED.txt says why, so a partial recording can't pass for a complete one. --allow-partial keeps
the assets that succeeded and lists the failures in the manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from terminal_models.adapters.base import AdapterError
from terminal_models.adapters.deribit import DeribitAdapter
from terminal_models.adapters.deribit.chains import BOOKS
from terminal_models.adapters.deribit.client import BASE_URL, DeribitClient
from terminal_models.core.manifest import load_manifest
from terminal_models.core.paths import library_dir
from terminal_models.runner.execute import git_sha, load_model

IMPLEMENTATION = "bs_pricer_black76"
DVOL_HISTORY = {"BTC": "BTC", "ETH": "ETH"}


def utc(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Recorder:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.calls_dir = out / "calls"
        self.calls_dir.mkdir(parents=True)
        self.entries: list[dict[str, Any]] = []

    def capture(self, method: str, params: dict[str, Any], text: str) -> None:
        seq = len(self.entries) + 1
        stem = f"{seq:03d}_{method}"
        raw = text.encode()
        body_rel, meta_rel = f"calls/{stem}.body.gz", f"calls/{stem}.meta.json"
        (self.out / body_rel).write_bytes(gzip.compress(raw, mtime=0))
        meta = {
            "seq": seq,
            "method": method,
            "params": {k: v for k, v in params.items()},
            "url": f"{BASE_URL}/api/v2/public/{method}?{urlencode(params)}",
            "received_at_local": utc(datetime.now(UTC)),
            "body_bytes": len(raw),
        }
        (self.out / meta_rel).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
        self.entries.append(
            {
                "seq": seq,
                "method": method,
                "meta": meta_rel,
                "body": body_rel,
                "body_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )


def record(out: Path, assets: list[str], allow_partial: bool, client_factory=DeribitClient) -> int:
    if out.exists() and any(out.iterdir()):
        print(f"{out} is not empty; record into a new directory", file=sys.stderr)
        return 2
    rec = Recorder(out)
    client = client_factory(capture=rec.capture)
    adapter = DeribitAdapter(client)
    manifest = load_manifest(library_dir() / IMPLEMENTATION / "manifest.yaml")
    model = load_model(manifest)
    started = datetime.now(UTC)
    results: dict[str, dict[str, Any]] = {}

    for asset in assets:
        status: dict[str, Any] = {"ok": True, "inputs": {}, "optional_errors": []}
        try:
            for spec in model.required_inputs({"underlying": asset}):
                data = adapter.fetch(spec, started)
                status["inputs"][spec.name] = {"snapshot_at": utc(data.snapshot_at)}
        except AdapterError as exc:
            status = {"ok": False, "error": {"code": exc.code, "message": exc.message}}
        if status["ok"] and asset in DVOL_HISTORY:
            lookback = timedelta(days=manifest.params["candle_lookback_days"])
            try:
                client.volatility_index(DVOL_HISTORY[asset], started - lookback, started)
            except AdapterError as exc:  # display/validation data only: not fatal
                status["optional_errors"].append({"code": exc.code, "message": exc.message})
        results[asset] = status
        print(
            f"{asset}: {'ok' if status['ok'] else 'FAILED ' + status['error']['message']}",
            file=sys.stderr,
        )

    failed = [a for a, s in results.items() if not s["ok"]]
    if failed and not allow_partial:
        (out / "FAILED.txt").write_text(
            "Recording incomplete; no manifest written.\n"
            + "".join(
                f"{a}: {results[a]['error']['code']}: {results[a]['error']['message']}\n"
                for a in failed
            )
        )
        print(
            f"recording failed for {', '.join(failed)}; see {out / 'FAILED.txt'}", file=sys.stderr
        )
        return 1

    doc = {
        "recorder": "terminal_models.tools.record_deribit",
        "git_sha": git_sha(),
        "base_url": BASE_URL,
        "started_at": utc(started),
        "finished_at": utc(datetime.now(UTC)),
        "implementation": IMPLEMENTATION,
        "complete": not failed,
        "assets": results,
        "calls": rec.entries,
    }
    (out / "manifest.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"recorded {len(rec.entries)} calls for {', '.join(assets)} into {out}", file=sys.stderr)
    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m terminal_models.tools.record_deribit")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assets", default=",".join(BOOKS))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]
    unknown = [a for a in assets if a not in BOOKS]
    if unknown:
        parser.error(f"unknown assets: {', '.join(unknown)}")
    return record(args.out, assets, args.allow_partial)


if __name__ == "__main__":
    sys.exit(main())
