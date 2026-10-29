"""`python -m terminal_models.runner …` (MODELS.md §6).

run <slot> [--scope k=v …] [--role production|shadow|all] [--dry-run]
          [--snapshot <hash> [--as-of <UTC>]]
run-due                         # everything whose schedule is due now (scheduler entrypoint)
replay <run_id> | --envelope <file>
sync-registry                   # M-14: registry.yaml → slots table, then NOTIFY slots
validate-registry [--registry <file>]
diff <slot> --against shadow:<impl>   # not implemented yet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from terminal_models.adapters.deribit import DeribitAdapter
from terminal_models.adapters.hyperliquid import HyperliquidAdapter
from terminal_models.adapters.rv_daily import DayCache
from terminal_models.core.manifest import Manifest
from terminal_models.core.paths import contracts_dir, library_dir, registry_path
from terminal_models.core.registry import Registry, RegistryError, load_registry, validate_registry
from terminal_models.core.scope import scope_key
from terminal_models.core.slots import SlotDefinition, load_slots
from terminal_models.runner.execute import RunSpec, execute, git_sha
from terminal_models.runner.schedule import is_due
from terminal_models.runner.snapshots import SnapshotStore

log = logging.getLogger("terminal_models.runner")


def _scope_arg(value: str) -> tuple[str, str]:
    key, sep, val = value.partition("=")
    if not sep or not key or not val:
        raise argparse.ArgumentTypeError(f"expected key=value, got '{value}'")
    return key, val


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m terminal_models.runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one slot")
    run.add_argument("slot")
    run.add_argument("--scope", type=_scope_arg, action="append", default=[])
    run.add_argument("--role", choices=["production", "shadow", "all"], default="all")
    run.add_argument("--dry-run", action="store_true", help="validate and print; do not write")
    run.add_argument("--snapshot", help="run on stored inputs (input hash) instead of fetching")
    run.add_argument("--as-of", help="valuation time (UTC, ends in Z); requires --snapshot")

    sub.add_parser("run-due", help="run everything whose schedule is due")
    sub.add_parser("sync-registry", help="sync registry.yaml into the slots table")

    diff = sub.add_parser("diff", help="shadow diff report")
    diff.add_argument("slot")
    diff.add_argument("--against", required=True)
    diff.add_argument("--from", dest="from_")
    diff.add_argument("--to")

    validate = sub.add_parser("validate-registry", help="check registry.yaml (M-13)")
    validate.add_argument("--registry", type=Path, default=None)

    replay = sub.add_parser("replay", help="re-run with recorded inputs; assert identical output")
    replay.add_argument("run_id", nargs="?")
    replay.add_argument("--envelope", type=Path, help="replay a stored envelope file instead")
    return parser


class Context:
    def __init__(self, registry_file: Path | None = None) -> None:
        self.registry_file = registry_file or registry_path()
        self.registry: Registry = load_registry(self.registry_file)
        self.slots: dict[str, SlotDefinition] = load_slots(contracts_dir())
        self.manifests: dict[str, Manifest] = validate_registry(
            self.registry, self.slots, library_dir()
        )


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _parse_utc(value: str) -> datetime:
    if not value.endswith("Z"):
        raise SystemExit("--as-of must be UTC and end in Z")
    return datetime.fromisoformat(value).astimezone(UTC)


def _adapters() -> dict:
    rv_cache = DayCache.default()  # completed-day candles (rv_daily), fetched once
    return {
        "deribit": DeribitAdapter(rv_cache=rv_cache),
        "hyperliquid": HyperliquidAdapter(rv_cache=rv_cache),
    }


def _summary(envelope: dict, role: str) -> str:
    return json.dumps(
        {
            "event": "run_finished",
            "run_id": envelope["run_id"],
            "slot_id": envelope["slot_id"],
            "implementation": envelope["implementation_id"],
            "version": envelope["implementation_version"],
            "role": role,
            "scope": envelope["scope"],
            "status": envelope["status"],
            "error": envelope.get("error"),
            "warnings": len(envelope["warnings"]),
            "started_at": envelope["started_at"],
            "finished_at": envelope["finished_at"],
        }
    )


def _targets(ctx: Context, slot_id: str, role: str):
    entry = ctx.registry.slots.get(slot_id)
    if entry is None:
        raise SystemExit(f"slot '{slot_id}' is not in the registry")
    for r, ref in entry.refs():
        if role in (r, "all"):
            yield r, ctx.manifests[ref.implementation]


def cmd_run(args) -> int:
    ctx = Context()
    entry = ctx.registry.slots.get(args.slot)
    if entry is None:
        print(f"slot '{args.slot}' is not in the registry", file=sys.stderr)
        return 2
    if args.as_of and not args.snapshot:
        print("--as-of needs --snapshot: live inputs are always 'now'", file=sys.stderr)
        return 2
    scopes = [dict(args.scope)] if args.scope else entry.scopes
    as_of = _parse_utc(args.as_of) if args.as_of else now_utc()
    store = SnapshotStore()
    inputs = store.load(args.snapshot) if args.snapshot else None
    adapters = _adapters()
    sha = git_sha()
    targets = list(_targets(ctx, args.slot, args.role))
    if not targets:
        print(f"no {args.role} implementation for '{args.slot}'", file=sys.stderr)
        return 2

    writer = None
    if not args.dry_run:
        from terminal_models.runner.writer import Writer

        writer = Writer()
        writer.sync_registry(ctx.registry, ctx.slots)
    failed = 0
    try:
        for role, manifest in targets:
            for scope in scopes:
                spec = RunSpec(ctx.slots[args.slot], manifest, role, scope, as_of)
                if writer is None:
                    envelope = execute(spec, adapters, store, sha, inputs)
                    print(json.dumps(envelope, indent=2))
                else:
                    with writer.lock(args.slot, scope_key(scope), role) as acquired:
                        if not acquired:
                            print(
                                f"skipped {args.slot} {scope} {role}: already running",
                                file=sys.stderr,
                            )
                            continue
                        envelope = execute(spec, adapters, store, sha, inputs)
                        writer.write_run(envelope, role, scope_key(scope))
                print(_summary(envelope, role), file=sys.stderr)
                failed += envelope["status"] == "failed"
    finally:
        if writer:
            writer.close()
    return 1 if failed else 0


def cmd_run_due(_args) -> int:
    from terminal_models.runner.writer import Writer

    ctx = Context()
    at = now_utc().replace(second=0)
    store, adapters, sha = SnapshotStore(), _adapters(), git_sha()
    writer = Writer()
    failed = 0
    try:
        writer.sync_registry(ctx.registry, ctx.slots)
        for slot_id, entry in ctx.registry.slots.items():
            for role, ref in entry.refs():
                manifest = ctx.manifests[ref.implementation]
                if not is_due(manifest.schedule, at):
                    continue
                for scope in entry.scopes:
                    key = scope_key(scope)
                    with writer.lock(slot_id, key, role) as acquired:
                        if not acquired:
                            continue
                        spec = RunSpec(ctx.slots[slot_id], manifest, role, scope, now_utc())
                        envelope = execute(spec, adapters, store, sha)
                        writer.write_run(envelope, role, key)
                    print(_summary(envelope, role), file=sys.stderr)
                    failed += envelope["status"] == "failed"
    finally:
        writer.close()
    return 1 if failed else 0


def cmd_sync_registry(_args) -> int:
    from terminal_models.runner.writer import Writer

    ctx = Context()
    writer = Writer()
    try:
        changed = writer.sync_registry(ctx.registry, ctx.slots)
    finally:
        writer.close()
    print("slots table updated; NOTIFY slots sent" if changed else "slots table already in sync")
    return 0


def cmd_replay(args) -> int:
    ctx = Context()
    if args.envelope:
        stored, role = json.loads(args.envelope.read_text()), "production"
    elif args.run_id:
        from terminal_models.runner.writer import Writer

        writer = Writer()
        try:
            found = writer.fetch_envelope(args.run_id)
        finally:
            writer.close()
        if found is None:
            print(f"run {args.run_id} not found", file=sys.stderr)
            return 2
        stored, role = found
    else:
        print("replay needs a run_id or --envelope", file=sys.stderr)
        return 2

    manifest = ctx.manifests.get(stored["implementation_id"])
    if manifest is None or manifest.version != stored["implementation_version"]:
        print(
            f"{stored['implementation_id']} {stored['implementation_version']} is not the "
            "registered version; check out the commit in code.git_sha",
            file=sys.stderr,
        )
        return 2
    store = SnapshotStore()
    inputs = store.load(stored["inputs"]["hash"])
    spec = RunSpec(
        ctx.slots[stored["slot_id"]], manifest, role, stored["scope"], _parse_utc(stored["as_of"])
    )
    fresh = execute(spec, {}, None, stored["code"]["git_sha"], inputs)
    fields = ("status", "payload", "warnings", "error")
    diffs = [f for f in fields if fresh.get(f) != stored.get(f)]
    if diffs:
        print(f"replay differs in: {', '.join(diffs)}", file=sys.stderr)
        return 1
    print(f"replay identical: {stored['run_id']}")
    return 0


def cmd_validate_registry(registry: Path | None) -> int:
    path = registry or registry_path()
    try:
        ctx = Context(path)
    except RegistryError as exc:
        for problem in exc.problems:
            print(f"registry: {problem}", file=sys.stderr)
        return 1
    print(f"registry ok: {path} ({len(ctx.manifests)} implementations)")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format='{"level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    )
    args = build_parser().parse_args(argv)
    if args.command == "validate-registry":
        return cmd_validate_registry(args.registry)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "run-due":
        return cmd_run_due(args)
    if args.command == "sync-registry":
        return cmd_sync_registry(args)
    if args.command == "replay":
        return cmd_replay(args)
    print(f"'{args.command}' is not implemented yet", file=sys.stderr)
    return 2
