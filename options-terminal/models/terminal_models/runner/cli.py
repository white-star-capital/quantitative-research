"""`python -m terminal_models.runner …` (MODELS.md §6)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from terminal_models.core.paths import contracts_dir, library_dir, registry_path
from terminal_models.core.registry import RegistryError, load_registry, validate_registry
from terminal_models.core.slots import load_slots


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
    run.add_argument("--as-of")
    run.add_argument("--role", choices=["production", "shadow", "all"], default="production")
    run.add_argument("--dry-run", action="store_true")

    sub.add_parser("run-due", help="run everything whose schedule is due")

    diff = sub.add_parser("diff", help="shadow diff report")
    diff.add_argument("slot")
    diff.add_argument("--against", required=True)
    diff.add_argument("--from", dest="from_")
    diff.add_argument("--to")

    validate = sub.add_parser("validate-registry", help="check registry.yaml (M-13)")
    validate.add_argument("--registry", type=Path, default=None)

    replay = sub.add_parser("replay", help="re-run with recorded inputs; assert identical output")
    replay.add_argument("run_id")
    return parser


def cmd_validate_registry(registry: Path | None) -> int:
    path = registry or registry_path()
    try:
        registry_ = load_registry(path)
        manifests = validate_registry(registry_, load_slots(contracts_dir()), library_dir())
    except RegistryError as exc:
        for problem in exc.problems:
            print(f"registry: {problem}", file=sys.stderr)
        return 1
    print(f"registry ok: {path} ({len(manifests)} implementations)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-registry":
        return cmd_validate_registry(args.registry)
    print(f"'{args.command}' is not implemented yet (M1)", file=sys.stderr)
    return 2
