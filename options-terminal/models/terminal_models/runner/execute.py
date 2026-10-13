"""Per-run pipeline (MODELS.md §6). Every run ends in an envelope, success or failed (M-17):

1. Resolve the implementation (caller) and its inputs.
2. Fetch inputs via adapters; reject stale inputs (M-8) and cross-source skew (M-9b).
3. Run the model in a spawned subprocess with the manifest timeout.
4. Validate: no NaN/Inf (C-3), schema + C-4 shapes, manifest bounds (hard), sanity checks (soft
   unless the check itself fails hard), payload size (C-6).
5. Build the envelope. Writing is writer.py's job.
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing
import os
import secrets
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from terminal_models.adapters.base import AdapterError, DataAdapter
from terminal_models.core.base import InputData, Model, ModelError, RunContext, Scope
from terminal_models.core.envelope import Envelope
from terminal_models.core.manifest import Manifest
from terminal_models.core.paths import MODELS_DIR
from terminal_models.core.slots import SlotDefinition
from terminal_models.core.validation import (
    SANITY_CHECKS,
    ValidationFailure,
    assert_no_nan_inf,
    check_bounds,
    check_payload_size,
    validate_envelope,
    validate_payload_full,
)
from terminal_models.runner.snapshots import SnapshotStore, input_hash

log = logging.getLogger("terminal_models.runner")


class RunFailed(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def uuid7() -> str:
    """Time-ordered UUIDv7 (RFC 9562). Python 3.12 has no uuid.uuid7."""
    ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(secrets.token_bytes(10), "big")
    value = (ms & ((1 << 48) - 1)) << 80 | 0x7 << 76 | (rand >> 68 & 0xFFF) << 64
    value |= 0b10 << 62 | (rand & ((1 << 62) - 1))
    h = f"{value:032x}"
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def utc(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha() -> str:
    """Commit the runner executes (envelope `code.git_sha`). TERMINAL_GIT_SHA wins (images)."""
    sha = os.environ.get("TERMINAL_GIT_SHA")
    if sha:
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=MODELS_DIR,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("cannot determine git SHA; set TERMINAL_GIT_SHA") from exc
    return out.stdout.strip()


def load_model(manifest: Manifest) -> Model:
    """library/<implementation_id>/model.py exposes IMPLEMENTATION, a Model subclass."""
    module = importlib.import_module(f"terminal_models.library.{manifest.implementation_id}.model")
    return module.IMPLEMENTATION(manifest)


# ---- model execution ----------------------------------------------------------------------


def _run_model(manifest: Manifest, scope: Scope, as_of: datetime, inputs: dict[str, InputData]):
    model = load_model(manifest)
    ctx = RunContext(
        as_of=as_of,
        scope=scope,
        inputs=inputs,
        params=manifest.params,
        logger=logging.getLogger(f"terminal_models.library.{manifest.implementation_id}"),
    )
    payload = model.run(ctx)
    return payload, ctx.warnings


def _child(conn, manifest_json: str, scope: Scope, as_of: datetime, inputs) -> None:
    try:
        manifest = Manifest.model_validate_json(manifest_json)
        conn.send(("ok", *_run_model(manifest, scope, as_of, inputs)))
    except ModelError as exc:
        conn.send(("model_error", exc.code, exc.message))
    except BaseException as exc:  # recorded as `unhandled` (M-6)
        conn.send(("error", type(exc).__name__, str(exc)))
    finally:
        conn.close()


def run_isolated(manifest: Manifest, scope: Scope, as_of: datetime, inputs: dict[str, InputData]):
    """Run in a spawned process; kill it at the manifest timeout (MODELS §6 step 3)."""
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    slim = {k: replace(v, raw=None) for k, v in inputs.items()}  # raw bodies stay in the parent
    proc = ctx.Process(
        target=_child, args=(child, manifest.model_dump_json(), scope, as_of, slim), daemon=True
    )
    proc.start()
    child.close()
    try:
        if not parent.poll(manifest.timeout_seconds):
            raise RunFailed("timeout", f"model exceeded {manifest.timeout_seconds}s")
        result = parent.recv()
    except EOFError as exc:
        raise RunFailed("unhandled", "model process exited without a result") from exc
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(5)
    if result[0] == "model_error":
        raise ModelError(result[1], result[2])
    if result[0] == "error":
        raise RunFailed("unhandled", f"{result[1]}: {result[2]}")
    return result[1], result[2]


# ---- pipeline -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    slot: SlotDefinition
    manifest: Manifest
    role: str
    scope: Scope
    as_of: datetime


def fetch_inputs(spec: RunSpec, adapters: dict[str, DataAdapter]) -> dict[str, InputData]:
    model = load_model(spec.manifest)
    inputs = {}
    for input_spec in model.required_inputs(spec.scope):
        adapter = adapters.get(input_spec.source)
        if adapter is None:
            raise RunFailed("bad_input_spec", f"no adapter '{input_spec.source}'")
        try:
            inputs[input_spec.name] = adapter.fetch(input_spec, spec.as_of)
        except AdapterError as exc:
            raise RunFailed(exc.code, exc.message) from exc
    return inputs


def check_input_times(spec: RunSpec, inputs: dict[str, InputData]) -> None:
    """M-8 staleness, M-9b cross-source skew."""
    max_age = spec.manifest.max_input_age_seconds
    for d in inputs.values():
        age = (spec.as_of - d.snapshot_at).total_seconds()
        if age > max_age:
            raise RunFailed("stale_input", f"{d.name} is {age:.0f}s old (max {max_age}s)")
    skew_limit = spec.manifest.max_input_skew_seconds
    if skew_limit and inputs:
        times = [d.snapshot_at for d in inputs.values()]
        skew = (max(times) - min(times)).total_seconds()
        if skew > skew_limit:
            raise RunFailed("input_skew", f"input snapshots span {skew:.0f}s (max {skew_limit}s)")


def validate_output(spec: RunSpec, payload: Any) -> list[dict[str, str]]:
    kind, version = spec.slot.payload_kind, spec.slot.payload_version
    assert_no_nan_inf(payload)
    validate_payload_full(kind, version, payload)
    check_bounds(kind, payload, spec.manifest.bounds)
    warnings = check_payload_size(payload)
    for name in spec.manifest.sanity_checks:
        warnings += SANITY_CHECKS[name](payload, spec.manifest.params)
    return warnings


def execute(
    spec: RunSpec,
    adapters: dict[str, DataAdapter],
    store: SnapshotStore | None,
    sha: str,
    inputs: dict[str, InputData] | None = None,
    isolate: bool = True,
) -> dict[str, Any]:
    """Run once and return the envelope. Never raises for run-level failures (M-17)."""
    started = datetime.now(UTC)
    payload, warnings, error = None, [], None
    digest = input_hash({})
    try:
        if inputs is None:
            inputs = fetch_inputs(spec, adapters)
        digest = store.save(inputs) if store else input_hash(inputs)
        check_input_times(spec, inputs)
        runner = run_isolated if isolate else _run_model
        payload, warnings = runner(spec.manifest, spec.scope, spec.as_of, inputs)
        warnings = list(warnings) + validate_output(spec, payload)
    except (RunFailed, ModelError, ValidationFailure) as exc:
        payload, error = None, {"code": exc.code, "message": exc.message}
    except Exception as exc:  # M-6: anything else is `unhandled`; the cause goes to the log
        log.exception("unhandled error in run")
        payload, error = None, {"code": "unhandled", "message": f"{type(exc).__name__}: {exc}"}

    envelope = Envelope(
        run_id=uuid7(),
        slot_id=spec.slot.slot_id,
        implementation_id=spec.manifest.implementation_id,
        implementation_version=spec.manifest.version,
        scope=spec.scope,
        status="failed" if error else "success",
        as_of=utc(spec.as_of),
        started_at=utc(started),
        finished_at=utc(datetime.now(UTC)),
        inputs={
            "sources": [
                {"name": d.name, "snapshot_at": utc(d.snapshot_at)}
                for d in sorted((inputs or {}).values(), key=lambda d: d.name)
            ],
            "hash": digest,
        },
        code={"git_sha": sha},
        warnings=[] if error else warnings,
        payload_kind=spec.slot.payload_kind,
        payload_version=spec.slot.payload_version,
        payload=payload,
        error=error,
    ).to_json_dict()
    validate_envelope(envelope)
    return envelope
