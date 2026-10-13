"""Validation gate (MODELS.md §6 step 4). Nothing reaches the store unless it passes."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from terminal_models.core.paths import contracts_dir

PAYLOAD_SOFT_LIMIT_BYTES = 2 * 1024 * 1024  # C-6: SHOULD stay under
PAYLOAD_HARD_LIMIT_BYTES = 5 * 1024 * 1024  # C-6: rejected over


class ValidationFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@cache
def _validator(relative: str, root: Path) -> Draft202012Validator:
    path = root / "schemas" / relative
    if not path.exists():
        raise ValidationFailure("schema_missing", f"No schema at contracts/schemas/{relative}")
    with path.open() as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _check(validator: Draft202012Validator, instance: Any, code: str) -> None:
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        where = "/".join(str(p) for p in first.absolute_path) or "<root>"
        raise ValidationFailure(code, f"{where}: {first.message}")


def validate_envelope(envelope: dict[str, Any], root: Path | None = None) -> None:
    _check(_validator("envelope/v1.json", root or contracts_dir()), envelope, "invalid_envelope")


def validate_payload(kind: str, version: str, payload: Any, root: Path | None = None) -> None:
    validator = _validator(f"payloads/{kind}/v{version}.json", root or contracts_dir())
    _check(validator, payload, "invalid_payload")


def assert_no_nan_inf(value: Any, path: str = "") -> None:
    """C-3 / X-3: missing values are null, never NaN or Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        where = path or "<root>"
        raise ValidationFailure("non_finite", f"{where}: {value!r} is not allowed; use null")
    if isinstance(value, dict):
        for k, v in value.items():
            assert_no_nan_inf(v, f"{path}/{k}")
    elif isinstance(value, list | tuple):
        for i, v in enumerate(value):
            assert_no_nan_inf(v, f"{path}/{i}")


def payload_size_bytes(payload: Any) -> int:
    return len(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode())


def check_payload_size(payload: Any) -> list[dict[str, str]]:
    """C-6. Raises over the hard limit; returns a warning over the soft limit."""
    size = payload_size_bytes(payload)
    if size > PAYLOAD_HARD_LIMIT_BYTES:
        raise ValidationFailure("payload_too_large", f"payload is {size} bytes (limit 5 MB)")
    if size > PAYLOAD_SOFT_LIMIT_BYTES:
        return [{"code": "payload_large", "message": f"payload is {size} bytes (target < 2 MB)"}]
    return []


SanityCheck = Callable[[dict[str, Any]], list[dict[str, str]]]


def _pending(name: str) -> SanityCheck:
    def check(payload: dict[str, Any]) -> list[dict[str, str]]:
        raise NotImplementedError(f"sanity check '{name}' is implemented in M1")

    return check


# MODELS.md §6. Implementations opt in by name via manifest `sanity_checks`.
SANITY_CHECKS: dict[str, SanityCheck] = {
    name: _pending(name)
    for name in (
        "calendar_arbitrage",
        "butterfly_arbitrage",
        "put_call_parity",
        "monotonic_tenor",
        "greeks_signs",
    )
}
