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


def validate_payload_full(kind: str, version: str, payload: Any, root: Path | None = None) -> None:
    """Schema, then C-4 shape rules, recursing into composite sections (nesting depth 1)."""
    validate_payload(kind, version, payload, root)
    check_shapes(kind, payload)
    if kind == "composite":
        keys = [section["key"] for section in payload["sections"]]
        if len(keys) != len(set(keys)):
            raise ValidationFailure("invalid_payload", "composite: section keys must be unique")
        for section in payload["sections"]:
            try:
                validate_payload_full(
                    section["payload_kind"], section["payload_version"], section["payload"], root
                )
            except ValidationFailure as exc:
                raise ValidationFailure(
                    exc.code, f"section {section['key']}: {exc.message}"
                ) from exc


def _shape_error(message: str) -> ValidationFailure:
    return ValidationFailure("invalid_shape", message)


def _check_series(series: list[dict[str, Any]], n: int, axis: str) -> None:
    keys = [s["key"] for s in series]
    if len(keys) != len(set(keys)):
        raise _shape_error("series keys must be unique")
    for s in series:
        if len(s["values"]) != n:
            raise _shape_error(f"series {s['key']}: {len(s['values'])} values, {axis} has {n}")


_CELL_TYPES: dict[str, tuple[type, ...]] = {
    "number": (int, float),
    "string": (str,),
    "timestamp": (str,),
    "date": (str,),
    "boolean": (bool,),
}


def check_shapes(kind: str, payload: dict[str, Any]) -> None:
    """C-4: cross-field rules JSON Schema can't express."""
    if kind == "curve":
        _check_series(payload["series"], len(payload["x"]["values"]), "x")
    elif kind == "timeseries":
        t = payload["t"]
        if any(b <= a for a, b in zip(t, t[1:], strict=False)):
            raise _shape_error("t must be strictly increasing")
        _check_series(payload["series"], len(t), "t")
    elif kind == "surface":
        nx, ny = len(payload["x"]["values"]), len(payload["y"]["values"])
        rows = payload["z"]["values"]
        if len(rows) != ny or any(len(r) != nx for r in rows):
            raise _shape_error(f"z must be {ny} rows of {nx} values (len(y) x len(x))")
    elif kind == "table":
        columns = payload["columns"]
        keys = [c["key"] for c in columns]
        if len(keys) != len(set(keys)):
            raise _shape_error("column keys must be unique")
        for i, row in enumerate(payload["rows"]):
            if len(row) != len(columns):
                raise _shape_error(f"row {i}: {len(row)} cells, {len(columns)} columns")
            for col, cell in zip(columns, row, strict=True):
                if cell is None:
                    continue
                ok = isinstance(cell, _CELL_TYPES[col["type"]])
                if col["type"] == "number" and isinstance(cell, bool):
                    ok = False
                if not ok:
                    raise _shape_error(f"row {i}, column {col['key']}: expected {col['type']}")
    elif kind == "scalar_set":
        keys = [item["key"] for item in payload["items"]]
        if len(keys) != len(set(keys)):
            raise _shape_error("item keys must be unique")


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
