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
from terminal_models.core.payloads import PayloadError, get_section, records, scalars

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


# ---- manifest bounds (hard) ----------------------------------------------------------------


def _bounded_values(kind: str, payload: dict[str, Any], where: str):
    """Yield (path, key, value) for every number that carries a key."""
    if kind == "composite":
        for sec in payload["sections"]:
            yield from _bounded_values(sec["payload_kind"], sec["payload"], f"{where}{sec['key']}/")
    elif kind == "scalar_set":
        for it in payload["items"]:
            yield f"{where}{it['key']}", it["key"], it["value"]
    elif kind in ("curve", "timeseries"):
        for sr in payload["series"]:
            for v in sr["values"]:
                yield f"{where}{sr['key']}", sr["key"], v
    elif kind == "surface":
        for row in payload["z"]["values"]:
            for v in row:
                yield f"{where}z", "z", v
    elif kind == "table":
        keys = [c["key"] for c in payload["columns"]]
        for row in payload["rows"]:
            for key, v in zip(keys, row, strict=True):
                yield f"{where}{key}", key, v


def check_bounds(kind: str, payload: dict[str, Any], bounds: dict[str, Any]) -> None:
    """Manifest `bounds` are keyed by value key (column, item, series, or `z` for surfaces);
    an `abs_` prefix bounds the absolute value. Any violation fails the run."""
    if not bounds:
        return
    for path, key, value in _bounded_values(kind, payload, ""):
        if not isinstance(value, int | float) or isinstance(value, bool):
            continue
        for name, x in ((key, value), (f"abs_{key}", abs(value))):
            b = bounds.get(name)
            if b is None:
                continue
            lo, hi = getattr(b, "min", None), getattr(b, "max", None)
            if (lo is not None and x < lo) or (hi is not None and x > hi):
                raise ValidationFailure(
                    "out_of_bounds", f"{path}: {value!r} outside {name} [{lo}, {hi}]"
                )


# ---- sanity checks (MODELS §6, guide §5.4) ------------------------------------------------

Warnings = list[dict[str, str]]
SanityCheck = Callable[[dict[str, Any], dict[str, Any]], Warnings]

MARK_TOLERANCE = 0.01
MARK_HARD_FAIL_SHARE = 0.20
PARITY_TOLERANCE = 1e-6
SPOT_PERP_TOLERANCE = 0.02


def _need(payload: dict[str, Any], key: str) -> dict[str, Any]:
    try:
        return get_section(payload, key)
    except PayloadError as exc:
        raise ValidationFailure(
            "sanity_check_inapplicable", f"payload has no '{key}' section"
        ) from exc


def mark_reproduction(payload: dict[str, Any], params: dict[str, Any]) -> Warnings:
    """With r = 0, Black-76 at the venue's mark IV must reproduce the venue's mark price.
    Misses beyond 20% of nodes mean a units/conversion bug: hard failure."""
    if params.get("usd_rate", 0.0) != 0.0:
        return []
    rows = [
        row
        for row in records(_need(payload, "chain"))
        if row["is_node"] and row["venue_mark_usd"] is not None
    ]
    misses = [
        (
            abs(row["fair_usd"] - row["venue_mark_usd"]) / row["venue_mark_usd"],
            row["instrument_name"],
        )
        for row in rows
    ]
    misses = [m for m in misses if m[0] >= MARK_TOLERANCE]
    if not misses:
        return []
    worst = max(misses)
    message = (
        f"{len(misses)} of {len(rows)} nodes off venue mark by >= 1% "
        f"(worst {worst[1]}: {worst[0]:.1%})"
    )
    if len(misses) > MARK_HARD_FAIL_SHARE * len(rows):
        raise ValidationFailure("mark_reproduction", message)
    return [{"code": "mark_reproduction", "message": message}]


def put_call_parity(payload: dict[str, Any], params: dict[str, Any]) -> Warnings:
    """call − put = DF·(F − K) for every listed strike pair; holds by construction (hard)."""
    r = scalars(_need(payload, "context")).get("usd_rate", 0.0)
    expiries = {e["expiry_ts"]: e for e in records(_need(payload, "expiries"))}
    pairs: dict[tuple[str, float], dict[str, float]] = {}
    for row in records(_need(payload, "chain")):
        pairs.setdefault((row["expiry_ts"], row["strike"]), {})[row["option_type"]] = row[
            "fair_usd"
        ]
    for (expiry_ts, strike), legs in pairs.items():
        if "call" not in legs or "put" not in legs:
            continue
        e = expiries[expiry_ts]
        df = math.exp(-r * e["dte"] / 365)
        residual = legs["call"] - legs["put"] - df * (e["forward"] - strike)
        if abs(residual) > PARITY_TOLERANCE * e["forward"]:
            raise ValidationFailure(
                "put_call_parity", f"{expiry_ts} K={strike}: residual {residual:.3g}"
            )
    return []


def calendar_arbitrage(payload: dict[str, Any], params: dict[str, Any]) -> Warnings:
    """Total variance must not fall with tenor at fixed moneyness (warning; no clamping)."""
    surf = _need(payload, "surface")
    tenors = surf["y"]["values"]
    found = []
    for i in range(len(tenors) - 1):
        lower, upper = surf["z"]["values"][i], surf["z"]["values"][i + 1]
        for k, a, b in zip(surf["x"]["values"], lower, upper, strict=True):
            if a is None or b is None:
                continue
            drop = a * a * tenors[i] / 365 - b * b * tenors[i + 1] / 365
            if drop > 1e-12:
                found.append((drop, k, tenors[i], tenors[i + 1]))
    if not found:
        return []
    drop, k, t_a, t_b = max(found)
    return [
        {
            "code": "calendar_arbitrage",
            "message": f"total variance falls at {len(found)} grid points "
            f"(worst k={k:+.3f} between {t_a:.1f}d and {t_b:.1f}d)",
        }
    ]


def spot_vs_perp(payload: dict[str, Any], params: dict[str, Any]) -> Warnings:
    ctx = scalars(_need(payload, "context"))
    spot, perp = ctx.get("spot"), ctx.get("perp_last")
    if not spot or not perp:
        return []
    gap = abs(spot / perp - 1)
    if gap < SPOT_PERP_TOLERANCE:
        return []
    return [{"code": "spot_vs_perp", "message": f"index and perp differ by {gap:.2%}"}]


IMPLIED_FORWARD_TOLERANCE = 0.01


def parity_band_ordering(payload: dict[str, Any], params: dict[str, Any]) -> Warnings:
    """F_impl_bid <= F_impl_ask on every strike (parity guide §5). A crossed band means the
    book or the inverse/linear conversion is wrong: bad data, hard failure."""
    crossed = [
        row
        for row in records(_need(payload, "bands"))
        if row["F_impl_bid"] is not None
        and row["F_impl_ask"] is not None
        and row["F_impl_bid"] > row["F_impl_ask"] * (1 + PARITY_TOLERANCE)
    ]
    if crossed:
        r = crossed[0]
        raise ValidationFailure(
            "parity_band_ordering",
            f"{len(crossed)} crossed parity bands (first {r['expiry_ts']} K={r['strike']}: "
            f"{r['F_impl_bid']} > {r['F_impl_ask']})",
        )
    return []


def implied_vs_venue_forward(payload: dict[str, Any], params: dict[str, Any]) -> Warnings:
    """Options-implied forward within 1% of the venue's underlying_price per expiry, else warn."""
    off = []
    for row in records(_need(payload, "expiries")):
        implied, venue = row["F_impl"], row["venue_forward"]
        if implied is None or not venue:
            continue
        gap = abs(implied / venue - 1)
        if gap >= IMPLIED_FORWARD_TOLERANCE:
            off.append(f"{row['expiry_ts']} {gap:.2%}")
    if not off:
        return []
    return [
        {
            "code": "implied_vs_venue_forward",
            "message": f"implied forward >= 1% from venue forward: {', '.join(off)}",
        }
    ]


def _pending(name: str) -> SanityCheck:
    def check(payload: dict[str, Any], params: dict[str, Any]) -> Warnings:
        raise NotImplementedError(f"sanity check '{name}' is not implemented yet")

    return check


# MODELS.md §6. Implementations opt in by name via manifest `sanity_checks`.
SANITY_CHECKS: dict[str, SanityCheck] = {
    "calendar_arbitrage": calendar_arbitrage,
    "put_call_parity": put_call_parity,
    "mark_reproduction": mark_reproduction,
    "spot_vs_perp": spot_vs_perp,
    "parity_band_ordering": parity_band_ordering,
    "implied_vs_venue_forward": implied_vs_venue_forward,
    "butterfly_arbitrage": _pending("butterfly_arbitrage"),
    "monotonic_tenor": _pending("monotonic_tenor"),
    "greeks_signs": _pending("greeks_signs"),
}
