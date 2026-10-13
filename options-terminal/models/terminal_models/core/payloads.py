"""Builders and readers for payload kinds (CONTRACTS.md §3).

Payloads are plain JSON dicts validated against contracts/schemas (C-1); these helpers only keep
construction and lookup consistent between models, sanity checks and evaluation.
"""

from __future__ import annotations

from typing import Any

Json = dict[str, Any]


class PayloadError(KeyError):
    pass


# ---- builders ---------------------------------------------------------------------------


def unit(quantity: str, scale: str = "absolute", convention: str | None = None) -> Json:
    u: Json = {"quantity": quantity, "scale": scale}
    if convention:
        u["convention"] = convention
    return u


def item(key: str, label: str, value: Any, u: Json, flags: tuple[str, ...] = ()) -> Json:
    out: Json = {"key": key, "label": label, "value": value, "unit": u}
    if flags:
        out["flags"] = list(flags)
    return out


def series(key: str, label: str, u: Json, values: list) -> Json:
    return {"key": key, "label": label, "unit": u, "values": values}


def axis(label: str, u: Json, values: list[float]) -> Json:
    return {"label": label, "unit": u, "values": values}


def column(key: str, label: str, type_: str, u: Json | None = None) -> Json:
    col: Json = {"key": key, "label": label, "type": type_}
    if u is not None:
        col["unit"] = u
    return col


def section(key: str, label: str, kind: str, payload: Json, version: str = "1") -> Json:
    return {
        "key": key,
        "label": label,
        "payload_kind": kind,
        "payload_version": version,
        "payload": payload,
    }


# ---- readers ----------------------------------------------------------------------------


def get_section(composite: Json, key: str) -> Json:
    for s in composite["sections"]:
        if s["key"] == key:
            return s["payload"]
    raise PayloadError(f"no '{key}' section")


def records(table: Json) -> list[Json]:
    keys = [c["key"] for c in table["columns"]]
    return [dict(zip(keys, row, strict=True)) for row in table["rows"]]


def scalars(scalar_set: Json) -> Json:
    return {i["key"]: i["value"] for i in scalar_set["items"]}
