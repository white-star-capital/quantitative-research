"""Snapshot store (MODELS M-9, guide D-6): every run's inputs, keyed by input hash, so any run can
be replayed exactly. Local directory for now (SNAPSHOT_DIR); object storage is an M4 concern."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from terminal_models.core.base import InputData
from terminal_models.core.paths import MODELS_DIR


def _canonical(inputs: dict[str, InputData]) -> list[dict]:
    return [
        {"name": d.name, "snapshot_at": d.snapshot_at.isoformat(), "data": d.data}
        for d in sorted(inputs.values(), key=lambda d: d.name)
    ]


def input_hash(inputs: dict[str, InputData]) -> str:
    """sha256 over canonical JSON of (name, snapshot_at, data); raw bodies are not hashed."""
    blob = json.dumps(_canonical(inputs), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


class SnapshotStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(os.environ.get("SNAPSHOT_DIR", MODELS_DIR / ".snapshots"))

    def _path(self, digest: str) -> Path:
        return self.root / f"{digest.removeprefix('sha256:')}.json"

    def save(self, inputs: dict[str, InputData]) -> str:
        digest = input_hash(inputs)
        path = self._path(digest)
        if not path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            doc = {
                "hash": digest,
                "inputs": _canonical(inputs),
                "raw": {d.name: d.raw for d in inputs.values()},
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, sort_keys=True, allow_nan=False))
            tmp.replace(path)
        return digest

    def load(self, digest: str) -> dict[str, InputData]:
        path = self._path(digest)
        if not path.exists():
            raise FileNotFoundError(f"no snapshot for {digest} in {self.root}")
        doc = json.loads(path.read_text())
        raw = doc.get("raw", {})
        inputs = {
            i["name"]: InputData(
                i["name"],
                datetime.fromisoformat(i["snapshot_at"]),
                i["data"],
                raw=raw.get(i["name"]),
            )
            for i in doc["inputs"]
        }
        if input_hash(inputs) != digest:
            raise ValueError(f"snapshot {digest} does not hash to its name")
        return inputs
