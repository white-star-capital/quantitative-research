"""Recorded-snapshot adapter for tests and replay (MODELS.md §3.2)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from terminal_models.adapters.base import AdapterError, DataAdapter
from terminal_models.core.base import InputData, InputSpec


class FileAdapter(DataAdapter):
    """Reads `<root>/<spec.params["path"]>`: JSON shaped `{"snapshot_at": "...Z", "data": ...}`.

    `snapshot_at` comes from the file, never the clock, so staleness checks see the real age.
    """

    name = "file"

    def __init__(self, root: Path) -> None:
        self.root = root

    def fetch(self, spec: InputSpec, as_of: datetime) -> InputData:
        rel = spec.params.get("path")
        if not rel:
            raise AdapterError("bad_input_spec", f"{spec.name}: params.path is required")
        path = self.root / rel
        try:
            with path.open() as f:
                recorded = json.load(f)
            snapshot_at = datetime.fromisoformat(recorded["snapshot_at"])
            data = recorded["data"]
        except FileNotFoundError as exc:
            raise AdapterError("input_missing", f"{spec.name}: {path} not found") from exc
        except (KeyError, ValueError, TypeError) as exc:
            raise AdapterError("input_malformed", f"{spec.name}: {path}: {exc}") from exc
        if snapshot_at.tzinfo is None:
            raise AdapterError("input_malformed", f"{spec.name}: snapshot_at must be UTC with Z")
        return InputData(name=spec.name, snapshot_at=snapshot_at, data=data)
