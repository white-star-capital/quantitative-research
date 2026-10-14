"""Replay a Deribit recording (tools/record_deribit.py) through the production adapter code.

`RecordedClient` answers `call(method, **params)` from recorded response bodies, so
`DeribitAdapter(RecordedClient(dir))` produces InputData through the same parse/normalize/perp
discovery path as a live fetch. Bodies are checked against the manifest's sha256 on load: a
recording that has been edited is refused, never used.

Layout:
    manifest.json
    calls/<seq>_<method>.meta.json   request, receive time, body sha256
    calls/<seq>_<method>.body.gz     response body bytes exactly as received
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from terminal_models.adapters.base import AdapterError
from terminal_models.adapters.deribit.client import DeribitClient, Response, parse

# Candle/DVOL windows depend on the replay's as_of; the recording holds one window per series.
TIME_PARAMS = {"start_timestamp", "end_timestamp"}


def request_key(method: str, params: dict[str, Any]) -> str:
    kept = {k: str(v) for k, v in params.items() if k not in TIME_PARAMS}
    return json.dumps([method, kept], sort_keys=True)


class RecordingError(Exception):
    pass


def load_bodies(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise RecordingError(f"{root}: no manifest.json (incomplete or not a recording)")
    manifest = json.loads(manifest_path.read_text())
    bodies: dict[str, str] = {}
    for entry in manifest["calls"]:
        meta = json.loads((root / entry["meta"]).read_text())
        raw = gzip.decompress((root / entry["body"]).read_bytes())
        if hashlib.sha256(raw).hexdigest() != entry["body_sha256"]:
            raise RecordingError(f"{entry['body']}: sha256 mismatch; the recording was modified")
        bodies[request_key(meta["method"], meta["params"])] = raw.decode()
    return manifest, bodies


class RecordedClient(DeribitClient):
    def __init__(self, root: Path) -> None:
        super().__init__(source=None)
        self.root = root
        self.manifest, self._bodies = load_bodies(root)

    def call(self, method: str, ttl_s: float = 0, **params: Any) -> Response:
        body = self._bodies.get(request_key(method, params))
        if body is None:
            raise AdapterError("venue_unavailable", f"{method} {params} is not in the recording")
        return parse(method, body)
