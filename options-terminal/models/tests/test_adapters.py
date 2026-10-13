import json
from datetime import UTC, datetime

import httpx
import pytest

from terminal_models.adapters.base import AdapterError
from terminal_models.adapters.file import FileAdapter
from terminal_models.adapters.http import HttpSource
from terminal_models.core.base import InputSpec

AS_OF = datetime(2026, 10, 13, 17, tzinfo=UTC)


def test_file_adapter_uses_recorded_snapshot_time(tmp_path):
    (tmp_path / "snap.json").write_text(
        json.dumps({"snapshot_at": "2026-10-13T16:58:00Z", "data": {"rows": []}})
    )
    got = FileAdapter(tmp_path).fetch(InputSpec("chain", "file", {"path": "snap.json"}), AS_OF)
    assert got.snapshot_at == datetime(2026, 10, 13, 16, 58, tzinfo=UTC)
    assert got.data == {"rows": []}


@pytest.mark.parametrize(
    "content, code",
    [
        (None, "input_missing"),
        ('{"data": {}}', "input_malformed"),
        ('{"snapshot_at": "2026-10-13T16:58:00", "data": {}}', "input_malformed"),
    ],
)
def test_file_adapter_fails_rather_than_defaults(tmp_path, content, code):
    if content is not None:
        (tmp_path / "snap.json").write_text(content)
    with pytest.raises(AdapterError) as exc:
        FileAdapter(tmp_path).fetch(InputSpec("chain", "file", {"path": "snap.json"}), AS_OF)
    assert exc.value.code == code


def http_source(handler, attempts=3):
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://example.test")
    return HttpSource(
        "https://example.test", max_attempts=attempts, backoff_seconds=0, client=client
    )


def test_http_retries_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, json={"ok": True})

    assert http_source(handler).request_json("GET", "/x") == {"ok": True}
    assert len(calls) == 3


def test_http_gives_up_with_adapter_error():
    with pytest.raises(AdapterError) as exc:
        http_source(lambda r: httpx.Response(502)).request_json("GET", "/x")
    assert exc.value.code == "source_unavailable"


def test_http_client_error_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404)

    with pytest.raises(AdapterError) as exc:
        http_source(handler).request_json("GET", "/x")
    assert exc.value.code == "source_http_error"
    assert len(calls) == 1
