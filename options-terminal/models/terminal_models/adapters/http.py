"""Shared HTTP plumbing for live adapters: timeouts, retries with backoff, rate limits (M-7)."""

from __future__ import annotations

import json
import random
import time
from typing import Any

import httpx

from terminal_models.adapters.base import AdapterError

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class HttpSource:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        max_attempts: int = 4,
        backoff_seconds: float = 1.0,
        jitter: float = 0.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.jitter = jitter
        self.client = client or httpx.Client(base_url=base_url, timeout=timeout_seconds)

    def request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        text = self.request_text(method, path, **kwargs)
        try:
            return json.loads(text)
        except ValueError as exc:
            raise AdapterError("source_bad_json", f"{method} {path}: {exc}") from exc

    def request_text(self, method: str, path: str, **kwargs: Any) -> str:
        """The response body exactly as received (recording needs the original bytes)."""
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code not in RETRYABLE_STATUS:
                    if response.is_error:
                        raise AdapterError(
                            "source_http_error", f"{method} {path} -> {response.status_code}"
                        )
                    return response.text
                last_error = f"HTTP {response.status_code}"
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit() and attempt < self.max_attempts:
                    time.sleep(int(retry_after))
                    continue
            if attempt < self.max_attempts:
                delay = self.backoff_seconds * 2 ** (attempt - 1)
                time.sleep(delay + random.uniform(0, delay * self.jitter))
        raise AdapterError(
            "source_unavailable",
            f"{method} {path} failed after {self.max_attempts} attempts: {last_error}",
        )
