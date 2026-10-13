"""Model interface (MODELS.md §3.1)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from terminal_models.core.manifest import Manifest

Scope = dict[str, str]
Payload = dict[str, Any]


class ModelError(Exception):
    """Expected model failure (M-6). Recorded as a failed run with this code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class InputSpec:
    """One input a model needs. `source` names the adapter; `params` are adapter-specific."""

    name: str
    source: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InputData:
    """Fetched input. `snapshot_at` is the source timestamp, never fetch time (M-8)."""

    name: str
    snapshot_at: datetime
    data: Any


@dataclass(frozen=True)
class RunContext:
    as_of: datetime
    scope: Scope
    inputs: dict[str, InputData]
    params: dict[str, Any]
    logger: logging.Logger


class Model(ABC):
    manifest: Manifest

    @abstractmethod
    def required_inputs(self, scope: Scope) -> list[InputSpec]:
        """Declare which data the run needs. The runner fetches it via adapters."""

    @abstractmethod
    def run(self, ctx: RunContext) -> Payload:
        """Pure computation: no I/O (M-5), deterministic (M-4)."""
