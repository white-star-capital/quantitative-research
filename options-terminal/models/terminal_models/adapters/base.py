"""Data adapter interface (MODELS.md §3.2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from terminal_models.core.base import InputData, InputSpec


class AdapterError(Exception):
    """Input could not be fetched. The run fails; there is no fallback value (M-9a)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DataAdapter(ABC):
    name: str

    @abstractmethod
    def fetch(self, spec: InputSpec, as_of: datetime) -> InputData:
        """Return the input with the source's own snapshot time. Raise AdapterError on failure."""
