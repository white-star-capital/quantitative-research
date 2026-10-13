"""Implementation manifest (MODELS.md §3.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

IDENTIFIER = r"^[a-z][a-z0-9_]{2,63}$"
SEMVER = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"


class Bound(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min: float | None = None
    max: float | None = None


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    implementation_id: str = Field(pattern=IDENTIFIER)
    version: str = Field(pattern=SEMVER)
    fills_slot: str = Field(pattern=IDENTIFIER)
    owner: str
    description: str
    schedule: str
    timeout_seconds: int = Field(gt=0)
    max_input_age_seconds: int = Field(gt=0)
    max_input_skew_seconds: int | None = Field(default=None, gt=0)
    params: dict[str, Any] = Field(default_factory=dict)
    bounds: dict[str, Bound] = Field(default_factory=dict)
    sanity_checks: list[str] = Field(default_factory=list)


def load_manifest(path: Path) -> Manifest:
    with path.open() as f:
        return Manifest.model_validate(yaml.safe_load(f))
