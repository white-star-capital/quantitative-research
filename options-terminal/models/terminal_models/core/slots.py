"""Slot definitions from contracts/slots/*.yaml (CONTRACTS.md §4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from terminal_models.core.manifest import IDENTIFIER
from terminal_models.core.paths import contracts_dir

PayloadKind = Literal["scalar_set", "curve", "surface", "timeseries", "table", "composite"]


class SlotScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dimensions: list[str]
    allowed: dict[str, list[str]] = Field(default_factory=dict)


class Evaluation(BaseModel):
    """Optional evaluation endpoint (README §3.1, API §3.2). Extra keys are evaluator settings."""

    model_config = ConfigDict(extra="allow")
    evaluator: str = Field(pattern=IDENTIFIER)


class SlotDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_id: str = Field(pattern=IDENTIFIER)
    title: str
    description: str
    payload_kind: PayloadKind
    payload_version: str
    scope: SlotScope
    max_age_seconds: int = Field(gt=0)
    conventions: str = ""
    display: dict[str, Any] = Field(default_factory=dict)
    evaluation: Evaluation | None = None


def load_slots(root: Path | None = None) -> dict[str, SlotDefinition]:
    slots_dir = (root or contracts_dir()) / "slots"
    slots: dict[str, SlotDefinition] = {}
    for path in sorted(slots_dir.glob("*.yaml")):
        with path.open() as f:
            slot = SlotDefinition.model_validate(yaml.safe_load(f))
        if slot.slot_id != path.stem:
            raise ValueError(f"{path.name}: slot_id '{slot.slot_id}' does not match file name")
        slots[slot.slot_id] = slot
    return slots
