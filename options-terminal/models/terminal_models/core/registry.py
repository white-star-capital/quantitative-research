"""Production switchboard: models/registry.yaml (MODELS.md §4)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from terminal_models.core.manifest import IDENTIFIER, SEMVER, Manifest, load_manifest
from terminal_models.core.slots import SlotDefinition


class RegistryError(Exception):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


class ImplementationRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    implementation: str = Field(pattern=IDENTIFIER)
    version: str = Field(pattern=SEMVER)


class SlotEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    production: ImplementationRef  # M-12: exactly one
    shadow: list[ImplementationRef] = Field(default_factory=list)
    scopes: list[dict[str, str]] = Field(min_length=1)


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slots: dict[str, SlotEntry] = Field(default_factory=dict)


def load_registry(path: Path) -> Registry:
    with path.open() as f:
        return Registry.model_validate(yaml.safe_load(f) or {})


def validate_registry(
    registry: Registry, slots: dict[str, SlotDefinition], library: Path
) -> dict[str, Manifest]:
    """M-13. Returns the manifests of every referenced implementation; raises with all problems."""
    problems: list[str] = []
    manifests: dict[str, Manifest] = {}

    for slot_id, entry in registry.slots.items():
        slot = slots.get(slot_id)
        if slot is None:
            problems.append(f"{slot_id}: no slot definition in contracts/slots/")
            continue

        refs = [("production", entry.production)] + [("shadow", s) for s in entry.shadow]
        seen = [ref.implementation for _, ref in refs]
        if len(seen) != len(set(seen)):
            problems.append(f"{slot_id}: an implementation is listed more than once")

        for role, ref in refs:
            manifest_path = library / ref.implementation / "manifest.yaml"
            if not manifest_path.exists():
                problems.append(f"{slot_id}: {role} '{ref.implementation}' not found in library/")
                continue
            try:
                manifest = load_manifest(manifest_path)
            except Exception as exc:  # report every problem, not just the first
                problems.append(f"{slot_id}: {role} '{ref.implementation}' manifest invalid: {exc}")
                continue
            if manifest.implementation_id != ref.implementation:
                problems.append(
                    f"{slot_id}: {ref.implementation}/manifest.yaml declares "
                    f"implementation_id '{manifest.implementation_id}'"
                )
            if manifest.version != ref.version:
                problems.append(
                    f"{slot_id}: {role} '{ref.implementation}' pinned at {ref.version}, "
                    f"manifest is {manifest.version}"
                )
            if manifest.fills_slot != slot_id:
                problems.append(
                    f"{slot_id}: '{ref.implementation}' fills slot '{manifest.fills_slot}'"
                )
            manifests[ref.implementation] = manifest

        dims = set(slot.scope.dimensions)
        for scope in entry.scopes:
            if set(scope) != dims:
                problems.append(
                    f"{slot_id}: scope {scope} must have exactly the keys {sorted(dims)}"
                )
                continue
            for key, value in scope.items():
                allowed = slot.scope.allowed.get(key)
                if allowed is not None and value not in allowed:
                    problems.append(f"{slot_id}: scope {key}={value} not in allowed {allowed}")

    if problems:
        raise RegistryError(problems)
    return manifests
