"""Run envelope (CONTRACTS.md §2). contracts/schemas/envelope/v1.json is the source of truth;
validation.validate_envelope checks against it, this type is for building envelopes in Python."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from terminal_models.core.slots import PayloadKind


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    message: str


class InputSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    snapshot_at: str


class Inputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sources: list[InputSource]
    hash: str


class Code(BaseModel):
    model_config = ConfigDict(extra="forbid")
    git_sha: str


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    envelope_version: Literal["1"] = "1"
    run_id: str
    slot_id: str
    implementation_id: str
    implementation_version: str
    scope: dict[str, str]
    status: Literal["success", "failed"]
    as_of: str
    started_at: str
    finished_at: str
    inputs: Inputs
    code: Code
    warnings: list[Message]
    payload_kind: PayloadKind
    payload_version: str
    payload: dict[str, Any] | None
    error: Message | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """`payload` stays as null on failed runs; `error` is omitted on successful ones."""
        return self.model_dump(mode="json", exclude={"error"} if self.error is None else set())
