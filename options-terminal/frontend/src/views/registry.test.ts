import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type { SlotSummary } from "../api/types.ts";
import { bespokeFor, registerView } from "./registry.ts";

const slot = (view?: string): SlotSummary => ({
  slot_id: "x_slot", title: "", description: "", payload_kind: "composite", payload_version: "1",
  scope: { dimensions: [] }, max_age_seconds: 900, production_impl: null, production_version: null,
  ...(view ? { display: { default_view: view } } : {}),
});

describe("bespoke view registry", () => {
  it("falls back to generic rendering for unknown or missing views", () => {
    const View = () => null;
    registerView("pricer_view", View);
    assert.equal(bespokeFor(slot("pricer_view")), View);
    assert.equal(bespokeFor(slot("heatmap")), null);
    assert.equal(bespokeFor(slot()), null);
  });
});
