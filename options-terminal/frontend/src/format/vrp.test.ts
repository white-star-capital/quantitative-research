import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { fmtDay, fmtShare, fmtUnixDay, fmtZ } from "./vrp.ts";

describe("vrp formatting (guide §12)", () => {
  it("z-scores: signed, 2 dp, sigma", () => {
    assert.equal(fmtZ(1.8), "+1.80σ");
    assert.equal(fmtZ(-2.004), "−2.00σ");
    assert.equal(fmtZ(0), "0.00σ");
    assert.equal(fmtZ(null), "—");
  });
  it("shares and days", () => {
    assert.equal(fmtShare(0.354), "35%");
    assert.equal(fmtShare(undefined), "—");
    assert.equal(fmtUnixDay(Date.parse("2026-10-28T00:00:00Z") / 1000), "28 OCT, 00:00 UTC");
    assert.equal(fmtUnixDay(null), "—");
    assert.equal(fmtDay(Date.parse("2026-10-28T00:00:00Z")), "28 OCT 26");
    assert.equal(fmtDay(Number.NaN), "—");
  });
});
