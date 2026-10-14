import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { fmtBp, fmtCoinEdge, fmtNative, fmtRateAnn } from "./parity.ts";

describe("parity formatting (guide §7)", () => {
  it("rates: % annualized, 2 dp", () => {
    assert.equal(fmtRateAnn(0.0685), "6.85% ann.");
    assert.equal(fmtRateAnn(-0.0012), "−0.12% ann.");
    assert.equal(fmtRateAnn(null), "—");
  });
  it("dislocations: signed bp, 1 dp", () => {
    assert.equal(fmtBp(12.44), "+12.4 bp");
    assert.equal(fmtBp(-3.06), "−3.1 bp");
    assert.equal(fmtBp(undefined), "—");
  });
  it("coin edges carry a USD equivalent", () => {
    assert.equal(fmtCoinEdge(0.000412, "BTC", 100_000), "+0.000412 BTC ≈ +$41.20");
    assert.equal(fmtCoinEdge(0.000412, "BTC", null), "+0.000412 BTC");
  });
  it("native leg prices", () => {
    assert.equal(fmtNative(0.0512, "coin", "BTC"), "0.051200 BTC");
    assert.equal(fmtNative(61010, "usd", "BTC"), "$61,010");
  });
});
