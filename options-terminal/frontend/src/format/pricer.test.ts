import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  fmtDelta, fmtDte, fmtExpiry, fmtGamma, fmtGreekUsd, fmtIv, fmtPct, fmtPremium, fmtSpot,
  fmtUsdTotal, fmtVp, fmtVpPoints,
} from "./pricer.ts";
import { MISSING, formatCell, formatValue } from "./units.ts";

describe("guide §7.8 formatting table", () => {
  it("IV, vol points", () => {
    assert.equal(fmtIv(0.552), "55.2%");
    assert.equal(fmtVp(0.052), "+5.2 vp");
    assert.equal(fmtVp(-0.031), "−3.1 vp");
    assert.equal(fmtVpPoints(-0.1), "−0.1 vp");
  });

  it("premiums to 4 significant figures, 2 dp below $100", () => {
    assert.equal(fmtPremium(1234.4), "$1,234");
    assert.equal(fmtPremium(61.4), "$61.40");
    assert.equal(fmtPremium(0.8523), "$0.8523");
    assert.equal(fmtPremium(-5.5, true), "−$5.500"); // 4 significant figures
  });

  it("totals", () => {
    assert.equal(fmtUsdTotal(6059), "$6,059");
    assert.equal(fmtUsdTotal(-81), "−$81.00");
    assert.equal(fmtUsdTotal(812.5, true), "+$812.50");
  });

  it("spot and strike precision per asset", () => {
    assert.equal(fmtSpot("BTC", 62345.6), "62,346");
    assert.equal(fmtSpot("ETH", 2473.59), "2,473.6");
    assert.equal(fmtSpot("SOL", 145.678), "145.68");
    assert.equal(fmtSpot("HYPE", 38.1234), "38.123");
  });

  it("greeks", () => {
    assert.equal(fmtDelta(-0.33), "−0.33");
    assert.equal(fmtDelta(0.33), "+0.33");
    assert.equal(fmtGamma(0.0221), "0.022");
    assert.equal(fmtGreekUsd(2.6), "$2.60");
    assert.equal(fmtGreekUsd(-1.4), "−$1.40");
    assert.equal(fmtGreekUsd(0.0083), "$0.0083");
  });

  it("percentages, expiry, DTE", () => {
    assert.equal(fmtPct(0.3), "30%");
    assert.equal(fmtPct(0.987), "98.7%");
    assert.equal(fmtExpiry("2026-11-06T08:00:00Z"), "06 NOV 26 08:00 UTC");
    assert.equal(fmtDte(24.12), "24 d");
    assert.equal(fmtDte(2.4), "2.4 d");
  });

  it("unknown is a dash, never zero", () => {
    for (const f of [fmtIv, fmtVp, fmtPremium, fmtUsdTotal, fmtDelta, fmtGreekUsd, fmtPct, fmtDte]) {
      assert.equal(f(null), MISSING);
      assert.equal(f(Number.NaN), MISSING);
    }
  });
});

describe("unit vocabulary additions", () => {
  it("formats the new quantities", () => {
    assert.equal(formatValue(0.052, { quantity: "vol_spread", scale: "decimal" }), "+5.2 vp");
    assert.equal(formatValue(0.301, { quantity: "probability", scale: "decimal" }), "30.1%");
    assert.equal(formatValue(0.375, { quantity: "yield_annualized", scale: "decimal" }), "37.5%");
  });

  it("formats cells by type", () => {
    assert.equal(formatCell(true, "boolean"), "YES");
    assert.equal(formatCell(null, "number"), MISSING);
    assert.equal(formatCell("2026-11-06T08:00:00Z", "timestamp"), "6 NOV, 08:00 UTC");
  });
});
