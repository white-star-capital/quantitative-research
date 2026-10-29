import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  MISSING,
  formatAge,
  formatCompactUsd,
  formatNumber,
  formatUtc,
  formatValue,
  unitLabel,
} from "./units.ts";

describe("formatValue", () => {
  it("renders decimal implied vol as percent", () => {
    assert.equal(formatValue(0.5523, { quantity: "implied_vol", scale: "decimal" }), "55.23%");
  });

  it("renders null, undefined and non-finite as a dash, never zero", () => {
    const unit = { quantity: "implied_vol", scale: "decimal" } as const;
    for (const v of [null, undefined, Number.NaN, Number.POSITIVE_INFINITY]) {
      assert.equal(formatValue(v, unit), MISSING);
    }
  });

  it("keeps a real zero as zero", () => {
    assert.equal(formatValue(0, { quantity: "delta", scale: "absolute" }), "0.0000");
  });

  it("uses the real minus sign", () => {
    assert.equal(formatValue(-0.25, { quantity: "delta", scale: "absolute" }), "−0.2500");
  });

  it("formats usd with sign before the dollar", () => {
    const unit = { quantity: "price_usd", scale: "absolute" } as const;
    assert.equal(formatValue(1234.5, unit), "$1,234.50");
    assert.equal(formatValue(-1234.5, unit), "−$1,234.50");
    assert.equal(formatValue(12, unit, { signed: true }), "+$12.00");
  });

  it("formats percent and bps scales", () => {
    assert.equal(formatValue(55.2, { quantity: "ratio", scale: "percent" }), "55.20%");
    assert.equal(formatValue(-12.34, { quantity: "rate", scale: "bps" }), "−12.3 bps");
  });

  it("formats z-scores signed, in standard deviations", () => {
    const unit = { quantity: "zscore", scale: "absolute" } as const;
    assert.equal(formatValue(1.8, unit), "+1.80σ");
    assert.equal(formatValue(-0.456, unit), "−0.46σ");
    assert.equal(formatValue(null, unit), MISSING);
  });

  it("formats unix-second timestamps as UTC", () => {
    const unit = { quantity: "timestamp", scale: "absolute", convention: "unix_seconds" } as const;
    assert.equal(formatValue(Date.parse("2026-10-28T00:00:00Z") / 1000, unit), "28 OCT, 00:00 UTC");
  });

  it("does not print negative zero", () => {
    assert.equal(formatNumber(-0.0001, 2), "0.00");
  });
});

describe("labels and times", () => {
  it("adds the greek convention", () => {
    assert.equal(
      unitLabel("Vega", { quantity: "vega", scale: "absolute", convention: "per_1vol_point" }),
      "Vega (per 1 vol pt)",
    );
    assert.equal(unitLabel("Delta", { quantity: "delta", scale: "absolute" }), "Delta");
  });

  it("formats UTC timestamps", () => {
    assert.equal(formatUtc("2026-10-12T15:17:42Z"), "12 OCT, 15:17 UTC");
    assert.equal(formatUtc("not a date"), MISSING);
  });

  it("formats ages", () => {
    assert.equal(formatAge(42), "42s ago");
    assert.equal(formatAge(6 * 60 + 5), "6m ago");
    assert.equal(formatAge(125 * 60), "2h 5m ago");
    assert.equal(formatAge(-1), MISSING);
  });

  it("compacts usd for axes", () => {
    assert.equal(formatCompactUsd(12_500), "$12.5K");
    assert.equal(formatCompactUsd(-1_200_000), "−$1.2M");
  });
});
