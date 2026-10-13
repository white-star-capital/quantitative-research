import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  DEFAULT_THRESHOLDS, band, convertPrice, defaultExpiry, evaluateParams, loadPref, payoff,
  runSelection, savePref, strikeRows, toCsv, type QuoteInput,
} from "./logic.ts";

const INPUT: QuoteInput = {
  optionType: "call", expiry: "2026-11-06T23:59:00Z", strike: "2660", size: "100", side: "sell",
  premium: "6059", premiumUnit: "usd_total", spotRef: "2473.59", quotedAt: "", counterparty: "Acme Desk",
};

describe("evaluate request", () => {
  it("never sends the counterparty label (P-8)", () => {
    const p = evaluateParams("ETH", INPUT, { role: "shadow", implementation: "bs_pricer_black76" });
    assert.deepEqual(Object.keys(p).sort(), [
      "expiry", "implementation", "option_type", "premium", "premium_unit", "role", "side",
      "size", "spot_ref", "strike", "underlying",
    ]);
    assert.ok(!JSON.stringify(p).includes("Acme"));
  });

  it("omits quote fields without a premium", () => {
    const p = evaluateParams("ETH", { ...INPUT, premium: "", side: "" }, {});
    assert.equal(p.premium, undefined);
    assert.equal(p.side, undefined);
    assert.equal(p.premium_unit, undefined);
  });
});

describe("run selection", () => {
  it("prefers production, falls back to the first shadow", () => {
    const shadow = [{ implementation: "bs_pricer_black76", version: "1.0.0" }];
    assert.deepEqual(runSelection({ production: null, shadow }), { role: "shadow", implementation: "bs_pricer_black76" });
    assert.deepEqual(runSelection({ production: shadow[0]!, shadow: [] }), {});
  });
});

describe("verdict bands", () => {
  const t = DEFAULT_THRESHOLDS;
  it("sell side uses floor 90% / walk-away 80% of fair", () => {
    assert.equal(band("sell", 0.987, "inside", t), "at_or_above_floor");
    assert.equal(band("sell", 0.85, "inside", t), "negotiate");
    assert.equal(band("sell", 0.79, "inside", t), "below_walk_away");
    assert.equal(band("sell", 0.95, "below_bid", t), "negotiate");
  });
  it("buy side mirrors around fair", () => {
    assert.equal(band("buy", 1.02, "inside", t), "at_or_above_floor");
    assert.equal(band("buy", 1.15, "inside", t), "negotiate");
    assert.equal(band("buy", 1.25, "inside", t), "below_walk_away");
  });
  it("no band without a fair comparison", () => {
    assert.equal(band("sell", null, "n/a", t), null);
  });
});

describe("chain", () => {
  const chain = [
    { expiry_ts: "A", strike: 110, option_type: "put" },
    { expiry_ts: "A", strike: 100, option_type: "call" },
    { expiry_ts: "A", strike: 100, option_type: "put" },
    { expiry_ts: "B", strike: 100, option_type: "call" },
  ];
  it("builds T-layout rows per expiry", () => {
    const rows = strikeRows(chain, "A");
    assert.deepEqual(rows.map((r) => [r.strike, !!r.call, !!r.put]), [[100, true, true], [110, false, true]]);
  });
  it("picks the nearest expiry with at least 7 DTE", () => {
    const e = [{ expiry_ts: "a", dte: 2, forward: 1, atm_iv: null }, { expiry_ts: "b", dte: 9, forward: 1, atm_iv: null }];
    assert.equal(defaultExpiry(e), "b");
    assert.equal(defaultExpiry([e[0]!]), "a");
    assert.equal(defaultExpiry([]), null);
  });
  it("converts units and quantities", () => {
    const ctx = { spot: 2000, forward: 2010, contractSize: 10, positionSize: 3 };
    assert.equal(convertPrice(20, "usd", "unit", ctx), 20);
    assert.equal(convertPrice(20, "pct", "unit", ctx), 0.01);
    assert.equal(convertPrice(20.1, "coin", "unit", ctx), 0.01);
    assert.equal(convertPrice(20, "usd", "contract", ctx), 200);
    assert.equal(convertPrice(20, "usd", "position", ctx), 60);
    assert.equal(convertPrice(null, "usd", "unit", ctx), null);
  });
});

describe("payoff", () => {
  it("is from our side, net of premium", () => {
    const opt = { isCall: true, strike: 100, premium: 5, size: 2 };
    assert.deepEqual(payoff([90, 100, 120], { ...opt, side: "sell" }), [10, 10, -30]);
    assert.deepEqual(payoff([90, 120], { ...opt, side: "buy" }), [-10, 30]);
  });
});

describe("csv", () => {
  it("quotes and guards against formula injection", () => {
    const csv = toCsv(["label", "edge"], [["=HYPERLINK(\"x\")", -81], ["a,b", null]]);
    assert.equal(csv, 'label,edge\r\n"\'=HYPERLINK(""x"")",-81\r\n"a,b",\r\n');
  });
});

describe("preferences", () => {
  it("survive broken storage", () => {
    const broken = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); } };
    assert.equal(loadPref("gate", 5, broken), 5);
    savePref("gate", 6, broken);
    const mem = new Map<string, string>();
    const store = { getItem: (k: string) => mem.get(k) ?? null, setItem: (k: string, v: string) => void mem.set(k, v) };
    savePref("gate", 6, store);
    assert.equal(loadPref("gate", 5, store), 6);
  });
});
