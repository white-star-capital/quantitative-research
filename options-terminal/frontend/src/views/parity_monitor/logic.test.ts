import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { describe, it } from "node:test";
import type { SlotSummary } from "../../api/types.ts";
import {
  type FeeRates,
  type Opportunity,
  at30d,
  barStatus,
  breakevenFunding,
  filterOpps,
  findParitySlot,
  inventoryRows,
  netMetrics,
  perpOnly,
  pickup,
  recompute,
  returnOnCapital,
  structureFees,
} from "./logic.ts";

// Python-generated math vectors (models/terminal_models/tools/gen_parity_vectors.py).
const VECTORS = new URL("../../../../contracts/fixtures/parity/vectors.json", import.meta.url);
const close = (a: number | null, b: number | null, rel = 1e-12) =>
  a === null || b === null ? a === b : Math.abs(a - b) <= rel * Math.max(1, Math.abs(b));

describe("fee and net-edge recompute matches the Python reference", () => {
  const skip = !existsSync(VECTORS);
  const doc = skip ? { cases: [] } : JSON.parse(readFileSync(VECTORS, "utf8"));
  it("every vector case, both sides, futures and perps", { skip }, () => {
    assert.ok(doc.cases.length >= 40);
    for (const c of doc.cases) {
      const q = c.inputs;
      const rates: FeeRates = doc.fee_rates[q.side];
      const legs = {
        conversion: [q.c_bid, q.p_ask, q.F_ask],
        reversal: [q.c_ask, q.p_bid, q.F_bid],
      } as const;
      for (const structure of ["conversion", "reversal"] as const) {
        const e = c.expected[structure];
        const [call, put, hedge] = legs[structure];
        const fees = structureFees(q.inverse, rates, q.K, call, put, hedge, q.S, q.hedge_is_perp);
        assert.ok(close(fees, e.fees), `fees ${structure}`);
        const m = netMetrics(q.inverse, e.gross_edge, fees, q.K, q.S, q.T);
        for (const k of ["net_edge", "net_edge_usd", "net_edge_bps_notional", "net_edge_ann"] as const) {
          assert.ok(close(m[k], e[k], 1e-10), `${structure} ${k}: ${m[k]} vs ${e[k]}`);
        }
        if (q.hedge_is_perp) {
          assert.ok(close(breakevenFunding(structure, m.net_edge_usd, q.K, q.T), e.breakeven_funding_ann, 1e-10));
        }
      }
    }
  });
  it("pickup is r_lock minus the user's rates", { skip }, () => {
    for (const c of doc.cases) {
      const p = pickup(c.expected.inventory.r_lock, 4.5, 2);
      assert.ok(close(p.net, c.expected.inventory.r_lock - 0.045 - 0.02, 1e-12));
    }
  });
});

const OPP: Opportunity = {
  expiry_ts: "2026-11-14T08:00:00Z", strike: 60500, structure: "conversion", hedge_type: "locked_future",
  hedge_instrument: "BTC-14NOV26", call_px: 0.0726, call_px_unit: "coin", put_px: 0.0716, put_px_unit: "coin",
  hedge_px: 60310.9, gross_edge: 0.0042, fees: 0.0015, net_edge: 0.0027, net_edge_usd: 162, net_edge_bps_notional: 26.8,
  net_edge_ann: 0.0317, breakeven_funding_ann: null, confirmed: true, exec_size_units: 5, hedge_contracts: 30250,
  hedge_rounding_residual_usd: 0, snapshot_age_s: 1.4, flags: "coin_denominated_edge",
};
const TAKER: FeeRates = { option: 0.0003, option_cap: 0.125, future: 0.0005, perp: 0.0005, option_delivery: 0.00015, future_delivery: 0.00025 };
const MAKER: FeeRates = { ...TAKER, option: 0.0002, future: 0.0001, perp: 0.0001 };

describe("display toggles (guide §7.4)", () => {
  it("maker fees raise the net edge and keep gross unchanged", () => {
    const taker = recompute(OPP, TAKER, 60000, "2026-10-14T12:00:00Z");
    const maker = recompute(OPP, MAKER, 60000, "2026-10-14T12:00:00Z");
    assert.equal(maker.gross_edge, taker.gross_edge);
    assert.ok(maker.fees < taker.fees && maker.net_edge > taker.net_edge);
    assert.ok(close(taker.net_edge_usd, taker.net_edge * 60000));
  });
  it("perp rows get a breakeven funding rate on recompute", () => {
    const perp = recompute({ ...OPP, hedge_type: "perp_funding" }, TAKER, 60000, "2026-10-14T12:00:00Z");
    assert.ok(perp.breakeven_funding_ann !== null && perp.breakeven_funding_ann > 0);
  });
  it("capital assumption scales the annualized return", () => {
    assert.equal(returnOnCapital(0.03, 10), 0.3);
    assert.equal(returnOnCapital(0.03, 0), null);
    assert.equal(returnOnCapital(null, 10), null);
  });
});

describe("filters", () => {
  const opps = [OPP, { ...OPP, confirmed: false }, { ...OPP, hedge_type: "perp_funding" as const }];
  it("default shows confirmed locked_future only", () => {
    assert.deepEqual(filterOpps(opps, { showUnconfirmed: false, showPerp: false }), [OPP]);
    assert.equal(filterOpps(opps, { showUnconfirmed: true, showPerp: true }).length, 3);
  });
});

describe("inventory (guide §3.5, §7.2 panel 4)", () => {
  const expiries = [
    { expiry_ts: "a", dte: 31, r_lock_best: 0.06, r_lock_best_strike: 60000, r_lock_best_fees: 0.0008 },
    { expiry_ts: "b", dte: 95, r_lock_best: 0.08, r_lock_best_strike: 61000, r_lock_best_fees: null },
    { expiry_ts: "c", dte: 3, r_lock_best: null, r_lock_best_strike: null },
  ];
  it("sorts by pickup net of forgone yield and skips expiries without a lock", () => {
    const rows = inventoryRows(expiries, { holdingUnits: 2, usdcAltPct: 5, coinYieldPct: 0.5 }, 60000);
    assert.deepEqual(rows.map((r) => r.expiry_ts), ["b", "a"]);
    assert.ok(close(rows[0]!.pickup_net, 0.08 - 0.05 - 0.005));
    assert.ok(close(rows[0]!.usd_at_expiry, 2 * 60000 * Math.exp(0.08 * 95 / 365)));
  });
  it("inputs never leave the browser: the calculation makes no request", () => {
    const original = globalThis.fetch;
    let calls = 0;
    globalThis.fetch = (() => {
      calls++;
      throw new Error("no network");
    }) as typeof fetch;
    try {
      inventoryRows(expiries, { holdingUnits: 1234.5, usdcAltPct: 5, coinYieldPct: 0.5 }, 60000);
      recompute(OPP, MAKER, 60000, "2026-10-14T12:00:00Z");
      returnOnCapital(0.03, 12);
    } finally {
      globalThis.fetch = original;
    }
    assert.equal(calls, 0);
  });
});

describe("monitor columns (guide §7.1)", () => {
  const rows = [
    { dte: 10, implied_basis_ann: 0.05, hedge_type: "perp_funding" },
    { dte: 31, implied_basis_ann: 0.06, hedge_type: "perp_funding" },
    { dte: 95, implied_basis_ann: null, hedge_type: "perp_funding" },
  ];
  it("interpolates at 30d between bracketing expiries, never extrapolates", () => {
    assert.ok(close(at30d(rows, "implied_basis_ann"), 0.05 + 0.01 * (20 / 21)));
    assert.equal(at30d(rows, "implied_basis_ann", 120), null);
  });
  it("perp-only assets are recognised", () => {
    assert.equal(perpOnly(rows), true);
    assert.equal(perpOnly([{ hedge_type: "locked_future" }]), false);
  });
});

describe("band status", () => {
  it("net when the structure survives fees, gross otherwise", () => {
    const band = { expiry_ts: OPP.expiry_ts, strike: 60500, gross_conv: 0.004, gross_rev: -0.01 };
    assert.deepEqual(barStatus(band, [OPP]), { status: "net", side: "conversion", confirmed: true });
    assert.deepEqual(barStatus(band, []), { status: "gross", side: "conversion", confirmed: false });
    assert.deepEqual(barStatus({ ...band, gross_conv: -1 }, []), { status: "inside", side: null, confirmed: false });
  });
});

describe("catalog lookup", () => {
  it("finds the parity slot by view id", () => {
    const slot = { slot_id: "anything", display: { default_view: "parity_monitor" } } as unknown as SlotSummary;
    assert.equal(findParitySlot([slot]), slot);
    assert.equal(findParitySlot([]), null);
  });
});
