import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type { Cell, Composite, ScalarItem } from "../../api/payloads.ts";
import { checkPayload } from "../../api/payloads.ts";
import type { SlotSummary } from "../../api/types.ts";
import {
  type Gate,
  type SignalRow,
  chips,
  cone,
  contextItems,
  ctxNum,
  events,
  findVrpSlot,
  gate,
  hasValues,
  history,
  ivPoints,
  rowAt,
  signalRows,
  signalStatus,
  VRP_COLUMNS,
  vrpCovers,
  zTitle,
} from "./logic.ts";
import { MONITOR_COLUMNS } from "../bs_pricer/logic.ts";

// TEST-ONLY synthetic payload in the guide §12 shape. Illustrative numbers, not market data (D-10).
const vol = { quantity: "realized_vol", scale: "decimal" } as const;
const SIGNAL_COLS = [
  "tenor_days", "iv_impl", "rv_forecast", "rv_low", "rv_high", "vrp", "vrp_z", "comp_baseline", "comp_seasonal", "comp_event",
  "event_in_horizon", "rich", "event_hump", "signal_ok", "flags",
];
const BOOL = new Set(["event_in_horizon", "rich", "event_hump", "signal_ok"]);

function synthetic(rows: Cell[][], items: Partial<ScalarItem>[]): Composite {
  return {
    sections: [
      {
        key: "signal", label: "Signal", payload_kind: "table", payload_version: "1",
        payload: {
          columns: SIGNAL_COLS.map((k) => ({ key: k, label: k, type: k === "flags" ? "string" : BOOL.has(k) ? "boolean" : "number" })),
          rows,
        },
      },
      {
        key: "context", label: "Context", payload_kind: "scalar_set", payload_version: "1",
        payload: { items: items.map((i) => ({ label: i.key, unit: { quantity: "other", scale: "absolute" }, ...i })) },
      },
      {
        key: "cone", label: "Cone", payload_kind: "curve", payload_version: "1",
        payload: {
          x: { label: "Horizon (days)", unit: { quantity: "tenor_days", scale: "absolute" }, values: [1, 2, 3] },
          series: [
            { key: "rv_forecast", label: "RV", unit: vol, values: [0.4, null, 0.42] },
            { key: "atm_iv", label: "ATM IV", unit: { quantity: "implied_vol", scale: "decimal" }, values: [null, 0.5, null] },
          ],
        },
      },
      {
        key: "history", label: "History", payload_kind: "timeseries", payload_version: "1",
        payload: {
          t: ["2000-01-01T00:00:00Z", "2000-01-02T00:00:00Z"],
          series: [{ key: "rv_realized_30d", label: "Realized", unit: vol, values: [null, 0.4] }],
        },
      },
      {
        key: "events", label: "Events", payload_kind: "table", payload_version: "1",
        payload: {
          columns: [
            { key: "ts", label: "Time", type: "timestamp" }, { key: "type", label: "Type", type: "string" },
            { key: "scope", label: "Scope", type: "string" }, { key: "window_before", label: "b", type: "number" },
            { key: "window_after", label: "a", type: "number" }, { key: "horizon_day", label: "k", type: "number" },
            { key: "source", label: "Source", type: "string" },
          ],
          rows: [["2000-01-28T08:00:00Z", "opex_monthly", "GLOBAL", 1, 1, 27, "rule"]],
        },
      },
    ],
  };
}

const ROW_30: Cell[] = [30, 0.5, 0.4, 0.3, 0.45, 0.1, 1.8, 0.7, 0.1, 0.2, true, true, false, true, "z_basis_dvol"];
const ROW_7: Cell[] = [7, null, 0.4, 0.3, 0.6, null, null, 0.8, 0.1, 0.1, false, null, false, false, "iv_unavailable,z_unavailable"];
const ENABLED = { key: "signal_enabled", value: true };
const PAYLOAD = synthetic([ROW_30, ROW_7], [ENABLED, { key: "event_hump_share", value: 0.35 }]);
const row = (over: Partial<SignalRow> = {}): SignalRow => ({ ...rowAt(signalRows(PAYLOAD), 7)!, ...over });

describe("synthetic payload", () => {
  it("passes the frontend's own composite checks", () => {
    assert.equal(checkPayload("composite", PAYLOAD).ok, true);
  });
});

describe("catalog lookup (F-2)", () => {
  const slot = { slot_id: "anything", scope: { dimensions: ["underlying"], allowed: { underlying: ["BTC", "ETH"] } }, display: { default_view: "vrp_signal" } } as unknown as SlotSummary;
  it("finds the VRP slot by view id; none means no tab and no columns", () => {
    assert.equal(findVrpSlot([slot]), slot);
    assert.equal(findVrpSlot([]), null);
    assert.equal(findVrpSlot([{ ...slot, display: { default_view: "parity_monitor" } }]), null);
  });
  it("covers only the slot's allowed underlyings", () => {
    assert.equal(vrpCovers(slot, "BTC"), true);
    assert.equal(vrpCovers(slot, "SOL"), false);
    assert.equal(vrpCovers({ ...slot, scope: { dimensions: [] } }, "BTC"), false);
  });
});

describe("parsing (§12)", () => {
  it("signal rows sort by tenor, keep nulls as nulls and split flags", () => {
    const rows = signalRows(PAYLOAD);
    assert.deepEqual(rows.map((r) => r.tenor_days), [7, 30]);
    const r7 = rowAt(rows, 7)!;
    assert.equal(r7.iv_impl, null);
    assert.equal(r7.vrp, null);
    assert.equal(r7.vrp_z, null);
    assert.equal(r7.rich, null);
    assert.deepEqual(r7.flags, ["iv_unavailable", "z_unavailable"]);
    assert.deepEqual(rowAt(rows, 30)!.flags, ["z_basis_dvol"]);
    assert.equal(rowAt(rows, 60), null);
  });
  it("missing sections read as empty, never as zeros", () => {
    const empty = { sections: [] } as Composite;
    assert.deepEqual(signalRows(empty), []);
    assert.deepEqual(signalRows(null), []);
    assert.deepEqual(events(empty), []);
    assert.equal(cone(empty), null);
    assert.equal(history(empty), null);
    assert.equal(ctxNum(contextItems(empty), "dvol"), null);
  });
  it("cone: absent series are gaps, ATM IV markers only where priced", () => {
    const c = cone(PAYLOAD)!;
    assert.deepEqual(c.s("rv_forecast"), [0.4, null, 0.42]);
    assert.deepEqual(c.s("rv_path"), [null, null, null]);
    assert.deepEqual(ivPoints(c.x, c.s("atm_iv")), [{ x: 2, y: 0.5 }]);
    assert.equal(hasValues(c.s("rv_path")), false);
  });
  it("history times become epoch ms; events keep their fields", () => {
    const h = history(PAYLOAD)!;
    assert.deepEqual(h.t, [Date.parse("2000-01-01T00:00:00Z"), Date.parse("2000-01-02T00:00:00Z")]);
    assert.deepEqual(h.s("rv_realized_30d"), [null, 0.4]);
    assert.deepEqual(h.s("dvol"), [null, null]);
    assert.deepEqual(events(PAYLOAD)[0], {
      ts: "2000-01-28T08:00:00Z", type: "opex_monthly", scope: "GLOBAL", window_before: 1, window_after: 1, horizon_day: 27, source: "rule",
    });
  });
});

describe("staleness gate (§10, §12.4)", () => {
  const ctx = (items: Partial<ScalarItem>[]) => contextItems(synthetic([], items));
  it("open only when fresh and the model enabled the signal", () => {
    assert.deepEqual(gate(false, ctx([ENABLED])), { open: true, reason: null });
    assert.deepEqual(gate(true, ctx([ENABLED])), { open: false, reason: "RUN IS STALE" });
  });
  it("a lagging RV input or a missing flag shuts it", () => {
    assert.deepEqual(gate(false, ctx([{ key: "signal_enabled", value: false, flags: ["rv_input_lagging"] }])), { open: false, reason: "RV INPUT LAGGING" });
    assert.deepEqual(gate(false, ctx([{ key: "signal_enabled", value: false, flags: ["event_calendar_incomplete"] }])), { open: false, reason: "EVENT CALENDAR INCOMPLETE" });
    assert.equal(gate(false, ctx([{ key: "signal_enabled", value: null }])).open, false);
    assert.equal(gate(false, ctx([])).open, false);
  });
});

describe("chips and status (band discipline, event hump)", () => {
  const open: Gate = { open: true, reason: null };
  const shut: Gate = { open: false, reason: "RUN IS STALE" };
  const labels = (r: SignalRow, g = open) => chips(r, g, 0.35).map((c) => `${c.label}:${c.tone}`);
  it("RICH only when rich and no event hump", () => {
    assert.deepEqual(labels(row({ rich: true })), ["RICH:good"]);
    assert.deepEqual(labels(row({ rich: true, event_hump: true, comp_event: 0.5 })), ["EVENT HUMP:mid"]);
    assert.deepEqual(labels(row({ rich: false })), []);
    assert.deepEqual(labels(row({ rich: null })), []);
  });
  it("a hump explains itself; an event in horizon without a hump is a dim EVENT chip", () => {
    const hump = chips(row({ rich: true, event_hump: true, comp_event: 0.5 }), open, 0.35)[0]!;
    assert.match(hump.title, /event share 50% > 35%/);
    assert.match(hump.title, /Not premium/);
    assert.deepEqual(labels(row({ event_in_horizon: true })), ["EVENT:dim"]);
  });
  it("stale or disabled: never green", () => {
    assert.deepEqual(labels(row({ rich: true }), shut), ["RICH:dim"]);
    assert.deepEqual(signalStatus(row({ signal_ok: true }), shut), { label: "DISABLED", tone: "dim", title: "RUN IS STALE" });
  });
  it("status follows the model's signal_ok and names why not", () => {
    assert.equal(signalStatus(row({ signal_ok: true }), open).label, "SIGNAL");
    const no = signalStatus(row(), open);
    assert.equal(no.label, "NO SIGNAL");
    assert.equal(no.title, "no ATM IV, no z-score");
  });
});

describe("z cell (VRP-A-5)", () => {
  it("null z explains itself", () => {
    assert.equal(zTitle(row()), "no IV history (VRP-A-5)");
    assert.match(zTitle(row({ flags: ["z_insufficient_history"] })), /Insufficient history/);
  });
  it("a scored z names its DVOL basis", () => {
    assert.match(zTitle(rowAt(signalRows(PAYLOAD), 30)!), /DVOL − RV forecast 30d/);
  });
});

describe("pricer monitor columns", () => {
  it("never repeat a label once the VRP columns are appended (labels are React keys)", () => {
    const all = [...MONITOR_COLUMNS, ...VRP_COLUMNS];
    assert.equal(new Set(all).size, all.length);
  });
});
