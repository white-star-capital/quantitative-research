// Parity tab logic (PARITY_MONITOR_IMPLEMENTATION_GUIDE.md §3.3, §3.5, §7). Pure functions only:
// the maker/taker toggle recomputes fees and net edges from the run's recorded fee table with
// the same formulas as models/terminal_models/core/quant/parity.py, held together by
// contracts/fixtures/parity/vectors.json. Inventory and capital inputs never leave the browser.
import type { Cell, Composite, ScalarSet, Table } from "../../api/payloads.ts";
import { records, scalarMap, section } from "../../api/payloads.ts";
import type { SlotSummary } from "../../api/types.ts";

export const VIEW_ID = "parity_monitor";
const YEAR_MS = 365 * 86_400_000;

export type Side = "taker" | "maker";
export type HedgeType = "locked_future" | "perp_funding";

export interface FeeRates {
  option: number;
  option_cap: number;
  future: number;
  perp: number;
  option_delivery: number;
  future_delivery: number;
}

export interface Opportunity {
  expiry_ts: string;
  strike: number;
  structure: "conversion" | "reversal";
  hedge_type: HedgeType;
  hedge_instrument: string;
  call_px: number;
  call_px_unit: string;
  put_px: number;
  put_px_unit: string;
  hedge_px: number;
  gross_edge: number;
  fees: number;
  net_edge: number;
  net_edge_usd: number;
  net_edge_bps_notional: number;
  net_edge_ann: number | null;
  breakeven_funding_ann: number | null;
  confirmed: boolean;
  exec_size_units: number | null;
  hedge_contracts: number;
  hedge_rounding_residual_usd: number;
  snapshot_age_s: number;
  flags: string;
}

export type Row = Record<string, Cell>;

/** The catalog's parity slot, found by its view id, never by a hard-coded slot id (F-2). */
export function findParitySlot(slots: readonly SlotSummary[]): SlotSummary | null {
  return slots.find((s) => s.display?.default_view === VIEW_ID) ?? null;
}

export function table(payload: Composite | null, key: string): Row[] {
  const t = payload ? section<Table>(payload, key) : null;
  return t ? records(t) : [];
}

export function context(payload: Composite | null): Record<string, number | boolean | null> {
  const set = payload ? section<ScalarSet>(payload, "context") : null;
  if (!set) return {};
  return Object.fromEntries(Object.entries(scalarMap(set)).map(([k, v]) => [k, v.value]));
}

export function skewFlagged(payload: Composite | null): boolean {
  const set = payload ? section<ScalarSet>(payload, "context") : null;
  return set ? (scalarMap(set).snapshot_skew_s?.flags ?? []).includes("snapshot_skew") : false;
}

export function feeRates(payload: Composite | null, side: Side): FeeRates | null {
  const row = table(payload, "fees").find((r) => r.side === side);
  if (!row) return null;
  const keys = ["option", "option_cap", "future", "perp", "option_delivery", "future_delivery"] as const;
  if (keys.some((k) => typeof row[k] !== "number")) return null;
  return Object.fromEntries(keys.map((k) => [k, row[k] as number])) as unknown as FeeRates;
}

export function scheduleVersion(payload: Composite | null): string | null {
  const v = table(payload, "fees")[0]?.schedule_version;
  return typeof v === "string" ? v : null;
}

// ---- §3.3 fees and net edge (mirror of core/quant/parity.py) ------------------------------

export function optionFee(price: number, rate: number, cap: number, unitValue: number): number {
  return Math.min(rate * unitValue, cap * price);
}

export function hedgeFee(inverse: boolean, rate: number, K: number, F: number): number {
  return inverse ? (rate * K) / F : rate * F;
}

export function structureFees(
  inverse: boolean, r: FeeRates, K: number, call: number, put: number, hedge: number, index: number, hedgeIsPerp: boolean,
): number {
  const unitValue = inverse ? 1 : index;
  const trade =
    optionFee(call, r.option, r.option_cap, unitValue) +
    optionFee(put, r.option, r.option_cap, unitValue) +
    hedgeFee(inverse, hedgeIsPerp ? r.perp : r.future, K, hedge);
  let delivery = r.option_delivery * unitValue;
  if (!hedgeIsPerp) delivery += hedgeFee(inverse, r.future_delivery, K, hedge);
  return trade + delivery;
}

export interface NetMetrics {
  fees: number;
  net_edge: number;
  net_edge_usd: number;
  net_edge_bps_notional: number;
  net_edge_ann: number | null;
}

export function netMetrics(inverse: boolean, gross: number, fees: number, K: number, S: number, T: number): NetMetrics {
  const net = gross - fees;
  const usd = inverse ? net * S : net;
  return { fees, net_edge: net, net_edge_usd: usd, net_edge_bps_notional: (usd / K) * 1e4, net_edge_ann: T > 0 ? usd / K / T : null };
}

export function breakevenFunding(structure: string, netUsd: number, K: number, T: number): number | null {
  if (K <= 0 || T <= 0) return null;
  const rate = netUsd / (K * T);
  return structure === "conversion" ? rate : -rate;
}

export function yearsTo(expiryIso: string, asOfIso: string): number {
  return (Date.parse(expiryIso) - Date.parse(asOfIso)) / YEAR_MS;
}

/** One opportunity at another fee side (§7.4): fees, net edge and breakeven recomputed. */
export function recompute(o: Opportunity, rates: FeeRates, spot: number, asOf: string): Opportunity {
  const inverse = o.call_px_unit === "coin";
  const T = yearsTo(o.expiry_ts, asOf);
  const perp = o.hedge_type === "perp_funding";
  const fees = structureFees(inverse, rates, o.strike, o.call_px, o.put_px, o.hedge_px, spot, perp);
  const m = netMetrics(inverse, o.gross_edge, fees, o.strike, spot, T);
  return { ...o, ...m, breakeven_funding_ann: perp ? breakevenFunding(o.structure, m.net_edge_usd, o.strike, T) : null };
}

/** Annualized return on capital when `capitalPct`% of notional is posted as margin (§7.4). */
export function returnOnCapital(netAnn: number | null, capitalPct: number): number | null {
  if (netAnn === null || !(capitalPct > 0)) return null;
  return netAnn / (capitalPct / 100);
}

export interface Filter {
  showUnconfirmed: boolean;
  showPerp: boolean;
}

/** Default view: confirmed, locked_future. Under snapshot skew no unconfirmed row is ever
 *  shown as executable (D-12): they stay visible only when explicitly revealed. */
export function filterOpps(opps: Opportunity[], f: Filter): Opportunity[] {
  return opps.filter((o) => (f.showUnconfirmed || o.confirmed) && (f.showPerp || o.hedge_type === "locked_future"));
}

export function executable(o: Opportunity): boolean {
  return o.confirmed && o.net_edge > 0;
}

export const RESIDUAL_RISK: Record<HedgeType, string> = {
  locked_future: "Locked by the same-expiry future: venue/counterparty and leg-execution risk; inverse edges are locked in coin.",
  perp_funding: "Not locked: hedged with the perpetual, so cumulative funding to expiry and perp-to-delivery convergence decide the outcome.",
};

// ---- §3.5 inventory pickup ---------------------------------------------------------------

export interface InventoryInputs {
  holdingUnits: number;
  usdcAltPct: number;
  coinYieldPct: number;
}

export interface InventoryRow {
  expiry_ts: string;
  dte: number;
  strike: number;
  r_lock: number;
  pickup_vs_usdc: number;
  pickup_net: number;
  usd_at_expiry: number | null;
  fees: number | null;
}

export function pickup(rLock: number, usdcAltPct: number, coinYieldPct: number): { vsUsdc: number; net: number } {
  const vsUsdc = rLock - usdcAltPct / 100;
  return { vsUsdc, net: vsUsdc - coinYieldPct / 100 };
}

/** Per expiry with an r_lock: pickup and the USD the holding's value compounds to at r_lock
 *  (continuous) by expiry. Sorted by pickup net of forgone yield, best first. */
export function inventoryRows(expiries: Row[], inputs: InventoryInputs, spot: number | null): InventoryRow[] {
  const out: InventoryRow[] = [];
  for (const e of expiries) {
    const r = e.r_lock_best;
    if (typeof r !== "number" || typeof e.r_lock_best_strike !== "number" || typeof e.dte !== "number") continue;
    const p = pickup(r, inputs.usdcAltPct, inputs.coinYieldPct);
    const T = e.dte / 365;
    out.push({
      expiry_ts: String(e.expiry_ts),
      dte: e.dte,
      strike: e.r_lock_best_strike,
      r_lock: r,
      pickup_vs_usdc: p.vsUsdc,
      pickup_net: p.net,
      usd_at_expiry: spot !== null && inputs.holdingUnits > 0 ? inputs.holdingUnits * spot * Math.exp(r * T) : null,
      fees: typeof e.r_lock_best_fees === "number" ? e.r_lock_best_fees : null,
    });
  }
  return out.sort((a, b) => b.pickup_net - a.pickup_net);
}

// ---- monitor columns (§7.1) --------------------------------------------------------------

/** Linear in DTE between the bracketing expiries that have a value; null outside them. */
export function at30d(expiries: Row[], key: string, days = 30): number | null {
  const pts = expiries
    .filter((e) => typeof e[key] === "number" && typeof e.dte === "number")
    .map((e) => [e.dte as number, e[key] as number] as const)
    .sort((a, b) => a[0] - b[0]);
  for (let i = 0; i < pts.length; i++) {
    const [d, v] = pts[i]!;
    if (d === days) return v;
    const next = pts[i + 1];
    if (next && d < days && next[0] > days) return v + ((next[1] - v) * (days - d)) / (next[0] - d);
  }
  return null;
}

export function perpOnly(expiries: Row[]): boolean {
  return expiries.length > 0 && expiries.every((e) => e.hedge_type !== "locked_future");
}

// ---- band chart (§7.2 panel 2) -----------------------------------------------------------

export type BarStatus = "net" | "gross" | "inside";

/** A bar entirely above the hedge ask is a conversion, below the bid a reversal (gross). It is
 *  `net` when that structure also appears as an opportunity after fees. */
export function barStatus(band: Row, opps: Opportunity[]): { status: BarStatus; side: "conversion" | "reversal" | null; confirmed: boolean } {
  const conv = typeof band.gross_conv === "number" && band.gross_conv > 0;
  const rev = typeof band.gross_rev === "number" && band.gross_rev > 0;
  const side = conv ? "conversion" : rev ? "reversal" : null;
  if (!side) return { status: "inside", side: null, confirmed: false };
  const match = opps.find((o) => o.expiry_ts === band.expiry_ts && o.strike === band.strike && o.structure === side);
  return { status: match ? "net" : "gross", side, confirmed: match?.confirmed ?? false };
}
