// Pure logic for the bs_pricer views (guide §7). No React, no fetch: unit-tested with node:test.
import type { Cell, Composite, Table } from "../../api/payloads.ts";
import { records, section } from "../../api/payloads.ts";

// ---- run selection (shadow period) ---------------------------------------------------------

export interface Implementations {
  production: { implementation: string; version: string } | null;
  shadow: { implementation: string; version: string }[];
}

/** Production when there is one; during the initial shadow period, the first shadow (M-12). */
export function runSelection(impls: Implementations): Record<string, string> {
  if (impls.production) return {};
  const first = impls.shadow[0];
  return first ? { role: "shadow", implementation: first.implementation } : {};
}

export function query(params: Record<string, string>): string {
  const q = new URLSearchParams();
  for (const k of Object.keys(params).sort()) q.set(k, params[k] as string);
  const s = q.toString();
  return s ? `?${s}` : "";
}

// ---- evaluate request (P-8: the counterparty label never leaves the browser) ---------------

export interface QuoteInput {
  optionType: "call" | "put";
  expiry: string; // ISO UTC with time, ends in Z
  strike: string;
  size: string;
  side: "" | "sell" | "buy";
  premium: string;
  premiumUnit: "usd_per_unit" | "usd_total" | "pct_spot" | "coin_per_unit";
  spotRef: string;
  quotedAt: string;
  counterparty: string; // browser memory only
}

const SENT_KEYS = new Set([
  "underlying", "role", "implementation", "run_id", "option_type", "expiry", "strike", "size",
  "premium", "premium_unit", "side", "spot_ref", "quoted_at",
]);

export function evaluateParams(
  underlying: string,
  input: QuoteInput,
  selection: Record<string, string>,
): Record<string, string> {
  const p: Record<string, string> = {
    ...selection,
    underlying,
    option_type: input.optionType,
    expiry: input.expiry,
    strike: input.strike.trim(),
  };
  if (input.size.trim()) p.size = input.size.trim();
  if (input.premium.trim()) {
    p.premium = input.premium.trim();
    p.premium_unit = input.premiumUnit;
    if (input.side) p.side = input.side;
  }
  if (input.spotRef.trim()) p.spot_ref = input.spotRef.trim();
  if (input.quotedAt.trim()) p.quoted_at = input.quotedAt.trim();
  for (const k of Object.keys(p)) {
    if (!SENT_KEYS.has(k)) delete p[k];
  }
  return p;
}

// ---- verdict bands (display only; the API returns neutral metrics) ------------------------

export type Band = "at_or_above_floor" | "negotiate" | "below_walk_away";

export interface Thresholds {
  floor: number; // fraction of fair, e.g. 0.9
  walkAway: number; // e.g. 0.8
}

export const DEFAULT_THRESHOLDS: Thresholds = { floor: 0.9, walkAway: 0.8 };

/**
 * Selling: premium as % of fair against the floor / walk-away (desk playbook defaults 90% / 80%);
 * below the screen bid is at best "negotiate". Buying mirrors it around fair (110% / 120%), so
 * "at/above floor" always means "good for us".
 */
export function band(side: "sell" | "buy", pctOfFair: number | null, screen: string, t: Thresholds): Band | null {
  if (pctOfFair === null || !Number.isFinite(pctOfFair)) return null;
  const score = side === "sell" ? pctOfFair : 2 - pctOfFair;
  if (score < t.walkAway) return "below_walk_away";
  if (score < t.floor) return "negotiate";
  if (side === "sell" && screen === "below_bid") return "negotiate";
  if (side === "buy" && screen === "above_ask") return "negotiate";
  return "at_or_above_floor";
}

export const BAND_LABEL: Record<Band, string> = {
  at_or_above_floor: "AT/ABOVE FLOOR",
  negotiate: "NEGOTIATE ZONE",
  below_walk_away: "BELOW WALK-AWAY",
};

// ---- CSV (client-side only) -----------------------------------------------------------------

/** RFC 4180 quoting, plus a guard against spreadsheet formula injection in text cells. */
export function toCsv(header: string[], rows: (string | number | boolean | null)[][]): string {
  const cell = (v: string | number | boolean | null): string => {
    if (v === null) return "";
    let s = String(v);
    if (typeof v === "string" && /^[=+\-@\t\r]/.test(s)) s = `'${s}`;
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [header, ...rows].map((r) => r.map(cell).join(",")).join("\r\n") + "\r\n";
}

// ---- chain --------------------------------------------------------------------------------

export type ChainRow = Record<string, Cell>;
export type PriceUnit = "usd" | "pct" | "coin";
export type Quantity = "unit" | "contract" | "position";

export interface StrikeRow {
  strike: number;
  call: ChainRow | null;
  put: ChainRow | null;
}

export interface ExpiryInfo {
  expiry_ts: string;
  dte: number;
  forward: number;
  atm_iv: number | null;
}

export function expiriesOf(payload: Composite): ExpiryInfo[] {
  const t = section<Table>(payload, "expiries");
  return t ? (records(t) as unknown as ExpiryInfo[]) : [];
}

export function chainOf(payload: Composite): ChainRow[] {
  const t = section<Table>(payload, "chain");
  return t ? records(t) : [];
}

/** Nearest expiry with at least `minDte` days (default 7), else the last one. */
export function defaultExpiry(expiries: ExpiryInfo[], minDte = 7): string | null {
  const pick = expiries.find((e) => e.dte >= minDte) ?? expiries[expiries.length - 1];
  return pick ? pick.expiry_ts : null;
}

/** T-layout rows for one expiry: calls left, strike centre, puts right. */
export function strikeRows(chain: ChainRow[], expiryTs: string): StrikeRow[] {
  const by = new Map<number, StrikeRow>();
  for (const r of chain) {
    if (r.expiry_ts !== expiryTs) continue;
    const k = r.strike as number;
    const row = by.get(k) ?? { strike: k, call: null, put: null };
    if (r.option_type === "call") row.call = r;
    else row.put = r;
    by.set(k, row);
  }
  return [...by.values()].sort((a, b) => a.strike - b.strike);
}

/**
 * A USD-per-unit price in the chosen unit and quantity. Coin = USD / forward (Deribit's inverse
 * convention); only offered for BTC/ETH. Per contract multiplies by `contract_size`.
 */
export function convertPrice(
  usd: number | null,
  unit: PriceUnit,
  qty: Quantity,
  ctx: { spot: number; forward: number; contractSize: number; positionSize: number },
): number | null {
  if (usd === null || !Number.isFinite(usd)) return null;
  const perUnit = unit === "usd" ? usd : unit === "pct" ? usd / ctx.spot : usd / ctx.forward;
  const mult = qty === "unit" ? 1 : qty === "contract" ? ctx.contractSize : ctx.positionSize;
  return perUnit * mult;
}

export const COIN_ASSETS = new Set(["BTC", "ETH"]);

// ---- payoff -------------------------------------------------------------------------------

/** P&L at expiry per position for an option, from our side, net of premium. */
export function payoff(
  spots: number[],
  opt: { isCall: boolean; strike: number; premium: number; size: number; side: "sell" | "buy" },
): number[] {
  const sign = opt.side === "sell" ? -1 : 1;
  return spots.map((s) => {
    const intrinsic = opt.isCall ? Math.max(s - opt.strike, 0) : Math.max(opt.strike - s, 0);
    return sign * (intrinsic - opt.premium) * opt.size;
  });
}

// ---- per-viewer preferences (conveniences only: wrapped in try/catch) ---------------------

export function loadPref<T>(key: string, fallback: T, storage: Pick<Storage, "getItem"> | null, prefix = "bs_pricer"): T {
  try {
    const raw = storage?.getItem(`${prefix}.${key}`);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export function savePref(key: string, value: unknown, storage: Pick<Storage, "setItem"> | null, prefix = "bs_pricer"): void {
  try {
    storage?.setItem(`${prefix}.${key}`, JSON.stringify(value));
  } catch {
    // Private mode or blocked storage: the preference simply isn't remembered.
  }
}
