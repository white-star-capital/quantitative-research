// Pricer formatting conventions (BS_PRICER_IMPLEMENTATION_GUIDE.md §7.8). Format at display time;
// raw values stay exact. Unknown is "—", never zero.
import { MISSING, formatNumber } from "./units.ts";

const MINUS = "−";
const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

type N = number | null | undefined;
const ok = (v: N): v is number => typeof v === "number" && Number.isFinite(v);

/** Spot / strike precision per asset. */
export const ASSET_DP: Record<string, number> = { BTC: 0, ETH: 1, SOL: 2, HYPE: 3 };

export function fmtSpot(asset: string, v: N): string {
  return ok(v) ? formatNumber(v, ASSET_DP[asset] ?? 2) : MISSING;
}

function usd(body: string): string {
  return body.startsWith(MINUS) || body.startsWith("+") ? `${body[0]}$${body.slice(1)}` : `$${body}`;
}

/** Premium: 4 significant figures, at least 2 dp below $100. $1,234 · $61.40 · $0.8523 */
export function fmtPremium(v: N, signed = false): string {
  if (!ok(v)) return MISSING;
  const abs = Math.abs(v);
  const digits = abs >= 1 ? Math.floor(Math.log10(abs)) + 1 : abs > 0 ? Math.floor(Math.log10(abs)) + 1 : 1;
  let dp = Math.max(0, 4 - digits);
  if (abs < 100) dp = Math.max(dp, 2);
  return usd(formatNumber(v, Math.min(dp, 8), signed));
}

/** Totals: thousands separators, no decimals at or above $1,000. */
export function fmtUsdTotal(v: N, signed = false): string {
  if (!ok(v)) return MISSING;
  return usd(formatNumber(v, Math.abs(v) >= 1000 ? 0 : 2, signed));
}

export const fmtIv = (v: N): string => (ok(v) ? `${formatNumber(v * 100, 1)}%` : MISSING);
/** Vol differences (VRP, RR, BF, vol edge) from decimals: "+5.2 vp". */
export const fmtVp = (v: N): string => (ok(v) ? `${formatNumber(v * 100, 1, true)} vp` : MISSING);
/** Vol edge already in vol points (evaluate `vol_edge_vp`). */
export const fmtVpPoints = (v: N): string => (ok(v) ? `${formatNumber(v, 1, true)} vp` : MISSING);
export const fmtDelta = (v: N): string => (ok(v) ? formatNumber(v, 2, true) : MISSING);
export const fmtGamma = (v: N): string => (ok(v) ? formatNumber(v, 3) : MISSING);

/** Vega / theta in USD with at least 2 significant figures: $2.60, −$1.40, $0.0083. */
export function fmtGreekUsd(v: N): string {
  if (!ok(v)) return MISSING;
  const abs = Math.abs(v);
  const dp = abs >= 0.1 || abs === 0 ? 2 : Math.min(8, 1 - Math.floor(Math.log10(abs)));
  return usd(formatNumber(v, dp));
}

/** Probabilities, % of fair, yields from decimals: 0–1 dp. 30% · 98.7% */
export function fmtPct(v: N): string {
  if (!ok(v)) return MISSING;
  const pct = v * 100;
  return `${formatNumber(pct, Number.isInteger(Math.round(pct * 10) / 10) ? 0 : 1)}%`;
}

/** "06 NOV 26 08:00 UTC" */
export function fmtExpiry(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return MISSING;
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getUTCDate())} ${MONTHS[d.getUTCMonth()]} ${p(d.getUTCFullYear() % 100)} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())} UTC`;
}

/** DTE: one decimal under 7 days. "24 d" · "2.4 d" */
export function fmtDte(days: N): string {
  if (!ok(days)) return MISSING;
  return `${formatNumber(days, days < 7 ? 1 : 0)} d`;
}
