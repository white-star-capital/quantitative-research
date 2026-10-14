// Parity tab formatting (PARITY_MONITOR_IMPLEMENTATION_GUIDE.md §7; pricer §7.8 otherwise).
// Format at display time; raw values stay exact. Unknown is "—", never zero.
import { fmtUsdTotal } from "./pricer.ts";
import { MISSING, formatNumber } from "./units.ts";

type N = number | null | undefined;
const ok = (v: N): v is number => typeof v === "number" && Number.isFinite(v);

/** Basis / rates from decimals: "6.85% ann." */
export const fmtRateAnn = (v: N): string => (ok(v) ? `${formatNumber(v * 100, 2)}% ann.` : MISSING);

/** Dislocations: signed bp, 1 dp: "+12.4 bp" */
export const fmtBp = (v: N): string => (ok(v) ? `${formatNumber(v, 1, true)} bp` : MISSING);

/** Plain percentage input/output (already in %): "4.50%" */
export const fmtPctPoints = (v: N): string => (ok(v) ? `${formatNumber(v, 2, true)}%` : MISSING);

/** Coin edge with USD equivalent: "0.000412 BTC ≈ $41" */
export function fmtCoinEdge(v: N, asset: string, spot: N): string {
  if (!ok(v)) return MISSING;
  const coin = `${formatNumber(v, 6, true)} ${asset}`;
  return ok(spot) ? `${coin} ≈ ${fmtUsdTotal(v * spot, true)}` : coin;
}

/** Native leg price: coin to 6 dp on inverse books, USD otherwise. */
export function fmtNative(v: N, unit: string | null | undefined, asset: string): string {
  if (!ok(v)) return MISSING;
  return unit === "coin" ? `${formatNumber(v, 6)} ${asset}` : fmtUsdTotal(v);
}
