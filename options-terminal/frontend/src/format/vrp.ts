// VRP signal formatting (RV_FORECAST_IMPLEMENTATION_GUIDE.md §12). Format at display time; raw
// values stay exact. Unknown is "—", never zero. Vols and VRP use the pricer's fmtIv / fmtVp.
import { fmtExpiry } from "./pricer.ts";
import { MISSING, formatNumber, formatValue } from "./units.ts";

type N = number | null | undefined;
const ok = (v: N): v is number => typeof v === "number" && Number.isFinite(v);

/** z-scores: signed, 2 dp, in standard deviations: "+1.80σ" */
export const fmtZ = (v: N): string => (ok(v) ? `${formatNumber(v, 2, true)}σ` : MISSING);

/** Variance shares from decimals, whole percent: "35%" */
export const fmtShare = (v: N): string => (ok(v) ? `${formatNumber(v * 100, 0)}%` : MISSING);

/** Unix epoch seconds as "28 OCT, 00:00 UTC" (§12.4 `last_complete_day`). */
export const fmtUnixDay = (v: N): string => formatValue(v, { quantity: "timestamp", scale: "absolute", convention: "unix_seconds" });

/** Daily closes on a time axis, from epoch ms: "28 OCT 26" */
export const fmtDay = (ms: N): string => (ok(ms) && Number.isFinite(new Date(ms).getTime()) ? fmtExpiry(new Date(ms).toISOString()).slice(0, 9) : MISSING);
