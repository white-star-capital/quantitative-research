// Unit-driven display formatting (FRONTEND.md F-15/F-16, front_end_design.md §3).
// Raw values stay exact; format once, at display time.
import type { Unit } from "../api/types.ts";

export const MISSING = "—";
const MINUS = "−";

const VOL_QUANTITIES = new Set(["implied_vol", "realized_vol"]);
const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

const CONVENTIONS: Record<string, string> = {
  per_1vol_point: "per 1 vol pt",
  per_calendar_day: "per cal day",
};

function fixed(abs: number, decimals: number): string {
  return abs.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/** Real minus sign; "+" only when asked for (signed changes). */
export function formatNumber(value: number, decimals = 2, signed = false): string {
  const body = fixed(Math.abs(value), decimals);
  if (value < 0 && body !== fixed(0, decimals)) return `${MINUS}${body}`;
  return signed && value > 0 ? `+${body}` : body;
}

export function formatValue(
  value: number | null | undefined,
  unit: Unit,
  opts: { decimals?: number; signed?: boolean } = {},
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return MISSING;
  const signed = opts.signed ?? false;

  if (unit.scale === "decimal" && VOL_QUANTITIES.has(unit.quantity)) {
    return `${formatNumber(value * 100, opts.decimals ?? 2, signed)}%`;
  }
  if (unit.scale === "percent") return `${formatNumber(value, opts.decimals ?? 2, signed)}%`;
  if (unit.scale === "bps" || unit.quantity === "bps") {
    return `${formatNumber(value, opts.decimals ?? 1, signed)} bps`;
  }
  if (unit.quantity === "price_usd") {
    const n = formatNumber(value, opts.decimals ?? 2, signed);
    return n.startsWith(MINUS) || n.startsWith("+") ? `${n[0]}$${n.slice(1)}` : `$${n}`;
  }
  if (unit.quantity === "count") return formatNumber(value, 0, signed);
  return formatNumber(value, opts.decimals ?? 4, signed);
}

/** Axis labels: $12.5K, $1.2M (front_end_design.md §3). */
export function formatCompactUsd(value: number): string {
  const abs = Math.abs(value);
  const sign = value < 0 ? MINUS : "";
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(0)}`;
}

/** Header/axis label with the greek convention, e.g. "Vega (per 1 vol pt)". */
export function unitLabel(label: string, unit: Unit): string {
  if (!unit.convention) return label;
  return `${label} (${CONVENTIONS[unit.convention] ?? unit.convention.replaceAll("_", " ")})`;
}

/** "12 OCT, 15:17 UTC". */
export function formatUtc(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return MISSING;
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}, ${hh}:${mm} UTC`;
}

/** "6m ago", "2h 5m ago". */
export function formatAge(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return MISSING;
  if (seconds < 60) return `${Math.floor(seconds)}s ago`;
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ${m % 60}m ago`;
  return `${Math.floor(h / 24)}d ago`;
}
