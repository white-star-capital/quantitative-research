// VRP signal logic (RV_FORECAST_IMPLEMENTATION_GUIDE.md §10, §12). Pure functions only: section
// readers for the composite payload, the staleness gate and the band-discipline / event-hump
// chip rules. The model decides `rich`, `event_hump` and `signal_ok`; the view never recomputes
// them, it only refuses to show them as live when the run or its RV input is stale (§10).
import type { Cell, Composite, Curve, ScalarItem, ScalarSet, Table, Timeseries } from "../../api/payloads.ts";
import { records, scalarMap, section } from "../../api/payloads.ts";
import type { SlotSummary } from "../../api/types.ts";
import { fmtShare } from "../../format/vrp.ts";

export const VIEW_ID = "vrp_signal";

/** Monitor columns for the 30d row. "VRP fcst" is IV minus the RV forecast, distinct from the
 *  pricer's own "VRP 30d" (IV minus trailing RV) in the same header. */
export const VRP_COLUMNS = ["VRP fcst 30d", "VRP z 30d", "VRP read 30d"];

/** The catalog's VRP slot, found by its view id, never by a hard-coded slot id (F-2). */
export function findVrpSlot(slots: readonly SlotSummary[]): SlotSummary | null {
  return slots.find((s) => s.display?.default_view === VIEW_ID) ?? null;
}

/** Whether the slot serves this underlying, so other slots' assets never request a 400. */
export function vrpCovers(slot: SlotSummary, underlying: string): boolean {
  return slot.scope.allowed?.underlying?.includes(underlying) ?? false;
}

// ---- §12.3 signal rows ---------------------------------------------------------------------

export interface SignalRow {
  tenor_days: number;
  iv_impl: number | null;
  rv_forecast: number | null;
  rv_low: number | null;
  rv_high: number | null;
  vrp: number | null;
  vrp_z: number | null;
  comp_baseline: number | null;
  comp_seasonal: number | null;
  comp_event: number | null;
  event_in_horizon: boolean;
  /** null when `iv_impl` is null: no richness call without an implied vol. */
  rich: boolean | null;
  event_hump: boolean;
  signal_ok: boolean;
  flags: string[];
}

const num = (v: Cell | undefined): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const bool = (v: Cell | undefined): boolean | null => (typeof v === "boolean" ? v : null);

/** Signal rows by ascending tenor. A missing boolean reads as the cautious value (false). */
export function signalRows(payload: Composite | null): SignalRow[] {
  const t = payload ? section<Table>(payload, "signal") : null;
  if (!t) return [];
  return records(t)
    .filter((r) => num(r.tenor_days) !== null)
    .map((r) => ({
      tenor_days: r.tenor_days as number,
      iv_impl: num(r.iv_impl),
      rv_forecast: num(r.rv_forecast),
      rv_low: num(r.rv_low),
      rv_high: num(r.rv_high),
      vrp: num(r.vrp),
      vrp_z: num(r.vrp_z),
      comp_baseline: num(r.comp_baseline),
      comp_seasonal: num(r.comp_seasonal),
      comp_event: num(r.comp_event),
      event_in_horizon: bool(r.event_in_horizon) ?? false,
      rich: bool(r.rich),
      event_hump: bool(r.event_hump) ?? false,
      signal_ok: bool(r.signal_ok) ?? false,
      flags: typeof r.flags === "string" && r.flags ? r.flags.split(",").map((f) => f.trim()).filter(Boolean) : [],
    }))
    .sort((a, b) => a.tenor_days - b.tenor_days);
}

export function rowAt(rows: SignalRow[], tenor: number): SignalRow | null {
  return rows.find((r) => r.tenor_days === tenor) ?? null;
}

// ---- §12.4 context -------------------------------------------------------------------------

export function contextItems(payload: Composite | null): Record<string, ScalarItem> {
  const set = payload ? section<ScalarSet>(payload, "context") : null;
  return set ? scalarMap(set) : {};
}

export function ctxNum(ctx: Record<string, ScalarItem>, key: string): number | null {
  const v = ctx[key]?.value;
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// ---- §12.5 cone and §12.6 history ----------------------------------------------------------

export type Values = (number | null)[];

/** Curve series by key; a missing series is all null (a gap), never zeros. */
export function cone(payload: Composite | null): { x: number[]; s: (key: string) => Values } | null {
  const c = payload ? section<Curve>(payload, "cone") : null;
  if (!c) return null;
  return { x: c.x.values, s: (key) => c.series.find((x) => x.key === key)?.values ?? c.x.values.map(() => null) };
}

/** ATM IV markers on the cone: only where the model priced a tenor (sparse, §12.5). */
export function ivPoints(x: number[], atmIv: Values): { x: number; y: number }[] {
  return x.flatMap((xi, i) => {
    const y = atmIv[i];
    return typeof y === "number" ? [{ x: xi, y }] : [];
  });
}

/** Timeseries with `t` as epoch ms for the chart's linear x axis. */
export function history(payload: Composite | null): { t: number[]; s: (key: string) => Values } | null {
  const h = payload ? section<Timeseries>(payload, "history") : null;
  if (!h || h.t.length === 0) return null;
  return { t: h.t.map((v) => Date.parse(v)), s: (key) => h.series.find((x) => x.key === key)?.values ?? h.t.map(() => null) };
}

export function hasValues(values: Values): boolean {
  return values.some((v) => v !== null);
}

// ---- §12.7 events --------------------------------------------------------------------------

export interface EventRow {
  ts: string;
  type: string;
  scope: string;
  window_before: number | null;
  window_after: number | null;
  horizon_day: number | null;
  source: string;
}

export function events(payload: Composite | null): EventRow[] {
  const t = payload ? section<Table>(payload, "events") : null;
  if (!t) return [];
  return records(t).map((r) => ({
    ts: String(r.ts ?? ""),
    type: String(r.type ?? ""),
    scope: String(r.scope ?? ""),
    window_before: num(r.window_before),
    window_after: num(r.window_after),
    horizon_day: num(r.horizon_day),
    source: String(r.source ?? ""),
  }));
}

// ---- §10 staleness gate --------------------------------------------------------------------

export interface Gate {
  open: boolean;
  reason: string | null;
}

/** The signal shows only when the run is fresh (meta.stale, from the API) and the model says
 *  its RV input and event calendar are current (`signal_enabled`, §12.4). A missing flag keeps
 *  the gate shut. */
export function gate(stale: boolean, ctx: Record<string, ScalarItem>): Gate {
  if (stale) return { open: false, reason: "RUN IS STALE" };
  const enabled = ctx.signal_enabled?.value;
  if (enabled === true) return { open: true, reason: null };
  if (enabled === false) {
    const flags = ctx.signal_enabled?.flags ?? [];
    if (flags.includes("rv_input_lagging")) return { open: false, reason: "RV INPUT LAGGING" };
    if (flags.includes("event_calendar_incomplete")) return { open: false, reason: "EVENT CALENDAR INCOMPLETE" };
    return { open: false, reason: "SIGNAL DISABLED BY THE MODEL" };
  }
  return { open: false, reason: "SIGNAL STATE MISSING" };
}

export type Tone = "good" | "mid" | "bad" | "dim";

export interface Chip {
  label: string;
  tone: Tone;
  title: string;
}

/** Band discipline and the event-hump rule (§10): RICH only when IV clears the upper cone and
 *  the event share is under the threshold; a hump replaces RICH and is never read as premium.
 *  With the gate shut every chip is dim, never green. */
export function chips(row: SignalRow, g: Gate, humpShare: number | null): Chip[] {
  const out: Chip[] = [];
  const share = `event share ${fmtShare(row.comp_event)}${humpShare === null ? "" : ` > ${fmtShare(humpShare)}`}`;
  if (row.rich === true && !row.event_hump) {
    out.push({ label: "RICH", tone: g.open ? "good" : "dim", title: g.open ? "ATM IV above the 0.9 forecast cone (band discipline)" : `Disabled: ${g.reason}` });
  }
  if (row.event_hump) {
    const why = row.rich === true ? "IV clears the upper cone, but " : "";
    out.push({ label: "EVENT HUMP", tone: g.open ? "mid" : "dim", title: `${why}${share}: the market may be pricing a known catalyst. Not premium.` });
  } else if (row.event_in_horizon) {
    out.push({ label: "EVENT", tone: "dim", title: "An event window overlaps this tenor" });
  }
  return out;
}

const NOT_OK_REASONS: [(r: SignalRow) => boolean, string][] = [
  [(r) => r.iv_impl === null, "no ATM IV"],
  [(r) => r.vrp_z === null, "no z-score"],
  [(r) => r.event_hump, "event hump"],
];

/** Per-row signal status: live only when the gate is open and the model's `signal_ok` holds. */
export function signalStatus(row: SignalRow, g: Gate): { label: string; tone: Tone; title: string } {
  if (!g.open) return { label: "DISABLED", tone: "dim", title: g.reason ?? "" };
  if (row.signal_ok) return { label: "SIGNAL", tone: "good", title: "z-score and ATM IV present, no event hump" };
  const why = NOT_OK_REASONS.filter(([test]) => test(row)).map(([, w]) => w);
  return { label: "NO SIGNAL", tone: "dim", title: why.length ? why.join(", ") : "model signal_ok is false" };
}

/** The z cell: "—" with the reason when null (VRP-A-5); the DVOL basis noted when scored. */
export function zTitle(row: SignalRow): string {
  if (row.vrp_z === null) {
    return row.flags.includes("z_insufficient_history")
      ? "Insufficient history for the trailing 90-day window (VRP-A-5)"
      : "no IV history (VRP-A-5)";
  }
  return row.flags.includes("z_basis_dvol")
    ? "Scored on DVOL − RV forecast 30d against its trailing 90 days, not on this row's VRP (VRP-A-5)"
    : "Trailing 90-day z-score";
}
