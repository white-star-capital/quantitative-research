// Payload kinds (CONTRACTS.md §3) and their runtime checks (FRONTEND F-5). Mirrors
// contracts/schemas/payloads/*/v1.json plus the C-4 shape rules; anything that fails lands in the
// Unsupported state, never in a renderer.
import type { Checked } from "./validate.ts";
import type { PayloadKind, Unit } from "./types.ts";

export type Cell = number | string | boolean | null;

export interface ScalarItem {
  key: string;
  label: string;
  value: number | boolean | null;
  unit: Unit;
  flags?: string[];
}
export interface ScalarSet {
  items: ScalarItem[];
}
export interface Axis {
  label: string;
  unit: Unit;
  values: number[];
}
export interface Series {
  key: string;
  label: string;
  unit: Unit;
  values: (number | null)[];
}
export interface Curve {
  x: Axis;
  series: Series[];
}
export interface SurfacePayload {
  x: Axis;
  y: Axis;
  z: { label: string; unit: Unit; values: (number | null)[][] };
}
export interface Timeseries {
  t: string[];
  series: Series[];
}
export type ColumnType = "number" | "string" | "timestamp" | "date" | "boolean";
export interface Column {
  key: string;
  label: string;
  type: ColumnType;
  unit?: Unit;
}
export interface Table {
  columns: Column[];
  rows: Cell[][];
}
export interface Section {
  key: string;
  label: string;
  payload_kind: PayloadKind;
  payload_version: string;
  payload: unknown;
}
export interface Composite {
  sections: Section[];
}

class Bad extends Error {}
type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const need = (cond: boolean, msg: string): void => {
  if (!cond) throw new Bad(msg);
};
const numOrNull = (v: unknown) => v === null || (typeof v === "number" && Number.isFinite(v));

function unit(v: unknown, path: string): void {
  need(isObj(v) && typeof v.quantity === "string" && typeof v.scale === "string", `${path}: unit`);
}

function axis(v: unknown, path: string): number {
  need(isObj(v) && Array.isArray(v.values), `${path}: axis`);
  const a = v as Obj;
  unit(a.unit, `${path}.unit`);
  const values = a.values as unknown[];
  need(values.length > 0 && values.every((x) => typeof x === "number" && Number.isFinite(x)), `${path}: values`);
  return values.length;
}

function seriesList(v: unknown, n: number, path: string): void {
  need(Array.isArray(v), `${path}: series`);
  const keys = new Set<string>();
  (v as unknown[]).forEach((s, i) => {
    need(isObj(s) && typeof s.key === "string" && Array.isArray(s.values), `${path}[${i}]`);
    const sr = s as Obj;
    unit(sr.unit, `${path}[${i}].unit`);
    need(!keys.has(sr.key as string), `${path}: duplicate key`);
    keys.add(sr.key as string);
    const values = sr.values as unknown[];
    need(values.length === n, `${path}[${i}]: ${values.length} values for ${n} x`);
    need(values.every(numOrNull), `${path}[${i}]: non-numeric value`);
  });
}

const CELL_OK: Record<ColumnType, (c: unknown) => boolean> = {
  number: (c) => typeof c === "number" && Number.isFinite(c),
  string: (c) => typeof c === "string",
  timestamp: (c) => typeof c === "string",
  date: (c) => typeof c === "string",
  boolean: (c) => typeof c === "boolean",
};

function check(kind: string, p: unknown, path: string, depth: number): void {
  need(isObj(p), `${path}: payload`);
  const o = p as Obj;
  switch (kind) {
    case "scalar_set":
      need(Array.isArray(o.items), `${path}: items`);
      (o.items as unknown[]).forEach((it, i) => {
        need(isObj(it) && typeof it.key === "string", `${path}.items[${i}]`);
        const v = (it as Obj).value;
        need(numOrNull(v) || typeof v === "boolean", `${path}.items[${i}].value`);
        unit((it as Obj).unit, `${path}.items[${i}].unit`);
      });
      return;
    case "curve":
      seriesList(o.series, axis(o.x, `${path}.x`), `${path}.series`);
      return;
    case "timeseries": {
      need(Array.isArray(o.t) && (o.t as unknown[]).every((t) => typeof t === "string"), `${path}.t`);
      const t = o.t as string[];
      need(t.every((v, i) => i === 0 || Date.parse(v) > Date.parse(t[i - 1] as string)), `${path}.t: not increasing`);
      seriesList(o.series, t.length, `${path}.series`);
      return;
    }
    case "surface": {
      const nx = axis(o.x, `${path}.x`);
      const ny = axis(o.y, `${path}.y`);
      need(isObj(o.z) && Array.isArray((o.z as Obj).values), `${path}.z`);
      const rows = (o.z as Obj).values as unknown[];
      need(rows.length === ny, `${path}.z: ${rows.length} rows for ${ny} y`);
      rows.forEach((r, i) => need(Array.isArray(r) && r.length === nx && r.every(numOrNull), `${path}.z[${i}]`));
      return;
    }
    case "table": {
      need(Array.isArray(o.columns) && Array.isArray(o.rows), `${path}: columns/rows`);
      const cols = o.columns as Column[];
      cols.forEach((c, i) => need(isObj(c) && typeof c.key === "string" && c.type in CELL_OK, `${path}.columns[${i}]`));
      (o.rows as unknown[]).forEach((r, i) => {
        need(Array.isArray(r) && r.length === cols.length, `${path}.rows[${i}]: width`);
        (r as unknown[]).forEach((cell, j) => {
          const col = cols[j] as Column;
          need(cell === null || CELL_OK[col.type](cell), `${path}.rows[${i}].${col.key}`);
        });
      });
      return;
    }
    case "composite":
      need(depth === 0, `${path}: composite cannot nest`);
      need(Array.isArray(o.sections) && (o.sections as unknown[]).length > 0, `${path}.sections`);
      for (const s of o.sections as unknown[]) {
        need(isObj(s) && typeof s.key === "string" && s.payload_version === "1", `${path}: section`);
        check(String((s as Obj).payload_kind), (s as Obj).payload, `${path}.${(s as Obj).key}`, depth + 1);
      }
      return;
    default:
      throw new Bad(`${path}: unknown kind ${kind}`);
  }
}

export function checkPayload(kind: string, payload: unknown): Checked<unknown> {
  try {
    check(kind, payload, kind, 0);
    return { ok: true, value: payload };
  } catch (err) {
    if (err instanceof Bad) return { ok: false, error: err.message };
    throw err;
  }
}

// ---- readers ------------------------------------------------------------------------------

export function section<T>(composite: Composite, key: string): T | null {
  const s = composite.sections.find((x) => x.key === key);
  return s ? (s.payload as T) : null;
}

export function records(table: Table): Record<string, Cell>[] {
  return table.rows.map((row) => Object.fromEntries(table.columns.map((c, i) => [c.key, row[i] ?? null])));
}

export function scalarMap(set: ScalarSet): Record<string, ScalarItem> {
  return Object.fromEntries(set.items.map((i) => [i.key, i]));
}
