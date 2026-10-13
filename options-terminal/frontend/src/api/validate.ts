// Runtime guards for API responses (FRONTEND.md F-5). Anything that fails goes to the
// Unsupported state, never into a renderer.
import {
  PAYLOAD_KINDS,
  type Envelope,
  type LatestResponse,
  type Problem,
  type SlotSummary,
} from "./types.ts";

export type Checked<T> = { ok: true; value: T } | { ok: false; error: string };

const IDENTIFIER = /^[a-z][a-z0-9_]{2,63}$/;
const UTC = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?Z$/;
const UUID_V7 = /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const SEMVER = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;

class Invalid extends Error {}

type Obj = Record<string, unknown>;

function obj(v: unknown, path: string): Obj {
  if (typeof v !== "object" || v === null || Array.isArray(v)) {
    throw new Invalid(`${path}: expected object`);
  }
  return v as Obj;
}

function str(v: unknown, path: string, pattern?: RegExp): string {
  if (typeof v !== "string") throw new Invalid(`${path}: expected string`);
  if (pattern && !pattern.test(v)) throw new Invalid(`${path}: '${v}' has the wrong format`);
  return v;
}

function num(v: unknown, path: string): number {
  if (typeof v !== "number" || !Number.isFinite(v)) throw new Invalid(`${path}: expected number`);
  return v;
}

function bool(v: unknown, path: string): boolean {
  if (typeof v !== "boolean") throw new Invalid(`${path}: expected boolean`);
  return v;
}

function arr(v: unknown, path: string): unknown[] {
  if (!Array.isArray(v)) throw new Invalid(`${path}: expected array`);
  return v;
}

function message(v: unknown, path: string): void {
  const m = obj(v, path);
  str(m.code, `${path}.code`);
  str(m.message, `${path}.message`);
}

// JSON cannot carry NaN/Infinity, but a non-JSON parse path could; reject anything non-finite.
function assertFinite(v: unknown, path: string): void {
  if (typeof v === "number" && !Number.isFinite(v)) throw new Invalid(`${path}: non-finite`);
  if (Array.isArray(v)) v.forEach((x, i) => assertFinite(x, `${path}[${i}]`));
  else if (typeof v === "object" && v !== null) {
    for (const [k, x] of Object.entries(v)) assertFinite(x, `${path}.${k}`);
  }
}

function envelope(v: unknown, path: string): Envelope {
  const e = obj(v, path);
  if (e.envelope_version !== "1") throw new Invalid(`${path}.envelope_version: expected "1"`);
  str(e.run_id, `${path}.run_id`, UUID_V7);
  str(e.slot_id, `${path}.slot_id`, IDENTIFIER);
  str(e.implementation_id, `${path}.implementation_id`, IDENTIFIER);
  str(e.implementation_version, `${path}.implementation_version`, SEMVER);
  for (const [k, x] of Object.entries(obj(e.scope, `${path}.scope`))) {
    str(x, `${path}.scope.${k}`);
  }
  for (const f of ["as_of", "started_at", "finished_at"]) str(e[f], `${path}.${f}`, UTC);
  const inputs = obj(e.inputs, `${path}.inputs`);
  arr(inputs.sources, `${path}.inputs.sources`).forEach((s, i) => {
    const src = obj(s, `${path}.inputs.sources[${i}]`);
    str(src.name, `${path}.inputs.sources[${i}].name`);
    str(src.snapshot_at, `${path}.inputs.sources[${i}].snapshot_at`, UTC);
  });
  str(inputs.hash, `${path}.inputs.hash`, /^sha256:[0-9a-f]{64}$/);
  str(obj(e.code, `${path}.code`).git_sha, `${path}.code.git_sha`, /^[0-9a-f]{7,40}$/);
  arr(e.warnings, `${path}.warnings`).forEach((w, i) => message(w, `${path}.warnings[${i}]`));
  const kind = str(e.payload_kind, `${path}.payload_kind`);
  if (!(PAYLOAD_KINDS as readonly string[]).includes(kind)) {
    throw new Invalid(`${path}.payload_kind: unknown kind '${kind}'`);
  }
  str(e.payload_version, `${path}.payload_version`, /^[1-9]\d*$/);

  if (e.status === "success") {
    obj(e.payload, `${path}.payload`);
    if ("error" in e) throw new Invalid(`${path}.error: not allowed on a successful run`);
  } else if (e.status === "failed") {
    if (e.payload !== null) throw new Invalid(`${path}.payload: must be null on a failed run`);
    message(e.error, `${path}.error`);
  } else {
    throw new Invalid(`${path}.status: expected success or failed`);
  }
  assertFinite(e.payload, `${path}.payload`);
  return e as unknown as Envelope;
}

function run<T>(fn: () => T): Checked<T> {
  try {
    return { ok: true, value: fn() };
  } catch (err) {
    if (err instanceof Invalid) return { ok: false, error: err.message };
    throw err;
  }
}

export function checkEnvelope(v: unknown): Checked<Envelope> {
  return run(() => envelope(v, "envelope"));
}

export function checkLatest(v: unknown): Checked<LatestResponse> {
  return run(() => {
    const r = obj(v, "response");
    const data = envelope(r.data, "data");
    const m = obj(r.meta, "meta");
    str(m.api_version, "meta.api_version");
    if (m.role !== "production" && m.role !== "shadow") throw new Invalid("meta.role: invalid");
    num(m.age_seconds, "meta.age_seconds");
    num(m.max_age_seconds, "meta.max_age_seconds");
    bool(m.stale, "meta.stale");
    str(m.served_at, "meta.served_at", UTC);
    return { data, meta: m } as unknown as LatestResponse;
  });
}

export function checkSlotList(v: unknown): Checked<SlotSummary[]> {
  return run(() =>
    arr(obj(v, "response").data, "data").map((item, i) => {
      const p = `data[${i}]`;
      const s = obj(item, p);
      str(s.slot_id, `${p}.slot_id`, IDENTIFIER);
      str(s.title, `${p}.title`);
      str(s.description, `${p}.description`);
      str(s.payload_kind, `${p}.payload_kind`);
      str(s.payload_version, `${p}.payload_version`);
      const scope = obj(s.scope, `${p}.scope`);
      arr(scope.dimensions, `${p}.scope.dimensions`).forEach((d, j) =>
        str(d, `${p}.scope.dimensions[${j}]`, IDENTIFIER),
      );
      if (scope.allowed !== undefined) {
        for (const [k, vals] of Object.entries(obj(scope.allowed, `${p}.scope.allowed`))) {
          arr(vals, `${p}.scope.allowed.${k}`).forEach((x, j) =>
            str(x, `${p}.scope.allowed.${k}[${j}]`),
          );
        }
      }
      num(s.max_age_seconds, `${p}.max_age_seconds`);
      return s as unknown as SlotSummary;
    }),
  );
}

export function asProblem(v: unknown): Problem | null {
  const r = run(() => {
    const p = obj(v, "problem");
    str(p.type, "problem.type");
    str(p.title, "problem.title");
    num(p.status, "problem.status");
    return p as unknown as Problem;
  });
  return r.ok ? r.value : null;
}
