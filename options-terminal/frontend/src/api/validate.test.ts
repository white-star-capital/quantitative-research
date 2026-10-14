import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { describe, it } from "node:test";
import { checkEnvelope, checkLatest, checkSlotList } from "./validate.ts";

// Structural values only, no market data; payload schemas and real fixtures arrive with M0.
const BASE = {
  envelope_version: "1",
  run_id: "0192f3a0-6b1e-7c3a-9a51-2f8e4d1c7b20",
  slot_id: "vol_surface",
  implementation_id: "vol_surface_svi",
  implementation_version: "2.1.0",
  scope: { underlying: "BTC" },
  status: "success",
  as_of: "2026-10-13T17:00:00Z",
  started_at: "2026-10-13T17:00:02Z",
  finished_at: "2026-10-13T17:00:09Z",
  inputs: {
    sources: [{ name: "options_chain", snapshot_at: "2026-10-13T17:00:00Z" }],
    hash: `sha256:${"0".repeat(64)}`,
  },
  code: { git_sha: "a1b2c3d" },
  warnings: [],
  payload_kind: "scalar_set",
  payload_version: "1",
  payload: { items: [] },
};

const FAILED = { ...BASE, status: "failed", payload: null, error: { code: "stale_input", message: "old" } };

describe("checkEnvelope", () => {
  it("accepts success and failed envelopes", () => {
    assert.equal(checkEnvelope(BASE).ok, true);
    assert.equal(checkEnvelope(FAILED).ok, true);
  });

  const invalid: [string, unknown][] = [
    ["failed with payload", { ...FAILED, payload: {} }],
    ["failed without error", { ...FAILED, error: undefined }],
    ["success with null payload", { ...BASE, payload: null }],
    ["success with error", { ...BASE, error: { code: "x", message: "y" } }],
    ["non-UTC timestamp", { ...BASE, as_of: "2026-10-13T17:00:00+02:00" }],
    ["unknown kind", { ...BASE, payload_kind: "heatmap" }],
    ["NaN in payload", { ...BASE, payload: { items: [], z: [Number.NaN] } }],
    ["payload shape", { ...BASE, payload: { items: [{ key: "a" }] } }],
    ["missing field", { ...BASE, run_id: undefined }],
    ["not an object", "<html>"],
  ];
  for (const [name, value] of invalid) {
    it(`rejects ${name}`, () => {
      assert.equal(checkEnvelope(value).ok, false);
    });
  }
});

describe("checkLatest", () => {
  const meta = {
    api_version: "1",
    role: "production",
    age_seconds: 412,
    max_age_seconds: 900,
    stale: false,
    served_at: "2026-10-13T17:06:52Z",
  };

  it("accepts envelope + meta", () => {
    assert.equal(checkLatest({ data: BASE, meta }).ok, true);
  });

  it("rejects bad meta", () => {
    assert.equal(checkLatest({ data: BASE, meta: { ...meta, stale: "no" } }).ok, false);
  });
});

describe("checkSlotList", () => {
  it("accepts the slot list shape and rejects bad ids", () => {
    const slot = {
      slot_id: "vol_surface",
      title: "Implied volatility surface",
      description: "",
      payload_kind: "surface",
      payload_version: "1",
      scope: { dimensions: ["underlying"], allowed: { underlying: ["BTC"] } },
      max_age_seconds: 900,
      production_impl: null,
      production_version: null,
    };
    assert.equal(checkSlotList({ data: [slot], meta: {} }).ok, true);
    assert.equal(checkSlotList({ data: [{ ...slot, slot_id: "Bad" }], meta: {} }).ok, false);
  });
});

// CONTRACTS §7 "Frontend render": every valid fixture passes, every invalid one fails.
// Runs against contracts/fixtures once M0 adds them.
describe("contract fixtures", () => {
  for (const kind of ["valid", "invalid"] as const) {
    const root = new URL(`../../../contracts/fixtures/${kind}/`, import.meta.url);
    const files = existsSync(root)
      ? readdirSync(root, { recursive: true }).filter((f) => f.endsWith(".json"))
      : [];
    for (const file of files) {
      it(`${kind}/${file}`, () => {
        // An invalid fixture may break JSON itself (NaN): the client rejects it at parse.
        let ok: boolean;
        try {
          ok = checkEnvelope(JSON.parse(readFileSync(new URL(file, root), "utf8"))).ok;
        } catch (exc) {
          if (!(exc instanceof SyntaxError)) throw exc;
          ok = false;
        }
        assert.equal(ok, kind === "valid");
      });
    }
  }
});
