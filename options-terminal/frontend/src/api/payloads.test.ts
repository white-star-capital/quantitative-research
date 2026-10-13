import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { describe, it } from "node:test";
import { checkPayload, records, scalarMap, section, type Composite, type Table } from "./payloads.ts";

const U = { quantity: "implied_vol", scale: "decimal" };
const AX = { label: "k", unit: U, values: [-0.1, 0, 0.1] };

describe("checkPayload (F-5, C-4)", () => {
  it("accepts well-formed payloads with nulls", () => {
    assert.equal(checkPayload("curve", { x: AX, series: [{ key: "s", label: "s", unit: U, values: [null, 0.5, null] }] }).ok, true);
    assert.equal(checkPayload("surface", { x: AX, y: AX, z: { label: "z", unit: U, values: [[null, 1, 1], [1, 1, 1], [1, 1, null]] } }).ok, true);
    assert.equal(checkPayload("table", { columns: [{ key: "a", label: "A", type: "boolean" }], rows: [[true], [null]] }).ok, true);
  });

  const bad: [string, string, unknown][] = [
    ["curve length", "curve", { x: AX, series: [{ key: "s", label: "s", unit: U, values: [1] }] }],
    ["surface shape", "surface", { x: AX, y: AX, z: { label: "z", unit: U, values: [[1, 1, 1]] } }],
    ["table width", "table", { columns: [{ key: "a", label: "A", type: "number" }], rows: [[1, 2]] }],
    ["table type", "table", { columns: [{ key: "a", label: "A", type: "number" }], rows: [["1"]] }],
    ["unknown kind", "pie", {}],
    ["nested composite", "composite", { sections: [{ key: "c", label: "c", payload_kind: "composite", payload_version: "1", payload: { sections: [] } }] }],
  ];
  for (const [name, kind, payload] of bad) {
    it(`rejects ${name}`, () => assert.equal(checkPayload(kind, payload).ok, false));
  }
});

describe("synthetic bs_pricer run (math vectors)", () => {
  const path = new URL("../../../contracts/fixtures/pricing/evaluate_synthetic.json", import.meta.url);
  it("passes the payload checks and reads back", { skip: !existsSync(path) }, () => {
    const run = JSON.parse(readFileSync(path, "utf8")).run;
    assert.equal(checkPayload(run.payload_kind, run.payload).ok, true);
    const payload = run.payload as Composite;
    assert.equal(scalarMap(section(payload, "context")!).spot!.value, 100);
    assert.ok(records(section<Table>(payload, "chain")!).length > 100);
  });
});
