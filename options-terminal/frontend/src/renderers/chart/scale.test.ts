import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { extent, linearScale, niceTicks } from "./scale.ts";

describe("chart scale", () => {
  it("maps domain to range", () => {
    const s = linearScale([0, 10], [100, 0]);
    assert.equal(s(0), 100);
    assert.equal(s(5), 50);
    assert.equal(s(10), 0);
  });

  it("centres a degenerate domain", () => {
    assert.equal(linearScale([3, 3], [0, 100])(3), 50);
  });

  it("skips nulls in extent", () => {
    assert.deepEqual(extent([null, 0.4, null, 0.6]), [0.4, 0.6]);
    assert.equal(extent([null, null]), null);
  });

  it("produces round ticks", () => {
    assert.deepEqual(niceTicks(0, 1, 5), [0, 0.2, 0.4, 0.6, 0.8, 1]);
    assert.deepEqual(niceTicks(0.43, 0.71, 4), [0.5, 0.6, 0.7]);
    assert.deepEqual(niceTicks(0.43, 0.71, 6), [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]);
  });
});
