import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { heat } from "./palette.ts";
import { bandPath, linePath, nearestIndex } from "./paths.ts";
import { linearScale } from "./scale.ts";

describe("chart paths", () => {
  const sx = linearScale([0, 3], [0, 300]);
  const sy = linearScale([0, 1], [100, 0]);
  it("breaks lines at nulls instead of drawing zero", () => {
    assert.equal(linePath([0, 1, 2, 3], [0.5, null, 0.5, 1], sx, sy), "M0.0,50.0M200.0,50.0L300.0,0.0");
  });
  it("finds the nearest x", () => {
    assert.equal(nearestIndex([0, 1, 2, 3], 140, sx), 1);
  });
  it("heat scale runs from track to amber", () => {
    assert.equal(heat(1), "rgb(251,139,30)");
    assert.notEqual(heat(0), heat(1));
  });
});

describe("bandPath", () => {
  it("fills between lo and hi and breaks on nulls", () => {
    const sx = linearScale([0, 3], [0, 300]);
    const sy = linearScale([0, 1], [100, 0]);
    assert.equal(
      bandPath([0, 1, 2, 3], [0.2, 0.2, null, 0.4], [0.6, 0.8, 0.5, 0.6], sx, sy),
      "M0.0,40.0L100.0,20.0L100.0,80.0L0.0,80.0Z",
    );
    assert.equal(bandPath([0, 1], [null, null], [1, 1], sx, sy), "");
  });
});
