import type { LinearScale } from "./scale.ts";

/** SVG path for a series; nulls break the line (a gap, never zero). */
export function linePath(xs: number[], ys: (number | null)[], sx: LinearScale, sy: LinearScale): string {
  let d = "";
  let pen = false;
  ys.forEach((y, i) => {
    const x = xs[i];
    if (y === null || x === undefined || !Number.isFinite(y)) {
      pen = false;
      return;
    }
    d += `${pen ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`;
    pen = true;
  });
  return d;
}

/** Index of the x value nearest to `px` in pixel space. */
export function nearestIndex(xs: number[], px: number, sx: LinearScale): number {
  let best = 0;
  let dist = Infinity;
  xs.forEach((x, i) => {
    const d = Math.abs(sx(x) - px);
    if (d < dist) {
      dist = d;
      best = i;
    }
  });
  return best;
}
