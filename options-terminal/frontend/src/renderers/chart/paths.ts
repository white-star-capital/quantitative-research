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

/** Closed SVG path(s) filling between lo and hi; a null on either side ends a segment. */
export function bandPath(xs: number[], lo: (number | null)[], hi: (number | null)[], sx: LinearScale, sy: LinearScale): string {
  let d = "";
  let seg: number[] = [];
  const flush = () => {
    if (seg.length >= 2) {
      const top = seg.map((i, j) => `${j ? "L" : "M"}${sx(xs[i] as number).toFixed(1)},${sy(hi[i] as number).toFixed(1)}`).join("");
      const bottom = [...seg].reverse().map((i) => `L${sx(xs[i] as number).toFixed(1)},${sy(lo[i] as number).toFixed(1)}`).join("");
      d += `${top}${bottom}Z`;
    }
    seg = [];
  };
  xs.forEach((_, i) => {
    const a = lo[i];
    const b = hi[i];
    if (a == null || b == null || !Number.isFinite(a) || !Number.isFinite(b)) flush();
    else seg.push(i);
  });
  flush();
  return d;
}
