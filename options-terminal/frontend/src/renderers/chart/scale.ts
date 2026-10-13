// Linear scales and tick generation for hand-drawn SVG charts (front_end_design.md §5).

export interface LinearScale {
  (value: number): number;
  domain: [number, number];
  range: [number, number];
}

export function linearScale(domain: [number, number], range: [number, number]): LinearScale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;
  const f = ((v: number) => (span === 0 ? (r0 + r1) / 2 : r0 + ((v - d0) / span) * (r1 - r0))) as LinearScale;
  f.domain = domain;
  f.range = range;
  return f;
}

/** Min/max over finite values, ignoring nulls (null is a gap, never zero). */
export function extent(values: readonly (number | null)[]): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v === null || !Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo === Infinity ? null : [lo, hi];
}

/** About `count` round ticks (1, 2, 5 × 10^n steps) covering the domain. */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || count < 1) return [];
  if (lo === hi) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 5, 10].find((m) => m * mag >= raw) ?? 10) * mag;
  const ticks: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) {
    ticks.push(Number(t.toPrecision(12)));
  }
  return ticks;
}
