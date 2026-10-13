// Surface heatmap (FRONTEND §4.3): sequential single-hue scale with legend; null cells are blank,
// never interpolated. Hover a cell for its value; sr-only table with the same data.
import { useState } from "react";
import { SrTable } from "./SrTable.tsx";
import { heat } from "./palette.ts";
import { extent } from "./scale.ts";
import { useWidth } from "./useWidth.ts";

export interface Props {
  title: string;
  x: number[];
  y: number[];
  z: (number | null)[][];
  xLabel: string;
  yLabel: string;
  formatX: (v: number) => string;
  formatY: (v: number) => string;
  formatZ: (v: number) => string;
  footer: string;
}

const LEFT = 52;
const ROW = 18;

export function Heatmap(p: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<[number, number] | null>(null);
  const zs = p.z.flat();
  const [lo, hi] = extent(zs) ?? [0, 1];
  const span = hi - lo || 1;
  const cellW = Math.max(1, (width - LEFT) / Math.max(1, p.x.length));
  const height = p.y.length * ROW + 22;
  const every = Math.max(1, Math.ceil(p.x.length / Math.max(2, Math.floor((width - LEFT) / 56))));
  const hv = hover ? p.z[hover[0]]?.[hover[1]] : undefined;

  return (
    <div className="chart" ref={ref}>
      <div className="legend" aria-hidden="true">
        <span className="dim">{p.formatZ(lo)}</span>
        <svg width="120" height="8">
          {Array.from({ length: 24 }, (_, i) => <rect key={i} x={i * 5} y={0} width={5} height={8} fill={heat(i / 23)} />)}
        </svg>
        <span className="dim">{p.formatZ(hi)}</span>
        <span className="dim">· BLANK = NO NODE DATA</span>
      </div>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={p.title} onPointerLeave={() => setHover(null)}>
          {p.y.map((yv, r) => (
            <g key={yv}>
              <text className="chart-text" x={LEFT - 4} y={r * ROW + 13} textAnchor="end">{p.formatY(yv)}</text>
              {p.x.map((_, c) => {
                const v = p.z[r]?.[c];
                return v === null || v === undefined ? null : (
                  <rect
                    key={c}
                    x={LEFT + c * cellW}
                    y={r * ROW + 1}
                    width={Math.max(1, cellW - 1)}
                    height={ROW - 2}
                    fill={heat((v - lo) / span)}
                    onPointerEnter={() => setHover([r, c])}
                  />
                );
              })}
            </g>
          ))}
          {p.x.map((xv, c) => (c % every === 0 ? (
            <text key={xv} className="chart-text" x={LEFT + c * cellW + cellW / 2} y={height - 6} textAnchor="middle">{p.formatX(xv)}</text>
          ) : null))}
        </svg>
      )}
      {hover && hv != null && (
        <div className="tooltip" style={{ left: Math.min(LEFT + hover[1] * cellW + 10, Math.max(0, width - 180)) }}>
          <div className="dim">{p.yLabel}: {p.formatY(p.y[hover[0]] as number)} · {p.xLabel}: {p.formatX(p.x[hover[1]] as number)}</div>
          <div className="num">{p.formatZ(hv)}</div>
        </div>
      )}
      <SrTable
        caption={p.title}
        columns={[p.yLabel, ...p.x.map(p.formatX)]}
        rows={p.y.map((yv, r) => [p.formatY(yv), ...p.x.map((_, c) => { const v = p.z[r]?.[c]; return v == null ? "—" : p.formatZ(v); })])}
      />
      <div className="chart-footer dim">{p.footer}</div>
    </div>
  );
}
