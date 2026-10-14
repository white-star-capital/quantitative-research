// Line chart per front_end_design.md §5: measured width, 2px lines, one y-axis, crosshair with
// tooltip (mouse and keyboard), legend for ≥2 series, direct labels on key points (markers,
// reference lines), sr-only table, footer.
import { useState, type KeyboardEvent, type PointerEvent } from "react";
import { SrTable } from "./SrTable.tsx";
import { YAxis } from "./YAxis.tsx";
import { SERIES_STYLES, type SeriesStyle } from "./palette.ts";
import { bandPath, linePath, nearestIndex } from "./paths.ts";
import { extent, linearScale, niceTicks } from "./scale.ts";
import { useWidth } from "./useWidth.ts";

export interface LineSeries {
  key: string;
  label: string;
  values: (number | null)[];
  style?: SeriesStyle;
}

export interface Props {
  title: string;
  x: number[];
  series: LineSeries[];
  xLabel: string;
  formatX: (v: number) => string;
  formatY: (v: number) => string;
  footer: string;
  height?: number;
  /** Horizontal reference lines, e.g. RV 30d (dashed, dim). */
  refLines?: { y: number; label: string }[];
  /** Vertical markers with direct labels, e.g. SPOT, K, BE. */
  markers?: { x: number; label: string }[];
  /** Background silhouette scaled to the plot (no second axis), e.g. a density. */
  silhouette?: { values: number[]; label: string };
  /** Shade the area between the first series and zero: green above, red below (P&L). */
  pnlFill?: boolean;
  /** Neutral shaded range between two value arrays (e.g. a bid–ask band); nulls break it. */
  band?: { lo: (number | null)[]; hi: (number | null)[]; label: string };
  /** Data points (≥ 8px, 2px black ring) with optional vertical error bars, e.g. smile nodes. */
  points?: { x: number; y: number; lo?: number | null; hi?: number | null; color: string; dim?: boolean }[];
}

const PAD = { top: 12, right: 8, bottom: 22, left: 8 };

export function LineChart(p: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const height = p.height ?? 220;
  const series = p.series.slice(0, SERIES_STYLES.length);
  const pts = p.points ?? [];
  const all = series
    .flatMap((s) => s.values)
    .concat((p.refLines ?? []).map((r) => r.y))
    .concat(pts.flatMap((q) => [q.y, q.lo ?? null, q.hi ?? null]))
    .concat(p.band ? [...p.band.lo, ...p.band.hi] : []);
  const yExt = extent(all) ?? [0, 1];
  const xExt = extent(p.x) ?? [0, 1];
  const pad = (yExt[1] - yExt[0]) * 0.08 || Math.abs(yExt[0]) * 0.1 || 1;
  const sx = linearScale(xExt, [PAD.left, Math.max(PAD.left + 1, width - PAD.right)]);
  const sy = linearScale([yExt[0] - pad, yExt[1] + pad], [height - PAD.bottom, PAD.top]);
  const yTicks = niceTicks(sy.domain[0], sy.domain[1], 4);
  const xTicks = niceTicks(xExt[0], xExt[1], width < 480 ? 3 : 6);

  const move = (e: PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setHover(nearestIndex(p.x, e.clientX - rect.left, sx));
  };
  const key = (e: KeyboardEvent) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const step = e.key === "ArrowRight" ? 1 : -1;
      setHover((h) => Math.min(p.x.length - 1, Math.max(0, (h ?? 0) + step)));
    } else if (e.key === "Escape") setHover(null);
  };

  const sil = p.silhouette;
  const silMax = sil ? Math.max(...sil.values.filter(Number.isFinite), 0) : 0;
  const silPath = sil && silMax > 0
    ? `M${sx(p.x[0] as number)},${height - PAD.bottom}` +
      sil.values.map((v, i) => `L${sx(p.x[i] as number).toFixed(1)},${(height - PAD.bottom - (v / silMax) * (height - PAD.top - PAD.bottom) * 0.4).toFixed(1)}`).join("") +
      `L${sx(p.x[p.x.length - 1] as number)},${height - PAD.bottom}Z`
    : null;
  const first = series[0];
  const zeroY = sy(0);
  const hx = hover !== null ? p.x[hover] : undefined;

  return (
    <div className="chart" ref={ref}>
      {series.length >= 2 && (
        <div className="legend" aria-hidden="true">
          {series.map((s, i) => (
            <span key={s.key} className="legend-item">
              <svg width="18" height="8"><line x1="0" x2="18" y1="4" y2="4" stroke={(s.style ?? SERIES_STYLES[i])!.color} strokeWidth={2} strokeDasharray={(s.style ?? SERIES_STYLES[i])!.dash} /></svg>
              {s.label}
            </span>
          ))}
        </div>
      )}
      {width > 0 && (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={p.title}
          tabIndex={0}
          onPointerMove={move}
          onPointerLeave={() => setHover(null)}
          onKeyDown={key}
        >
          <YAxis scale={sy} ticks={yTicks} width={width} format={p.formatY} />
          {silPath && <path d={silPath} fill="#8d887c" opacity={0.18} />}
          {p.band && <path d={bandPath(p.x, p.band.lo, p.band.hi, sx, sy)} fill="#8d887c" opacity={0.22} />}
          {p.pnlFill && first && (
            <>
              <clipPath id={`above-${p.title}`}><rect x={0} y={0} width={width} height={Math.max(0, zeroY)} /></clipPath>
              <clipPath id={`below-${p.title}`}><rect x={0} y={zeroY} width={width} height={Math.max(0, height - zeroY)} /></clipPath>
              {(["above", "below"] as const).map((side) => (
                <path
                  key={side}
                  d={`${linePath(p.x, first.values, sx, sy)}L${sx(p.x[p.x.length - 1] as number)},${zeroY}L${sx(p.x[0] as number)},${zeroY}Z`}
                  fill={side === "above" ? "#00d964" : "#ff3b30"}
                  opacity={0.14}
                  clipPath={`url(#${side}-${p.title})`}
                />
              ))}
            </>
          )}
          {(p.refLines ?? []).map((r) => (
            <g key={r.label}>
              <line x1={PAD.left} x2={width - PAD.right} y1={sy(r.y)} y2={sy(r.y)} stroke="#8d887c" strokeDasharray="4 4" />
              <text className="chart-text" x={PAD.left + 2} y={sy(r.y) - 3}>{r.label}</text>
            </g>
          ))}
          {(p.markers ?? []).map((m) => {
            const mx = sx(m.x);
            const flip = mx > width - 60;
            return (
              <g key={m.label}>
                <line x1={mx} x2={mx} y1={PAD.top} y2={height - PAD.bottom} stroke="#8d887c" strokeDasharray="2 3" />
                <text className="chart-text" x={flip ? mx - 3 : mx + 3} y={PAD.top + 9} textAnchor={flip ? "end" : "start"}>{m.label}</text>
              </g>
            );
          })}
          {series.map((s, i) => {
            const style = s.style ?? SERIES_STYLES[i]!;
            return <path key={s.key} d={linePath(p.x, s.values, sx, sy)} fill="none" stroke={style.color} strokeWidth={2} strokeDasharray={style.dash} />;
          })}
          {pts.map((q, i) => (
            <g key={i} opacity={q.dim ? 0.45 : 1}>
              {q.lo != null && q.hi != null && (
                <line x1={sx(q.x)} x2={sx(q.x)} y1={sy(q.lo)} y2={sy(q.hi)} stroke={q.color} strokeWidth={1} />
              )}
              <circle cx={sx(q.x)} cy={sy(q.y)} r={4} fill={q.color} stroke="#000" strokeWidth={2} />
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={t} className="chart-text" x={sx(t)} y={height - 6} textAnchor="middle">{p.formatX(t)}</text>
          ))}
          {hx !== undefined && (
            <g aria-hidden="true">
              <line x1={sx(hx)} x2={sx(hx)} y1={PAD.top} y2={height - PAD.bottom} stroke="#f2efe6" strokeWidth={1} opacity={0.5} />
              {series.map((s, i) => {
                const v = s.values[hover as number];
                return v === null || v === undefined ? null : (
                  <circle key={s.key} cx={sx(hx)} cy={sy(v)} r={4} fill={(s.style ?? SERIES_STYLES[i])!.color} stroke="#000" strokeWidth={2} />
                );
              })}
            </g>
          )}
        </svg>
      )}
      {hx !== undefined && (
        <div className="tooltip" style={{ left: Math.min(Math.max(0, sx(hx) + 10), Math.max(0, width - 180)) }}>
          <div className="dim">{p.xLabel}: {p.formatX(hx)}</div>
          {series.map((s) => (
            <div key={s.key}>
              {s.label}: <span className="num">{s.values[hover as number] == null ? "—" : p.formatY(s.values[hover as number] as number)}</span>
            </div>
          ))}
        </div>
      )}
      <SrTable
        caption={p.title}
        columns={[p.xLabel, ...series.map((s) => s.label), ...(p.band ? [`${p.band.label} low`, `${p.band.label} high`] : [])]}
        rows={p.x.map((x, i) => [
          p.formatX(x),
          ...series.map((s) => (s.values[i] == null ? "—" : p.formatY(s.values[i] as number))),
          ...(p.band ? [p.band.lo, p.band.hi].map((v) => (v[i] == null ? "—" : p.formatY(v[i] as number))) : []),
        ])}
      />
      <div className="chart-footer dim">{p.footer}</div>
    </div>
  );
}
