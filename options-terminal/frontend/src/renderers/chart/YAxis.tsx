import type { LinearScale } from "./scale.ts";

// Gridlines in --bar-track, zero line in --dim, 11px dim tick text; single y-axis only
// (front_end_design.md §5).
export function YAxis({
  scale,
  ticks,
  width,
  format,
}: {
  scale: LinearScale;
  ticks: number[];
  width: number;
  format: (v: number) => string;
}) {
  return (
    <g aria-hidden="true">
      {ticks.map((t) => (
        <g key={t}>
          <line
            x1={0}
            x2={width}
            y1={scale(t)}
            y2={scale(t)}
            stroke={t === 0 ? "var(--dim)" : "var(--bar-track)"}
            strokeWidth={1}
          />
          <text className="chart-text" x={width - 2} y={scale(t) - 3} textAnchor="end">
            {format(t)}
          </text>
        </g>
      ))}
    </g>
  );
}
