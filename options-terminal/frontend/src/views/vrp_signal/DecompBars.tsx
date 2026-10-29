// Forecast decomposition (guide §6, §12.2): per tenor one horizontal bar of variance shares,
// baseline | seasonal | event, summing to 100%. The event segment sits at the right end, so the
// dashed hump line at 100% − event_hump_share is crossed exactly when the event share exceeds
// the threshold. Share colour is always paired with the word in the tooltip and the sr-only table.
import { useState, type KeyboardEvent, type PointerEvent } from "react";
import { SrTable } from "../../renderers/chart/SrTable.tsx";
import { AMBER, CYAN } from "../../renderers/chart/palette.ts";
import { linearScale } from "../../renderers/chart/scale.ts";
import { useWidth } from "../../renderers/chart/useWidth.ts";
import { fmtShare } from "../../format/vrp.ts";
import type { SignalRow } from "./logic.ts";

const PAD = { top: 6, right: 12, bottom: 22, left: 44 }; // left: room for the tenor labels
const ROW_H = 26;
const PARTS = [
  { key: "comp_baseline", word: "BASELINE", color: "var(--dim)" },
  { key: "comp_seasonal", word: "SEASONAL (DAY OF WEEK)", color: CYAN },
  { key: "comp_event", word: "EVENT", color: AMBER },
] as const;

export function DecompBars({ title, rows, humpShare, footer }: { title: string; rows: SignalRow[]; humpShare: number | null; footer: string }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const height = PAD.top + PAD.bottom + ROW_H * Math.max(1, rows.length);
  const sx = linearScale([0, 1], [PAD.left, Math.max(PAD.left + 1, width - PAD.right)]);
  const barH = ROW_H * 0.6;

  const move = (e: PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const i = Math.floor((e.clientY - rect.top - PAD.top) / ROW_H);
    setHover(i >= 0 && i < rows.length ? i : null);
  };
  const key = (e: KeyboardEvent) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const step = e.key === "ArrowDown" ? 1 : -1;
      setHover((h) => Math.min(rows.length - 1, Math.max(0, (h ?? 0) + step)));
    } else if (e.key === "Escape") setHover(null);
  };
  const hr = hover !== null ? rows[hover] : undefined;

  return (
    <div className="chart" ref={ref}>
      <div className="legend" aria-hidden="true">
        {PARTS.map((p) => (
          <span key={p.key} className="legend-item">
            <svg width="10" height="10"><rect width="10" height="10" fill={p.color} /></svg>
            {p.word}
          </span>
        ))}
        {humpShare !== null && (
          <span className="legend-item">
            <svg width="8" height="10"><line x1="4" x2="4" y1="0" y2="10" stroke="var(--paper)" strokeDasharray="2 2" /></svg>
            EVENT HUMP AT {fmtShare(humpShare)}
          </span>
        )}
      </div>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={title} tabIndex={0}
          onPointerMove={move} onPointerLeave={() => setHover(null)} onKeyDown={key}>
          {rows.map((r, i) => {
            const y = PAD.top + i * ROW_H + (ROW_H - barH) / 2;
            let acc = 0;
            return (
              <g key={r.tenor_days} opacity={hover === null || hover === i ? 1 : 0.7}>
                <text className="chart-text" x={PAD.left - 6} y={y + barH / 2 + 4} textAnchor="end">{r.tenor_days}d</text>
                {PARTS.map((p) => {
                  const v = r[p.key];
                  // A null share is a gap, never a zero-width segment presented as data (F-16).
                  if (v === null) return null;
                  const x0 = sx(acc);
                  acc += v;
                  return <rect key={p.key} x={x0} y={y} width={Math.max(0, sx(acc) - x0)} height={barH} fill={p.color} />;
                })}
                {r.event_hump && <text className="chart-text" x={sx(1) - 2} y={y - 1} textAnchor="end">HUMP</text>}
              </g>
            );
          })}
          {humpShare !== null && (
            <line x1={sx(1 - humpShare)} x2={sx(1 - humpShare)} y1={PAD.top} y2={height - PAD.bottom} stroke="var(--paper)" strokeDasharray="2 2" />
          )}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <text key={t} className="chart-text" x={sx(t)} y={height - 6} textAnchor="middle">{fmtShare(t)}</text>
          ))}
        </svg>
      )}
      {hr && (
        <div className="tooltip" style={{ left: Math.max(0, width - 200) }}>
          <div className="dim">TENOR {hr.tenor_days}d</div>
          {PARTS.map((p) => (
            <div key={p.key}>{p.word}: <span className="num">{fmtShare(hr[p.key])}</span></div>
          ))}
          {hr.event_hump && <div>EVENT HUMP</div>}
        </div>
      )}
      <SrTable
        caption={title}
        columns={["Tenor", ...PARTS.map((p) => p.word), "Event hump"]}
        rows={rows.map((r) => [`${r.tenor_days}d`, ...PARTS.map((p) => fmtShare(r[p.key])), r.event_hump ? "yes" : "no"])}
      />
      <div className="chart-footer dim">{footer}</div>
    </div>
  );
}
