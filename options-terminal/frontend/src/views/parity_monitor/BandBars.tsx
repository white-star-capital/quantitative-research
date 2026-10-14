// Parity band chart (guide §7.2 panel 2): for each strike a vertical bar from F_impl_bid to
// F_impl_ask in bp vs the hedge mid, with horizontal lines at the hedge bid and ask. A bar wholly
// above the ask is a conversion, wholly below the bid a reversal. Status colour (green = survives
// fees, yellow = gross only, dim = inside) is always paired with ▲/▼ and a word in the tooltip.
import { useState, type KeyboardEvent, type PointerEvent } from "react";
import { SrTable } from "../../renderers/chart/SrTable.tsx";
import { YAxis } from "../../renderers/chart/YAxis.tsx";
import { nearestIndex } from "../../renderers/chart/paths.ts";
import { extent, linearScale, niceTicks } from "../../renderers/chart/scale.ts";
import { useWidth } from "../../renderers/chart/useWidth.ts";
import { fmtBp } from "../../format/parity.ts";
import { fmtSpot } from "../../format/pricer.ts";
import type { BarStatus } from "./logic.ts";

export interface Bar {
  strike: number;
  lo: number;
  hi: number;
  status: BarStatus;
  side: "conversion" | "reversal" | null;
  confirmed: boolean;
}

const PAD = { top: 14, right: 8, bottom: 22, left: 8 };
const COLOR: Record<BarStatus, string> = { net: "var(--up)", gross: "var(--yellow)", inside: "var(--dim)" };
const WORD: Record<BarStatus, string> = { net: "NET OF FEES", gross: "GROSS ONLY", inside: "INSIDE" };

export function BandBars({
  title, asset, bars, hedgeBidBp, hedgeAskBp, footer,
}: {
  title: string;
  asset: string;
  bars: Bar[];
  hedgeBidBp: number | null;
  hedgeAskBp: number | null;
  footer: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const height = 240;
  const xs = bars.map((b) => b.strike);
  const yExt = extent([...bars.flatMap((b) => [b.lo, b.hi]), hedgeBidBp, hedgeAskBp, 0]) ?? [-1, 1];
  const pad = (yExt[1] - yExt[0]) * 0.08 || 1;
  const xExt = extent(xs) ?? [0, 1];
  const xPad = (xExt[1] - xExt[0]) * 0.04 || 1;
  const sx = linearScale([xExt[0] - xPad, xExt[1] + xPad], [PAD.left, Math.max(PAD.left + 1, width - PAD.right)]);
  const sy = linearScale([yExt[0] - pad, yExt[1] + pad], [height - PAD.bottom, PAD.top]);
  const barW = Math.max(2, Math.min(40, ((width - PAD.left - PAD.right) / Math.max(1, bars.length)) * 0.6));
  const yTicks = niceTicks(sy.domain[0], sy.domain[1], 4);
  const xTicks = niceTicks(xExt[0], xExt[1], width < 480 ? 3 : 6);

  const move = (e: PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setHover(bars.length ? nearestIndex(xs, e.clientX - rect.left, sx) : null);
  };
  const key = (e: KeyboardEvent) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const step = e.key === "ArrowRight" ? 1 : -1;
      setHover((h) => Math.min(bars.length - 1, Math.max(0, (h ?? 0) + step)));
    } else if (e.key === "Escape") setHover(null);
  };
  const hb = hover !== null ? bars[hover] : undefined;
  const mark = (b: Bar) => (b.side === "conversion" ? "▲ CONV" : b.side === "reversal" ? "▼ REV" : "");

  return (
    <div className="chart" ref={ref}>
      <div className="legend" aria-hidden="true">
        {(["net", "gross", "inside"] as BarStatus[]).map((s) => (
          <span key={s} className="legend-item">
            <svg width="10" height="10"><rect width="10" height="10" fill={COLOR[s]} /></svg>
            {WORD[s]}
          </span>
        ))}
        <span className="legend-item">
          <svg width="10" height="10"><circle cx="5" cy="5" r="3" fill="none" stroke="var(--paper)" strokeWidth={2} /></svg>
          CONFIRMED ON ORDER BOOKS
        </span>
      </div>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={title} tabIndex={0}
          onPointerMove={move} onPointerLeave={() => setHover(null)} onKeyDown={key}>
          <YAxis scale={sy} ticks={yTicks} width={width} format={(v) => fmtBp(v)} />
          {([["HEDGE ASK", hedgeAskBp], ["HEDGE BID", hedgeBidBp]] as const).map(([label, y]) =>
            y === null ? null : (
              <g key={label}>
                <line x1={PAD.left} x2={width - PAD.right} y1={sy(y)} y2={sy(y)} stroke="var(--cyan)" strokeDasharray="6 4" />
                <text className="chart-text" x={PAD.left + 2} y={sy(y) + (label === "HEDGE ASK" ? -3 : 11)}>{label}</text>
              </g>
            ),
          )}
          {bars.map((b, i) => {
            const x = sx(b.strike) - barW / 2;
            const top = sy(b.hi);
            const h = Math.max(2, sy(b.lo) - top);
            return (
              <g key={b.strike} opacity={hover === null || hover === i ? 1 : 0.7}>
                <rect x={x} y={top} width={barW} height={h} fill={COLOR[b.status]} />
                {b.confirmed && <circle cx={sx(b.strike)} cy={top - 6} r={4} fill="none" stroke="var(--paper)" strokeWidth={2} />}
              </g>
            );
          })}
          {xTicks.map((t) => (
            <text key={t} className="chart-text" x={sx(t)} y={height - 6} textAnchor="middle">{fmtSpot(asset, t)}</text>
          ))}
          {hb && <line x1={sx(hb.strike)} x2={sx(hb.strike)} y1={PAD.top} y2={height - PAD.bottom} stroke="var(--paper)" opacity={0.5} />}
        </svg>
      )}
      {hb && (
        <div className="tooltip" style={{ left: Math.min(Math.max(0, sx(hb.strike) + 10), Math.max(0, width - 200)) }}>
          <div className="dim">STRIKE {fmtSpot(asset, hb.strike)}</div>
          <div>F_impl bid: <span className="num">{fmtBp(hb.lo)}</span></div>
          <div>F_impl ask: <span className="num">{fmtBp(hb.hi)}</span></div>
          <div>{WORD[hb.status]} {mark(hb)}{hb.confirmed ? " · CONFIRMED" : ""}</div>
        </div>
      )}
      <SrTable
        caption={title}
        columns={["Strike", "F_impl bid vs hedge mid", "F_impl ask vs hedge mid", "Status"]}
        rows={bars.map((b) => [fmtSpot(asset, b.strike), fmtBp(b.lo), fmtBp(b.hi), `${WORD[b.status]} ${mark(b)}${b.confirmed ? " confirmed" : ""}`])}
      />
      <div className="chart-footer dim">{footer}</div>
    </div>
  );
}
