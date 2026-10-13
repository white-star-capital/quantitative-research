// Vol tab (guide §7.5): term structure vs RV, smile overlay with nodes and bid–ask error bars,
// surface heatmap, skew table.
import { useMemo, useState } from "react";
import type { Composite, Curve, SurfacePayload, Table } from "../../api/payloads.ts";
import { section } from "../../api/payloads.ts";
import { fmtDte, fmtExpiry, fmtIv, fmtVp } from "../../format/pricer.ts";
import { formatNumber } from "../../format/units.ts";
import { DataTable } from "../../renderers/DataTable.tsx";
import { SurfaceHeatmap } from "../../renderers/Charts.tsx";
import { LineChart } from "../../renderers/chart/LineChart.tsx";
import { SERIES_STYLES } from "../../renderers/chart/palette.ts";
import { chainOf, expiriesOf } from "./logic.ts";
import { contextOf, num } from "./shared.ts";

type XMode = "k" | "pct" | "strike";

export function VolTab({ payload }: { payload: Composite }) {
  const ts = section<Curve>(payload, "term_structure");
  const smile = section<Curve>(payload, "smile");
  const surface = section<SurfacePayload>(payload, "surface");
  const expiriesTable = section<Table>(payload, "expiries");
  const expiries = expiriesOf(payload);
  const ctx = contextOf(payload);
  const rv30 = num(ctx.rv_30d);
  const vrp = num(ctx.vrp_30d);
  const chain = useMemo(() => chainOf(payload), [payload]);
  const [picked, setPicked] = useState<string[]>(() => (smile?.series ?? []).slice(0, 2).map((s) => s.key));
  const [xMode, setXMode] = useState<XMode>("k");

  const toggle = (key: string) =>
    setPicked((p) => (p.includes(key) ? p.filter((k) => k !== key) : p.length >= 4 ? p : [...p, key]));
  const shown = (smile?.series ?? []).filter((s) => picked.includes(s.key));
  // Strike axis differs per expiry (K = F·e^k), so it shows one expiry at a time.
  const strikeMode = xMode === "strike" && shown.length === 1;
  const keyOf = (ts: string) => `exp_${ts.slice(0, 10).replaceAll("-", "")}`;
  const fwdOf = (key: string) => expiries.find((e) => keyOf(e.expiry_ts) === key)?.forward ?? null;

  const xs = (smile?.x.values ?? []).map((k) => {
    if (strikeMode) return (fwdOf(shown[0]!.key) ?? 0) * Math.exp(k);
    if (xMode === "pct") return Math.exp(k) * 100;
    return k;
  });
  const nodePoints = shown.flatMap((s, i) => {
    const style = SERIES_STYLES[i]!;
    return chain
      .filter((r) => r.is_node === true && keyOf(String(r.expiry_ts)) === s.key && typeof r.k === "number")
      .map((r) => {
        const k = r.k as number;
        const x = strikeMode ? (r.strike as number) : xMode === "pct" ? Math.exp(k) * 100 : k;
        return { x, y: r.iv_mark as number, lo: r.iv_bid as number | null, hi: r.iv_ask as number | null, color: style.color, dim: r.node_quality !== "good" };
      })
      .filter((q) => q.x >= Math.min(...xs) && q.x <= Math.max(...xs));
  });
  const xLabel = strikeMode ? "Strike" : xMode === "pct" ? "% of forward" : "Log-moneyness ln(K/F)";
  const fmtX = (v: number) => (strikeMode ? formatNumber(v, 0) : xMode === "pct" ? `${formatNumber(v, 0)}%` : formatNumber(v, 2, true));

  const SKEW_KEYS = ["label", "dte", "atm_iv", "rr25", "bf25", "basis_ann", "n_good", "n_nodes", "median_spread_vp"];
  const keep = expiriesTable ? expiriesTable.columns.map((c, i) => (SKEW_KEYS.includes(c.key) ? i : -1)).filter((i) => i >= 0) : [];
  const skew: Table | null = expiriesTable && {
    columns: keep.map((i) => expiriesTable.columns[i]!),
    rows: expiriesTable.rows.map((row) => keep.map((i) => row[i] ?? null)),
  };

  return (
    <div className="panel">
      <div className="panel-body">
        {ts && (
          <section className="subsection">
            <div className="title sub-title">1) Term structure</div>
            <LineChart
              title="Term structure"
              x={ts.x.values}
              xLabel="Tenor (days)"
              series={ts.series.map((s) => ({ key: s.key, label: s.label, values: s.values }))}
              formatX={(v) => fmtDte(v)}
              formatY={(v) => fmtIv(v)}
              refLines={rv30 === null ? [] : [{ y: rv30, label: `RV 30D ${fmtIv(rv30)}` }]}
              markers={[{ x: 30, label: `30D · VRP ${fmtVp(vrp)}` }]}
              footer="ATM and 25Δ vols at each listed expiry; dashed line is realized vol over 30 days of hourly perp returns."
            />
          </section>
        )}
        {smile && (
          <section className="subsection">
            <div className="title sub-title">2) Smiles (up to 4)</div>
            <div className="toolbar">
              <span className="chips">
                {smile.series.map((s) => (
                  <button key={s.key} className={picked.includes(s.key) ? "active" : ""} onClick={() => toggle(s.key)}>{s.label}</button>
                ))}
              </span>
              <label className="label">X axis{" "}
                <select value={xMode} onChange={(e) => setXMode(e.target.value as XMode)}>
                  <option value="k">Log-moneyness</option>
                  <option value="pct">% of forward</option>
                  <option value="strike">Strike (one expiry)</option>
                </select>
              </label>
              {xMode === "strike" && shown.length !== 1 && <span className="badge">PICK ONE EXPIRY FOR THE STRIKE AXIS</span>}
            </div>
            <LineChart
              title="Smiles"
              x={xs}
              xLabel={xLabel}
              series={(strikeMode ? shown : xMode === "strike" ? [] : shown).map((s) => ({ key: s.key, label: s.label, values: s.values }))}
              points={strikeMode || xMode !== "strike" ? nodePoints : []}
              formatX={fmtX}
              formatY={(v) => fmtIv(v)}
              footer="Lines interpolate total variance between nodes and stop where the nodes stop (no extrapolation). Dots are listed nodes; bars span bid–ask IV; faded dots are not tight two-sided quotes."
            />
          </section>
        )}
        {surface && (
          <section className="subsection">
            <div className="title sub-title">3) Surface</div>
            <SurfaceHeatmap payload={surface} title="Vol surface" />
          </section>
        )}
        {skew && (
          <section className="subsection">
            <div className="title sub-title">4) Skew by expiry</div>
            <DataTable payload={skew} caption="Skew by expiry" />
            <div className="chart-footer dim">Expiries: {expiries.map((e) => fmtExpiry(e.expiry_ts)).join(" · ")}</div>
          </section>
        )}
      </div>
    </div>
  );
}
