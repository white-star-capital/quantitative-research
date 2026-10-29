// VRP panels (RV_FORECAST_IMPLEMENTATION_GUIDE.md §6, §10, §12): headline tiles, 1) IV vs forecast
// RV cone, 2) decomposition, 3) signal by tenor, 4) events in horizon, 5) history. Rendered by
// the vrp_signal view and by the pricer's VRP tab. Advisory only: nothing here sizes a trade.
import type { Composite } from "../../api/payloads.ts";
import { fmtIv, fmtVp } from "../../format/pricer.ts";
import { fmtDay, fmtShare, fmtUnixDay, fmtZ } from "../../format/vrp.ts";
import { formatNumber, formatUtc } from "../../format/units.ts";
import { LineChart } from "../../renderers/chart/LineChart.tsx";
import { AMBER, CYAN } from "../../renderers/chart/palette.ts";
import { PanelBoundary } from "../PanelBoundary.tsx";
import { DecompBars } from "./DecompBars.tsx";
import {
  type Chip,
  type Gate,
  type SignalRow,
  type Tone,
  calendarChip,
  chips,
  cone,
  contextItems,
  ctxNum,
  events,
  gate,
  hasValues,
  history,
  ivPoints,
  rowAt,
  signalRows,
  signalStatus,
  zTitle,
} from "./logic.ts";

const TONE: Record<Tone, string> = { good: "good", mid: "", bad: "bad", dim: "dim" };
const tenor = (v: number) => `${formatNumber(v, 0)}d`;

export function ChipList({ list }: { list: Chip[] }) {
  return <>{list.map((c) => <span key={c.label} className={`badge ${TONE[c.tone]}`} title={c.title}>{c.label}</span>)}</>;
}

export function StatusWord({ row, g }: { row: SignalRow; g: Gate }) {
  const s = signalStatus(row, g);
  return <span className={s.tone === "good" ? "status ok" : "status dim"} title={s.title}>{s.label}</span>;
}

/** Event-calendar countdown: warns before the calendar gate shuts (§12.4). */
export function CalendarNotice({ chip }: { chip: Chip | null }) {
  if (!chip) return null;
  return (
    <div className="notice">
      <ChipList list={[chip]} /> {chip.title}
    </div>
  );
}

/** Staleness gate notice: the signal is disabled, not shown green (§10). */
export function GateNotice({ g }: { g: Gate }) {
  if (g.open) return null;
  return (
    <div className="notice bad">
      <span className="status bad">SIGNAL DISABLED</span> {g.reason}. Forecasts are shown for reference only; no row reads as rich.
    </div>
  );
}

export function VrpPanels({ payload, stale }: { payload: Composite; stale: boolean }) {
  const ctx = contextItems(payload);
  const g = gate(stale, ctx);
  const rows = signalRows(payload);
  const humpShare = ctxNum(ctx, "event_hump_share");
  const r30 = rowAt(rows, 30);
  const tiles: [string, string, (string | undefined)?, (string | undefined)?][] = [
    ["ATM IV 30d", fmtIv(r30?.iv_impl)],
    ["RV forecast 30d", fmtIv(r30?.rv_forecast), `cone ${fmtIv(r30?.rv_low)}–${fmtIv(r30?.rv_high)}`],
    ["VRP 30d", fmtVp(r30?.vrp)],
    ["VRP z 30d", fmtZ(r30?.vrp_z), r30?.vrp_z == null ? "no z" : "DVOL basis", r30 ? zTitle(r30) : undefined],
    ["Event share 30d", fmtShare(r30?.comp_event), r30?.event_hump ? "EVENT HUMP" : undefined],
    ["DVOL", fmtIv(ctxNum(ctx, "dvol")), `vs RV fcst ${fmtVp(ctxNum(ctx, "vrp_30d_dvol"))}`],
    ["RV trailing 30d", fmtIv(ctxNum(ctx, "rv_trailing_30d"))],
    ["EWMA / HAR 30d", `${fmtIv(ctxNum(ctx, "rv_ewma_30d"))} / ${fmtIv(ctxNum(ctx, "rv_har_30d"))}`, "baselines"],
    [
      "Last complete day",
      fmtUnixDay(ctxNum(ctx, "last_complete_day")),
      `R² ${ctxNum(ctx, "fit_r2") === null ? "—" : formatNumber(ctxNum(ctx, "fit_r2") as number, 2)} · ${ctxNum(ctx, "n_days") ?? "—"} days`,
    ],
  ];
  return (
    <div className="panel">
      <GateNotice g={g} />
      <CalendarNotice chip={calendarChip(ctx, g)} />
      <div className="panel-body">
        <div className="tiles">
          {tiles.map(([label, value, sub, title]) => (
            <div key={label} className="tile" title={title}>
              <div className="label">{label}</div>
              <div className={`tile-value num ${g.open ? "" : "dim"}`}>{value}</div>
              {sub && <div className="tile-sub dim">{sub}</div>}
            </div>
          ))}
        </div>
        <PanelBoundary><ConePanel payload={payload} /></PanelBoundary>
        <PanelBoundary><DecompPanel rows={rows} humpShare={humpShare} /></PanelBoundary>
        <PanelBoundary><SignalPanel rows={rows} g={g} humpShare={humpShare} /></PanelBoundary>
        <PanelBoundary><EventsPanel payload={payload} /></PanelBoundary>
        <PanelBoundary><HistoryPanel payload={payload} /></PanelBoundary>
      </div>
    </div>
  );
}

// ---- 1) IV vs forecast RV cone -------------------------------------------------------------

function ConePanel({ payload }: { payload: Composite }) {
  const c = cone(payload);
  if (!c) return null;
  return (
    <section className="subsection">
      <div className="title sub-title">1) ATM IV vs forecast RV</div>
      <LineChart
        title="ATM IV vs forecast RV cone"
        x={c.x}
        xLabel="Horizon (days)"
        series={[
          { key: "rv_forecast", label: "RV forecast (to horizon)", values: c.s("rv_forecast"), style: { color: AMBER } },
          { key: "rv_path", label: "Daily path", values: c.s("rv_path"), style: { color: CYAN, dash: "6 4" } },
          { key: "rv_path_ex_event", label: "Daily path ex-event", values: c.s("rv_path_ex_event"), style: { color: AMBER, dash: "2 3" } },
        ]}
        band={{ lo: c.s("rv_low"), hi: c.s("rv_high"), label: "RV cone q0.1–q0.9" }}
        points={ivPoints(c.x, c.s("atm_iv")).map((p) => ({ ...p, color: CYAN }))}
        formatX={tenor}
        formatY={(v) => fmtIv(v)}
        footer="Solid: mean forecast RV annualized over days 1..T; shaded: the 0.1–0.9 cone; cyan dots: ATM IV at each tenor. Dashed: forecast RV of day T on its own, dotted: the same with event terms zeroed (the gap is the event hump). Rich only when ATM IV clears the upper cone."
      />
    </section>
  );
}

// ---- 2) decomposition ----------------------------------------------------------------------

function DecompPanel({ rows, humpShare }: { rows: SignalRow[]; humpShare: number | null }) {
  if (rows.length === 0) return null;
  return (
    <section className="subsection">
      <div className="title sub-title">2) Where the forecast comes from</div>
      <DecompBars
        title="Forecast variance decomposition by tenor"
        rows={rows}
        humpShare={humpShare}
        footer="Shares of forecast variance over each tenor: baseline (HAR terms), seasonal (day of week) and events, attributed counterfactually. An event share past the dashed line is an event hump: the richness may be the market pricing a known catalyst."
      />
    </section>
  );
}

// ---- 3) signal by tenor --------------------------------------------------------------------

function SignalPanel({ rows, g, humpShare }: { rows: SignalRow[]; g: Gate; humpShare: number | null }) {
  return (
    <section className="subsection">
      <div className="title sub-title">3) Signal by tenor</div>
      {rows.length === 0 ? (
        <div className="notice">No signal rows in this run.</div>
      ) : (
        <div className="table-wrap">
          <table className="grid">
            <thead>
              <tr>
                {["Tenor", "ATM IV", "RV forecast", "Cone q0.1–q0.9", "VRP", "VRP z", "Event share", "Read", "Signal", "Flags"].map((h) => (
                  <th key={h} className={["Read", "Signal", "Flags"].includes(h) ? undefined : "num"}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.tenor_days} className={g.open ? undefined : "dim"}>
                  <td data-label="Tenor" className="num">{tenor(r.tenor_days)}</td>
                  <td data-label="ATM IV" className="num">{fmtIv(r.iv_impl)}</td>
                  <td data-label="RV forecast" className="num">{fmtIv(r.rv_forecast)}</td>
                  <td data-label="Cone q0.1–q0.9" className="num">{fmtIv(r.rv_low)}–{fmtIv(r.rv_high)}</td>
                  <td data-label="VRP" className="num">{fmtVp(r.vrp)}</td>
                  <td data-label="VRP z" className="num" title={zTitle(r)}>{fmtZ(r.vrp_z)}</td>
                  <td data-label="Event share" className="num">{fmtShare(r.comp_event)}</td>
                  <td data-label="Read"><ChipList list={chips(r, g, humpShare)} /></td>
                  <td data-label="Signal"><StatusWord row={r} g={g} /></td>
                  <td data-label="Flags" className="dim">{r.flags.join(", ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="chart-footer dim">
        VRP = ATM IV − mean forecast RV. RICH needs ATM IV above the 0.9 cone and no event hump. The z-score exists for BTC and ETH at
        30d only, scored on DVOL − RV forecast against its trailing 90 days (VRP-A-5). A central estimate: it under-forecasts tail realized
        vol, so it gates direction and never sizes.
      </div>
    </section>
  );
}

// ---- 4) events in horizon ------------------------------------------------------------------

function EventsPanel({ payload }: { payload: Composite }) {
  const list = events(payload);
  return (
    <section className="subsection">
      <div className="title sub-title">4) Events in horizon</div>
      {list.length === 0 ? (
        <div className="notice">No calendar event window overlaps the forecast horizon.</div>
      ) : (
        <div className="table-wrap">
          <table className="grid">
            <thead>
              <tr>
                {["Time", "Type", "Scope", "Window", "Forecast day", "Source"].map((h) => (
                  <th key={h} className={["Window", "Forecast day"].includes(h) ? "num" : undefined}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {list.map((e) => (
                <tr key={`${e.ts}-${e.type}-${e.scope}`}>
                  <td data-label="Time" style={{ whiteSpace: "nowrap" }}>{formatUtc(e.ts)}</td>
                  <td data-label="Type">{e.type}</td>
                  <td data-label="Scope">{e.scope}</td>
                  <td data-label="Window" className="num">−{e.window_before ?? "—"}d / +{e.window_after ?? "—"}d</td>
                  <td data-label="Forecast day" className="num">
                    {e.horizon_day === null ? "—" : e.horizon_day <= 0 ? <span className="dim">past ({e.horizon_day})</span> : e.horizon_day}
                  </td>
                  <td data-label="Source" className="dim">{e.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

// ---- 5) history ----------------------------------------------------------------------------

function HistoryPanel({ payload }: { payload: Composite }) {
  const h = history(payload);
  if (!h) return null;
  const z = h.s("vrp_z_30d");
  return (
    <section className="subsection">
      <div className="title sub-title">5) History, 30d</div>
      <LineChart
        title="Realized vs forecast RV 30d and DVOL"
        x={h.t}
        xLabel="Day close (UTC)"
        series={[
          { key: "rv_realized_30d", label: "Realized 30d", values: h.s("rv_realized_30d"), style: { color: AMBER } },
          { key: "rv_forecast_30d", label: "Forecast 30d (as of)", values: h.s("rv_forecast_30d"), style: { color: CYAN, dash: "6 4" } },
          { key: "dvol", label: "DVOL", values: h.s("dvol"), style: { color: AMBER, dash: "2 3" } },
        ]}
        formatX={(v) => fmtDay(v)}
        formatY={(v) => fmtIv(v)}
        footer="Realized looks back over the 30 days ending at each close; the forecast at a close covers the 30 days after it, fitted on prior days only. DVOL is Deribit's 30d implied index (BTC and ETH). Gaps are missing days, never zero."
      />
      {hasValues(z) && (
        <LineChart
          title="VRP z-score 30d, DVOL basis"
          x={h.t}
          xLabel="Day close (UTC)"
          series={[{ key: "vrp_z_30d", label: "VRP z 30d", values: z, style: { color: AMBER } }]}
          refLines={[{ y: 2, label: "+2σ" }, { y: 0, label: "0σ" }, { y: -2, label: "−2σ" }]}
          formatX={(v) => fmtDay(v)}
          formatY={(v) => fmtZ(v)}
          height={160}
          footer="(DVOL − forecast RV 30d) against its trailing 90 days before each close (VRP-A-5). Positive: implied rich vs forecast realized."
        />
      )}
    </section>
  );
}
