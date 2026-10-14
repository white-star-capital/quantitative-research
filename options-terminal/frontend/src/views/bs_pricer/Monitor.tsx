// Cross-asset vol monitor (guide §7.2): one row per asset from its own /latest, so one asset's
// failure never blanks the others.
import { useState } from "react";
import { usePolled } from "../../api/usePolled.ts";
import { checkLatest } from "../../api/validate.ts";
import { fmtIv, fmtSpot, fmtVp } from "../../format/pricer.ts";
import { formatAge, formatNumber } from "../../format/units.ts";
import type { BespokeProps } from "../registry.ts";
import { loadPref, savePref } from "./logic.ts";
import { assetHref, contextOf, latestPath, num, payloadOf, storage } from "./shared.ts";
import type { SlotSummary } from "../../api/types.ts";
import { findParitySlot } from "../parity_monitor/logic.ts";
import { PARITY_COLUMNS, ParityCells, useParityLatest, useParitySelection, whyMissing } from "../parity_monitor/parts.tsx";

interface ParityRef {
  slot: SlotSummary;
  selection: Record<string, string> | null;
}

/** Parity columns (parity guide §7.1) from the parity slot's own /latest, so either slot can
 *  fail without blanking the other. */
function ParityPart({ parity, underlying }: { parity: ParityRef; underlying: string }) {
  const latest = useParityLatest(parity.slot, underlying, parity.selection);
  return <ParityCells payload={payloadOf(latest.data)} why={whyMissing(latest)} />;
}

function Row({ slotId, underlying, selection, gate, parity }: { slotId: string; underlying: string; selection: Record<string, string>; gate: number; parity: ParityRef | null }) {
  const latest = usePolled(latestPath(slotId, underlying, selection), checkLatest);
  const href = assetHref(slotId, underlying);
  const payload = payloadOf(latest.data);
  const ctx = contextOf(payload);
  const meta = latest.data?.meta;
  const env = latest.data?.data;

  if (!payload || !env || !meta) {
    const why = latest.loading ? "LOADING…" : latest.problem?.slug === "no-output-yet" ? "NO OUTPUT YET" : latest.error ?? latest.problem?.detail ?? "UNAVAILABLE";
    return (
      <tr className="clickable" onClick={() => (window.location.hash = href)}>
        <td data-label="Asset" className="row-code"><a href={href}>{underlying}</a></td>
        <td colSpan={11} className="dim">{why}</td>
        {parity && <ParityPart parity={parity} underlying={underlying} />}
      </tr>
    );
  }
  const spot = num(ctx.spot);
  const change = num(ctx.spot_change_24h);
  const vrp = num(ctx.vrp_30d);
  const vrpGood = vrp !== null && vrp * 100 >= gate;
  const flagged = (k: string) => (ctx[k]?.flags?.length ? <span className="badge" title={ctx[k]?.flags?.join(", ")}>OFF-GRID</span> : null);
  return (
    <tr className="clickable" onClick={() => (window.location.hash = href)}>
      <td data-label="Asset" className="row-code"><a href={href}>{underlying}</a></td>
      <td data-label="Spot · 24h" className="num">
        {fmtSpot(underlying, spot)}{" "}
        <span className={change === null || change === 0 ? "dim" : change > 0 ? "good" : "poor"}>{change === null ? "—" : `${formatNumber(change * 100, 1, true)}%`}</span>
      </td>
      <td data-label="ATM 7d" className="num">{fmtIv(num(ctx.atm_iv_7d))} {flagged("atm_iv_7d")}</td>
      <td data-label="ATM 30d" className="num">{fmtIv(num(ctx.atm_iv_30d))}</td>
      <td data-label="ATM 90d" className="num">{fmtIv(num(ctx.atm_iv_90d))} {flagged("atm_iv_90d")}</td>
      <td data-label="RV 30d" className="num">{fmtIv(num(ctx.rv_30d))}</td>
      <td data-label="VRP 30d" className={`num ${vrp === null ? "" : vrpGood ? "good" : "dim"}`}>
        {fmtVp(vrp)} {vrp !== null && <span className="dim">{vrpGood ? "≥ GATE" : "< GATE"}</span>}
      </td>
      <td data-label="RR25 30d" className="num">{fmtVp(num(ctx.rr25_30d))}</td>
      <td data-label="BF25 30d" className="num">{fmtVp(num(ctx.bf25_30d))}</td>
      <td data-label="Term ratio" className="num">
        {num(ctx.term_ratio) === null ? "—" : formatNumber(num(ctx.term_ratio) as number, 2)}{" "}
        {ctx.event_flag?.value === true && <span className="badge bad">EVENT</span>}
      </td>
      <td data-label="Data" className="num">
        {num(ctx.n_nodes_good) ?? "—"}/{num(ctx.n_nodes) ?? "—"} {env.warnings.length > 0 && <span className="badge" title={env.warnings.map((w) => w.code).join(", ")}>⚠ {env.warnings.length}</span>}
      </td>
      <td data-label="Updated" className="num">
        {formatAge(meta.age_seconds)} <span className={`status ${meta.stale ? "bad" : "ok"}`}>{meta.stale ? "STALE" : "FRESH"}</span>
      </td>
      {parity && <ParityPart parity={parity} underlying={underlying} />}
    </tr>
  );
}

export function Monitor({ slot, selection, slots }: BespokeProps) {
  const [gate, setGate] = useState(() => loadPref("vrp_gate_vp", 5, storage()));
  const assets = slot.scope.allowed?.underlying ?? [];
  const paritySlot = findParitySlot(slots);
  const paritySelection = useParitySelection(paritySlot);
  const parity = paritySlot ? { slot: paritySlot, selection: paritySelection } : null;
  return (
    <div className="panel">
      <div className="panel-title">
        <span className="title">1) Cross-asset vol monitor</span>
        <label className="dim">
          VRP GATE (VP){" "}
          <input
            type="number"
            step="0.5"
            value={gate}
            style={{ width: 70 }}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isFinite(v)) {
                setGate(v);
                savePref("vrp_gate_vp", v, storage());
              }
            }}
          />
        </label>
      </div>
      <div className="table-wrap">
        <table className="grid">
          <thead>
            <tr>
              {["Asset", "Spot · 24h", "ATM 7d", "ATM 30d", "ATM 90d", "RV 30d", "VRP 30d", "RR25 30d", "BF25 30d", "Term ratio", "Data", "Updated", ...(parity ? PARITY_COLUMNS : [])].map((h) => (
                <th key={h} className={h === "Asset" ? undefined : "num"}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {assets.map((u) => <Row key={u} slotId={slot.slot_id} underlying={u} selection={selection} gate={gate} parity={parity} />)}
          </tbody>
        </table>
      </div>
      <div className="chart-footer dim">
        ATM IVs are constant-maturity from the Deribit surface; VRP = ATM 30d − RV 30d (hourly perp returns). The VRP gate is
        your display threshold, kept in this browser only. Event = 1d/30d term ratio at or above the model's threshold.
        {parity && " Parity columns come from the parity monitor: 30d values interpolate between listed expiries; \"perp only\" means no dated future."}
      </div>
    </div>
  );
}
