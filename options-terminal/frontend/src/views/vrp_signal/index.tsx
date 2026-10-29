// vrp_signal bespoke view (RV_FORECAST_IMPLEMENTATION_GUIDE.md §12): a cross-asset landing table
// and an asset view with the VRP panels. The generic composite renderer remains available
// through the BESPOKE | GENERIC toggle.
import { usePolled } from "../../api/usePolled.ts";
import { checkLatest } from "../../api/validate.ts";
import { fmtIv } from "../../format/pricer.ts";
import { formatAge, formatUtc } from "../../format/units.ts";
import { assetHref, latestPath, payloadOf } from "../bs_pricer/shared.ts";
import { registerView, type BespokeProps } from "../registry.ts";
import { ErrorIndicator, FreshnessBadge, Loading, NoOutputYet, SlotRetired, Unsupported } from "../states.tsx";
import { calendarChip, contextItems, ctxNum, gate, rowAt, signalRows, VIEW_ID, VRP_COLUMNS } from "./logic.ts";
import { VrpCells, whyMissing } from "./parts.tsx";
import { ChipList, VrpPanels } from "./VrpPanels.tsx";

const COLUMNS = ["Asset", "ATM IV 30d", "RV forecast 30d", ...VRP_COLUMNS, "Updated"];

function MonitorRow({ slotId, underlying, selection }: { slotId: string; underlying: string; selection: Record<string, string> }) {
  const latest = usePolled(latestPath(slotId, underlying, selection), checkLatest);
  const href = assetHref(slotId, underlying);
  const payload = payloadOf(latest.data);
  const meta = latest.data?.meta;
  const r30 = rowAt(signalRows(payload), 30);
  // The calendar countdown here too, so the monitor warns before the gate shuts (§12.4).
  const ctx = contextItems(payload);
  const cal = calendarChip(ctx, gate(meta?.stale ?? true, ctx));
  return (
    <tr className="clickable" onClick={() => (window.location.hash = href)}>
      <td data-label="Asset" className="row-code"><a href={href}>{underlying}</a>{cal && <> <ChipList list={[cal]} /></>}</td>
      <td data-label="ATM IV 30d" className="num">{fmtIv(r30?.iv_impl)}</td>
      <td data-label="RV forecast 30d" className="num">{fmtIv(r30?.rv_forecast)}</td>
      <VrpCells payload={payload} stale={meta?.stale ?? true} why={whyMissing(latest)} />
      <td data-label="Updated" className="num">
        {meta ? <>{formatAge(meta.age_seconds)} <FreshnessBadge stale={meta.stale} /></> : "—"}
      </td>
    </tr>
  );
}

function VrpMonitor({ slot, selection }: BespokeProps) {
  const assets = slot.scope.allowed?.underlying ?? [];
  return (
    <div className="panel">
      <div className="panel-title"><span className="title">1) Cross-asset VRP monitor</span></div>
      <div className="table-wrap">
        <table className="grid">
          <thead>
            <tr>{COLUMNS.map((h) => <th key={h} className={["Asset", "VRP read 30d"].includes(h) ? undefined : "num"}>{h}</th>)}</tr>
          </thead>
          <tbody>
            {assets.map((u) => <MonitorRow key={u} slotId={slot.slot_id} underlying={u} selection={selection} />)}
          </tbody>
        </table>
      </div>
      <div className="chart-footer dim">
        RV forecast is a calendar- and event-aware HAR mean over the next 30 UTC days; VRP = ATM IV − RV forecast. RICH only when ATM IV
        clears the 0.9 cone without an event hump. z exists for BTC and ETH only (DVOL basis). Stale runs or a lagging RV input disable the
        signal. Advisory only; never sizes.
      </div>
    </div>
  );
}

function VrpAsset(props: BespokeProps & { underlying: string }) {
  const { slot, latest, underlying } = props;
  const assets = slot.scope.allowed?.underlying ?? [];
  const nav = (
    <div className="toolbar">
      <a href={assetHref(slot.slot_id, null)}>◂ MONITOR</a>
      <span className="chips">
        {assets.map((u) => (
          <button key={u} className={u === underlying ? "active" : ""} onClick={() => (window.location.hash = assetHref(slot.slot_id, u))}>{u}</button>
        ))}
      </span>
    </div>
  );
  const { data, problem, error, unsupported, loading } = latest;
  if (!data) {
    let body;
    if (problem?.slug === "slot-retired") body = <SlotRetired slotId={slot.slot_id} />;
    else if (problem?.slug === "no-output-yet") body = <NoOutputYet />;
    else if (unsupported) body = <Unsupported raw={unsupported.raw} reason={unsupported.reason} />;
    else if (loading) body = <Loading what={`${underlying} RUN`} />;
    else body = <ErrorIndicator error={error ?? problem?.detail ?? "NO DATA"} lastGoodAsOf={null} />;
    return <>{nav}{body}</>;
  }
  const env = data.data;
  const meta = data.meta;
  const payload = payloadOf(data);
  const ctx = contextItems(payload);
  return (
    <>
      {nav}
      <div className="panel">
        <div className="panel-title">
          <span className="title">
            {underlying} · {env.implementation_id} {env.implementation_version}{meta.role === "shadow" ? " · SHADOW" : ""}
          </span>
          <FreshnessBadge stale={meta.stale} />
        </div>
        <div className="panel-meta">
          AS OF <span className="num" title={new Date(env.as_of).toLocaleString()}>{formatUtc(env.as_of)}</span> · {formatAge(meta.age_seconds)} ·{" "}
          {ctxNum(ctx, "n_days") ?? "—"} RV DAYS · {ctxNum(ctx, "n_days_missing") ?? "—"} MISSING
        </div>
        {meta.stale && <ErrorIndicator error={`RUN IS OLDER THAN ${Math.round(meta.max_age_seconds / 60)} MIN`} lastGoodAsOf={env.as_of} />}
        {env.warnings.length > 0 && (
          <div className="notice"><span className="status mid">WARN</span> {env.warnings.map((w) => `${w.code}: ${w.message}`).join(" · ")}</div>
        )}
        {env.status === "failed" && <div className="notice bad">RUN FAILED · {env.error.code} · {env.error.message}</div>}
      </div>
      {env.status === "success" && payload && <VrpPanels payload={payload} stale={meta.stale} />}
    </>
  );
}

function VrpView(props: BespokeProps) {
  return props.scope.underlying ? <VrpAsset {...props} underlying={props.scope.underlying} /> : <VrpMonitor {...props} />;
}

registerView(VIEW_ID, VrpView);
