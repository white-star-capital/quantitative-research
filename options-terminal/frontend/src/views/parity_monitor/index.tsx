// parity_monitor bespoke view (PARITY_MONITOR_IMPLEMENTATION_GUIDE.md §7): a cross-asset landing
// table (§7.1) and an asset view with the four Parity panels (§7.2). The generic composite
// renderer remains available through the BESPOKE | GENERIC toggle.
import { fmtBp, fmtRateAnn } from "../../format/parity.ts";
import { fmtSpot } from "../../format/pricer.ts";
import { formatAge, formatNumber, formatUtc } from "../../format/units.ts";
import { registerView, type BespokeProps } from "../registry.ts";
import { ErrorIndicator, FreshnessBadge, Loading, NoOutputYet, SlotRetired, Unsupported } from "../states.tsx";
import { assetHref, payloadOf } from "../bs_pricer/shared.ts";
import { usePolled } from "../../api/usePolled.ts";
import { checkLatest } from "../../api/validate.ts";
import { latestPath } from "../bs_pricer/shared.ts";
import { at30d, context, perpOnly, skewFlagged, table, VIEW_ID } from "./logic.ts";
import { ParityPanels } from "./ParityPanels.tsx";
import { PARITY_COLUMNS, ParityCells, whyMissing } from "./parts.tsx";

function MonitorRow({ slotId, underlying, selection }: { slotId: string; underlying: string; selection: Record<string, string> }) {
  const latest = usePolled(latestPath(slotId, underlying, selection), checkLatest);
  const href = assetHref(slotId, underlying);
  const payload = payloadOf(latest.data);
  const meta = latest.data?.meta;
  return (
    <tr className="clickable" onClick={() => (window.location.hash = href)}>
      <td data-label="Asset" className="row-code"><a href={href}>{underlying}</a></td>
      <ParityCells payload={payload} why={whyMissing(latest)} />
      <td data-label="Updated" className="num">
        {meta ? <>{formatAge(meta.age_seconds)} <span className={`status ${meta.stale ? "bad" : "ok"}`}>{meta.stale ? "STALE" : "FRESH"}</span></> : "—"}
        {payload && skewFlagged(payload) && <span className="badge bad">SKEW</span>}
      </td>
    </tr>
  );
}

function ParityMonitor({ slot, selection }: BespokeProps) {
  const assets = slot.scope.allowed?.underlying ?? [];
  return (
    <div className="panel">
      <div className="panel-title"><span className="title">1) Cross-asset parity monitor</span></div>
      <div className="table-wrap">
        <table className="grid">
          <thead>
            <tr>{["Asset", ...PARITY_COLUMNS, "Updated"].map((h) => <th key={h} className={h === "Asset" ? undefined : "num"}>{h}</th>)}</tr>
          </thead>
          <tbody>
            {assets.map((u) => <MonitorRow key={u} slotId={slot.slot_id} underlying={u} selection={selection} />)}
          </tbody>
        </table>
      </div>
      <div className="chart-footer dim">
        30d values interpolate between listed expiries (never extrapolated). Implied basis comes from put-call parity on executable
        bid/ask; future basis and options-vs-futures only where a same-expiry dated future exists ("perp only" otherwise). Best net
        edge is after taker fees; confirmed opportunities were rechecked on order books. All indicative at the snapshot.
      </div>
    </div>
  );
}

function ParityAsset(props: BespokeProps & { underlying: string }) {
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
  const ctx = context(payload);
  const expiries = table(payload, "expiries");
  const n = (v: unknown) => (typeof v === "number" ? v : null);
  const tiles: [string, string, (string | undefined)?][] = [
    ["Spot", fmtSpot(underlying, n(ctx.spot))],
    ["Implied basis 30d", fmtRateAnn(at30d(expiries, "implied_basis_ann"))],
    ["Options vs futures 30d", perpOnly(expiries) ? "perp only" : fmtBp(at30d(expiries, "options_vs_future_bps"))],
    ["Perp funding", fmtRateAnn(n(ctx.funding_ann))],
    ["Best net edge", fmtRateAnn(n(ctx.best_net_edge_ann)), `${n(ctx.n_candidates) ?? "—"} after fees`],
    ["Confirmed", String(n(ctx.n_confirmed) ?? "—"), "on order books"],
    ["Book skew", n(ctx.snapshot_skew_s) === null ? "—" : `${formatNumber(n(ctx.snapshot_skew_s) as number, 1)}s`, skewFlagged(payload) ? "SKEW" : undefined],
  ];
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
          {n(ctx.n_expiries) ?? "—"} EXPIRIES · {n(ctx.n_locked_expiries) ?? "—"} WITH A DATED FUTURE
        </div>
        {meta.stale && <ErrorIndicator error={`RUN IS OLDER THAN ${Math.round(meta.max_age_seconds / 60)} MIN`} lastGoodAsOf={env.as_of} />}
        {env.warnings.length > 0 && (
          <div className="notice"><span className="status mid">WARN</span> {env.warnings.map((w) => `${w.code}: ${w.message}`).join(" · ")}</div>
        )}
        {env.status === "failed" ? (
          <div className="notice bad">RUN FAILED · {env.error.code} · {env.error.message}</div>
        ) : (
          <div className="panel-body">
            <div className="tiles">
              {tiles.map(([label, value, sub]) => (
                <div key={label} className="tile">
                  <div className="label">{label}</div>
                  <div className="tile-value num">{value}</div>
                  {sub && <div className="tile-sub dim">{sub}</div>}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      {env.status === "success" && payload && <ParityPanels env={env} payload={payload} underlying={underlying} />}
    </>
  );
}

function ParityView(props: BespokeProps) {
  return props.scope.underlying ? <ParityAsset {...props} underlying={props.scope.underlying} /> : <ParityMonitor {...props} />;
}

registerView(VIEW_ID, ParityView);
