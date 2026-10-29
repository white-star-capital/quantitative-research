// Asset view (guide §7.3): header tiles from `context`, freshness and audit line, then the
// Chain · Vol · Pricer & OTC · Data tabs.
import { useState } from "react";
import { fmtIv, fmtSpot, fmtVp } from "../../format/pricer.ts";
import { formatAge, formatNumber, formatUtc } from "../../format/units.ts";
import type { BespokeProps } from "../registry.ts";
import { ErrorIndicator, FreshnessBadge, Loading, NoOutputYet, SlotRetired, Unsupported } from "../states.tsx";
import { ChainTab } from "./ChainTab.tsx";
import { DataTab } from "./DataTab.tsx";
import { chainOf } from "./logic.ts";
import { PricerTab, type Prefill } from "./PricerTab.tsx";
import { assetHref, contextOf, num, payloadOf } from "./shared.ts";
import { VolTab } from "./VolTab.tsx";
import { findParitySlot } from "../parity_monitor/logic.ts";
import { ParityTab } from "../parity_monitor/parts.tsx";
import { findVrpSlot, vrpCovers } from "../vrp_signal/logic.ts";
import { VrpTab } from "../vrp_signal/parts.tsx";

type Tab = "chain" | "vol" | "pricer" | "data" | "parity" | "vrp";
const TABS: [Tab, string][] = [["chain", "Chain"], ["vol", "Vol"], ["pricer", "Pricer & OTC"], ["data", "Data"]];

export function AssetView(props: BespokeProps & { underlying: string }) {
  const { slot, latest, underlying, selection } = props;
  const [tab, setTab] = useState<Tab>("chain");
  // Parity tab only when the catalog serves a parity slot (found by view id, never hard-coded).
  const paritySlot = findParitySlot(props.slots);
  // VRP tab likewise, and only for assets the VRP slot covers.
  const vrpSlot = findVrpSlot(props.slots);
  const vrpShown = vrpSlot !== null && vrpCovers(vrpSlot, underlying);
  const tabs: [Tab, string][] = [...TABS];
  if (paritySlot) tabs.push(["parity", "Parity"]);
  if (vrpShown) tabs.push(["vrp", "VRP"]);
  const [prefill, setPrefill] = useState<Prefill | null>(null);
  const assets = slot.scope.allowed?.underlying ?? [];

  const nav = (
    <div className="toolbar">
      <a href={assetHref(slot.slot_id, null)}>◂ MONITOR</a>
      <span className="chips">
        {assets.map((u) => (
          <button key={u} className={u === underlying ? "active" : ""} onClick={() => (window.location.hash = assetHref(slot.slot_id, u))}>
            {u}
          </button>
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
  const ctx = contextOf(payload);
  const tiles: [string, string, (string | undefined)?][] = [
    ["Spot", fmtSpot(underlying, num(ctx.spot)), num(ctx.spot_change_24h) === null ? undefined : `${formatNumber((num(ctx.spot_change_24h) as number) * 100, 1, true)}% 24h`],
    ["ATM IV 30d", fmtIv(num(ctx.atm_iv_30d))],
    ["RV 30d", fmtIv(num(ctx.rv_30d))],
    ["VRP 30d", fmtVp(num(ctx.vrp_30d))],
    ["RR25 30d", fmtVp(num(ctx.rr25_30d)), ctx.rr25_30d?.flags?.length ? "OFF-GRID" : undefined],
    ["BF25 30d", fmtVp(num(ctx.bf25_30d))],
    ["Term ratio", num(ctx.term_ratio) === null ? "—" : formatNumber(num(ctx.term_ratio) as number, 2), ctx.event_flag?.value === true ? "EVENT" : undefined],
  ];
  if (ctx.dvol) tiles.push(["DVOL", fmtIv(num(ctx.dvol)), "display only"]);

  return (
    <>
      {nav}
      <div className="panel">
        <div className="panel-title">
          <span className="title">
            1) {underlying} · {env.implementation_id} {env.implementation_version}
            {meta.role === "shadow" ? " · SHADOW" : ""}
          </span>
          <FreshnessBadge stale={meta.stale} />
        </div>
        <div className="panel-meta">
          AS OF <span className="num" title={new Date(env.as_of).toLocaleString()}>{formatUtc(env.as_of)}</span> · {formatAge(meta.age_seconds)} · NODES{" "}
          {num(ctx.n_nodes_good) ?? "—"} GOOD / {num(ctx.n_nodes) ?? "—"} · {num(ctx.n_expiries) ?? "—"} EXPIRIES
        </div>
        {meta.stale && <ErrorIndicator error={`RUN IS OLDER THAN ${Math.round(meta.max_age_seconds / 60)} MIN`} lastGoodAsOf={env.as_of} />}
        {env.warnings.length > 0 && (
          <div className="notice">
            <span className="status mid">WARN</span> {env.warnings.map((w) => `${w.code}: ${w.message}`).join(" · ")}
          </div>
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
      {payload && (
        <>
          <div className="tabs" role="tablist">
            {tabs.map(([id, label]) => (
              <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>
                {label}
              </button>
            ))}
          </div>
          {tab === "chain" && (
            <ChainTab
              underlying={underlying}
              payload={payload}
              onPrice={(p) => {
                setPrefill(p);
                setTab("pricer");
              }}
            />
          )}
          {tab === "vol" && <VolTab payload={payload} />}
          {tab === "pricer" && (
            <PricerTab slotId={slot.slot_id} underlying={underlying} selection={selection} runId={env.run_id} spot={num(ctx.spot)} prefill={prefill} />
          )}
          {tab === "data" && <DataTab envelope={env} payload={payload} nodeRows={chainOf(payload).filter((r) => r.is_node === true)} />}
          {tab === "parity" && paritySlot && <ParityTab slot={paritySlot} underlying={underlying} />}
          {tab === "vrp" && vrpShown && vrpSlot && <VrpTab slot={vrpSlot} underlying={underlying} />}
        </>
      )}
    </>
  );
}
