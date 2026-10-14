// Parity tab, four panels in guide order (§7.2): 1) carry curve, 2) parity bands, 3) opportunities,
// 4) inventory carry. Rendered by the parity_monitor view and by the pricer's Parity tab.
// User inputs (USDC rate, capital, holdings, yields) stay in this browser (§12 security).
import { useMemo, useState } from "react";
import type { Composite, Curve } from "../../api/payloads.ts";
import { section } from "../../api/payloads.ts";
import type { SuccessEnvelope } from "../../api/types.ts";
import { fmtBp, fmtCoinEdge, fmtNative, fmtRateAnn } from "../../format/parity.ts";
import { fmtDte, fmtExpiry, fmtSpot, fmtUsdTotal } from "../../format/pricer.ts";
import { formatAge, formatNumber, formatUtc } from "../../format/units.ts";
import { LineChart } from "../../renderers/chart/LineChart.tsx";
import { AMBER, CYAN } from "../../renderers/chart/palette.ts";
import { loadPref, savePref } from "../bs_pricer/logic.ts";
import { storage } from "../bs_pricer/shared.ts";
import { PanelBoundary } from "../PanelBoundary.tsx";
import { BandBars, type Bar } from "./BandBars.tsx";
import {
  type Opportunity,
  type Side,
  RESIDUAL_RISK,
  barStatus,
  context,
  executable,
  feeRates,
  filterOpps,
  inventoryRows,
  recompute,
  returnOnCapital,
  scheduleVersion,
  skewFlagged,
  table,
} from "./logic.ts";

const PREFIX = "parity_monitor";
const pref = <T,>(key: string, fallback: T) => loadPref(key, fallback, storage(), PREFIX);
const keep = (key: string, value: unknown) => savePref(key, value, storage(), PREFIX);

function NumberInput({ label, value, onChange, step = "0.1", suffix }: { label: string; value: number; onChange: (v: number) => void; step?: string; suffix?: string }) {
  return (
    <label>
      <span className="label">{label}</span>
      <input type="number" step={step} value={Number.isFinite(value) ? value : ""} onChange={(e) => {
        const v = Number(e.target.value);
        if (Number.isFinite(v)) onChange(v);
      }} />{suffix && <span className="dim"> {suffix}</span>}
    </label>
  );
}

export function ParityPanels({ env, payload, underlying }: { env: SuccessEnvelope; payload: Composite; underlying: string }) {
  const ctx = context(payload);
  const spot = typeof ctx.spot === "number" ? ctx.spot : null;
  const skew = skewFlagged(payload);
  const schedule = scheduleVersion(payload);
  const realSchedule = schedule !== null && !["TBD", "TEST-ONLY"].includes(schedule);
  return (
    <div className="panel">
      {skew && (
        <div className="notice bad">
          <span className="status bad">SKEW</span> Option and futures books were {formatNumber(Number(ctx.snapshot_skew_s ?? 0), 1)}s apart at this
          snapshot. No unconfirmed row is executable.
        </div>
      )}
      {!realSchedule && (
        <div className="notice bad">
          <span className="status bad">FEES</span> Fee table "{schedule ?? "missing"}" is not Deribit's schedule. Net edges are not
          meaningful until it is filled (guide §9 step 2).
        </div>
      )}
      <div className="panel-body">
        <PanelBoundary><CarryPanel payload={payload} /></PanelBoundary>
        <PanelBoundary><BandsPanel payload={payload} underlying={underlying} /></PanelBoundary>
        <PanelBoundary><OpportunitiesPanel env={env} payload={payload} underlying={underlying} spot={spot} skew={skew} /></PanelBoundary>
        <PanelBoundary><InventoryPanel payload={payload} underlying={underlying} spot={spot} /></PanelBoundary>
      </div>
    </div>
  );
}

// ---- 1) carry curve ------------------------------------------------------------------------

function CarryPanel({ payload }: { payload: Composite }) {
  const curve = section<Curve>(payload, "carry_curve");
  const [usdc, setUsdc] = useState(() => pref("usdc_lending_pct", 5));
  if (!curve) return null;
  const s = (k: string) => curve.series.find((x) => x.key === k)?.values ?? [];
  const future = s("future_basis_ann");
  return (
    <section className="subsection">
      <div className="title sub-title">1) Carry curve</div>
      <div className="toolbar form-grid">
        <NumberInput label="USDC lending rate (this browser only)" value={usdc} suffix="% ann." onChange={(v) => { setUsdc(v); keep("usdc_lending_pct", v); }} />
      </div>
      <LineChart
        title="Carry curve"
        x={curve.x.values}
        xLabel="Tenor (days)"
        series={[
          { key: "implied", label: "Options-implied basis", values: s("implied_basis_ann"), style: { color: AMBER } },
          { key: "funding", label: "Perp funding", values: s("funding_ann"), style: { color: CYAN, dash: "6 4" } },
          { key: "r_lock", label: "Best inventory r_lock", values: s("r_lock_best"), style: { color: AMBER, dash: "2 3" } },
        ]}
        band={{ lo: s("implied_basis_ann_bid"), hi: s("implied_basis_ann_ask"), label: "Implied basis bid–ask" }}
        points={curve.x.values.flatMap((x, i) => (future[i] == null ? [] : [{ x, y: future[i] as number, color: CYAN }]))}
        refLines={[{ y: usdc / 100, label: `USDC ${formatNumber(usdc, 2)}%` }]}
        formatX={(v) => fmtDte(v)}
        formatY={(v) => fmtRateAnn(v)}
        footer="Simple annualized basis implied by put-call parity at each expiry (shaded: from the parity band's bid and ask), cyan dots: same-expiry dated future basis, dashed: perp funding, dotted: best inventory lock rate (continuous). Expiries without enough strikes are omitted."
      />
    </section>
  );
}

// ---- 2) parity bands -----------------------------------------------------------------------

function BandsPanel({ payload, underlying }: { payload: Composite; underlying: string }) {
  const bands = useMemo(() => table(payload, "bands"), [payload]);
  const opps = useMemo(() => table(payload, "opportunities") as unknown as Opportunity[], [payload]);
  const expiries = [...new Set(bands.map((b) => String(b.expiry_ts)))];
  const [picked, setPicked] = useState<string | null>(null);
  const expiry = picked && expiries.includes(picked) ? picked : (expiries[Math.min(2, expiries.length - 1)] ?? null);
  const rows = bands.filter((b) => b.expiry_ts === expiry && typeof b.band_bid_bps === "number" && typeof b.band_ask_bps === "number");
  const first = rows[0];
  const mid = first && typeof first.hedge_bid === "number" && typeof first.hedge_ask === "number" ? (first.hedge_bid + first.hedge_ask) / 2 : null;
  const bp = (v: unknown) => (mid !== null && typeof v === "number" ? (v / mid - 1) * 1e4 : null);
  const bars: Bar[] = rows.map((b) => ({ strike: b.strike as number, lo: b.band_bid_bps as number, hi: b.band_ask_bps as number, ...barStatus(b, opps) }));
  return (
    <section className="subsection">
      <div className="title sub-title">2) Parity bands</div>
      <div className="toolbar">
        <span className="chips">
          {expiries.map((e) => (
            <button key={e} className={e === expiry ? "active" : ""} onClick={() => setPicked(e)}>{fmtExpiry(e)}</button>
          ))}
        </span>
      </div>
      {bars.length === 0 ? (
        <div className="notice">No hedged two-sided strikes for this expiry.</div>
      ) : (
        <BandBars
          title={`Parity bands ${expiry ? fmtExpiry(expiry) : ""}`}
          asset={underlying}
          bars={bars}
          hedgeBidBp={bp(first?.hedge_bid)}
          hedgeAskBp={bp(first?.hedge_ask)}
          footer="Each bar spans the forwards at which a conversion (bid end) and a reversal (ask end) break even, in bp vs the hedge mid. Above the hedge ask: conversion; below the bid: reversal. Green survives taker fees, yellow is gross only."
        />
      )}
    </section>
  );
}

// ---- 3) opportunities ----------------------------------------------------------------------

function OpportunitiesPanel({ env, payload, underlying, spot, skew }: { env: SuccessEnvelope; payload: Composite; underlying: string; spot: number | null; skew: boolean }) {
  const [side, setSide] = useState<Side>("taker");
  const [showUnconfirmed, setShowUnconfirmed] = useState(false);
  const [showPerp, setShowPerp] = useState(false);
  const [capital, setCapital] = useState(() => pref("capital_pct", 15));
  const funding = context(payload).funding_ann;
  const rates = feeRates(payload, side);
  const all = useMemo(() => table(payload, "opportunities") as unknown as Opportunity[], [payload]);
  const shown = filterOpps(all, { showUnconfirmed, showPerp })
    .map((o) => (side === "taker" || !rates || spot === null ? o : recompute(o, rates, spot, env.as_of)))
    .sort((a, b) => (b.net_edge_ann ?? -Infinity) - (a.net_edge_ann ?? -Infinity));
  const indicative = `Indicative at ${formatUtc(env.as_of)}. Re-check live before trading.`;
  return (
    <section className="subsection">
      <div className="title sub-title">3) Opportunities</div>
      <div className="toolbar form-grid">
        <label><span className="label">Fees</span>
          <select value={side} onChange={(e) => setSide(e.target.value as Side)} disabled={!feeRates(payload, "maker")}>
            <option value="taker">Taker, all legs</option>
            <option value="maker">Maker, all legs</option>
          </select>
        </label>
        <NumberInput label="Capital (% of notional, this browser only)" value={capital} step="1" suffix="%" onChange={(v) => { setCapital(v); keep("capital_pct", v); }} />
        <label><input type="checkbox" checked={showUnconfirmed} onChange={(e) => setShowUnconfirmed(e.target.checked)} /> <span className="label">Show unconfirmed</span></label>
        <label><input type="checkbox" checked={showPerp} onChange={(e) => setShowPerp(e.target.checked)} /> <span className="label">Show perp-hedged</span></label>
      </div>
      <div className="dim">{indicative}</div>
      {shown.length === 0 ? (
        <div className="notice">
          No {showUnconfirmed ? "" : "executable "}parity breaks after fees at {formatUtc(env.as_of)}
          {all.length > shown.length ? ` (${all.length - shown.length} hidden by filters)` : ""}.
        </div>
      ) : (
        <div className="table-wrap">
          <table className="grid">
            <thead>
              <tr>
                {["Expiry", "Strike", "Structure", "Hedge", "Call", "Put", "Hedge px", "Net edge", "bp notional", "Ann.", "On capital", "Size", "Status", "Age", "Flags"].map((h) => (
                  <th key={h} className={["Expiry", "Structure", "Hedge", "Status", "Flags"].includes(h) ? undefined : "num"}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {shown.map((o) => {
                const perp = o.hedge_type === "perp_funding";
                const coin = o.call_px_unit === "coin";
                const exec = executable(o) && !(skew && !o.confirmed);
                return (
                  <tr key={`${o.expiry_ts}-${o.strike}-${o.structure}`} className={perp || !o.confirmed ? "dim" : undefined} title={`${RESIDUAL_RISK[o.hedge_type]} ${indicative}`}>
                    <td data-label="Expiry" style={{ whiteSpace: "nowrap" }}>{fmtExpiry(o.expiry_ts)}</td>
                    <td data-label="Strike" className="num">{fmtSpot(underlying, o.strike)}</td>
                    <td data-label="Structure">{o.structure === "conversion" ? "▲ CONVERSION" : "▼ REVERSAL"}</td>
                    <td data-label="Hedge" style={{ whiteSpace: "nowrap" }}>
                      <span className={`badge ${perp ? "bad" : ""}`} title={RESIDUAL_RISK[o.hedge_type]}>{perp ? "PERP · UNLOCKED" : "LOCKED FUTURE"}</span>{" "}
                      <span className="dim">{o.hedge_instrument}</span>
                    </td>
                    <td data-label="Call" className="num">{fmtNative(o.call_px, o.call_px_unit, underlying)}</td>
                    <td data-label="Put" className="num">{fmtNative(o.put_px, o.put_px_unit, underlying)}</td>
                    <td data-label="Hedge px" className="num">{fmtUsdTotal(o.hedge_px)}</td>
                    <td data-label="Net edge" className={`num ${o.net_edge > 0 ? "good" : "poor"}`}>
                      {coin ? fmtCoinEdge(o.net_edge, underlying, spot) : fmtUsdTotal(o.net_edge_usd, true)}
                    </td>
                    <td data-label="bp notional" className="num">{fmtBp(o.net_edge_bps_notional)}</td>
                    <td data-label="Ann." className="num">
                      {perp ? (
                        <span title="Constant funding at which the edge is zero, vs current funding">
                          BE {fmtRateAnn(o.breakeven_funding_ann)} vs {fmtRateAnn(typeof funding === "number" ? funding : null)}
                        </span>
                      ) : fmtRateAnn(o.net_edge_ann)}
                    </td>
                    <td data-label="On capital" className="num">{perp ? "—" : fmtRateAnn(returnOnCapital(o.net_edge_ann, capital))}</td>
                    <td data-label="Size" className="num">{o.exec_size_units === null ? <span className="dim">unconfirmed</span> : `${formatNumber(o.exec_size_units, 2)} ${underlying}`}</td>
                    <td data-label="Status"><span className={`status ${exec ? "ok" : "mid"}`}>{o.confirmed ? "CONFIRMED" : "UNCONFIRMED"}</span></td>
                    <td data-label="Age" className="num">{formatAge(o.snapshot_age_s)}</td>
                    <td data-label="Flags" className="dim">{o.flags || "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <div className="chart-footer dim">
        Edges per 1 {underlying} of options at executable bid/ask, after {side} fees on all legs and delivery when held to expiry; inverse
        edges are locked in coin (USD shown at spot). Confirmed rows were recomputed from all three legs' order books. Perp-hedged rows are not
        locked: they show the breakeven funding rate. "On capital" divides by your margin assumption; there is no margin model.
      </div>
    </section>
  );
}

// ---- 4) inventory carry --------------------------------------------------------------------

function InventoryPanel({ payload, underlying, spot }: { payload: Composite; underlying: string; spot: number | null }) {
  const [holding, setHolding] = useState(0);
  const [usdcAlt, setUsdcAlt] = useState(() => pref("usdc_lending_pct", 5));
  const [coinYield, setCoinYield] = useState(() => pref(`coin_yield_pct.${underlying}`, 0));
  const expiries = table(payload, "expiries");
  const rows = inventoryRows(expiries, { holdingUnits: holding, usdcAltPct: usdcAlt, coinYieldPct: coinYield }, spot);
  const inverse = table(payload, "opportunities")[0]?.call_px_unit === "coin" || ["BTC", "ETH"].includes(underlying);
  return (
    <section className="subsection">
      <div className="title sub-title">4) Inventory carry</div>
      <div className="toolbar form-grid">
        <NumberInput label={`Holding (${underlying}, not stored)`} value={holding} step="1" onChange={setHolding} />
        <NumberInput label="USDC alternative rate" value={usdcAlt} suffix="% ann." onChange={(v) => { setUsdcAlt(v); keep("usdc_lending_pct", v); }} />
        <NumberInput label={`${underlying} yield forgone (staking/lending)`} value={coinYield} suffix="% ann." onChange={(v) => { setCoinYield(v); keep(`coin_yield_pct.${underlying}`, v); }} />
      </div>
      <div className="dim">Hold {underlying}, sell the call and buy the put at K: you give up upside above K and are protected below K, and receive K USD at expiry.</div>
      {rows.length === 0 ? (
        <div className="notice">No expiry has a lockable rate in the ATM band.</div>
      ) : (
        <div className="table-wrap">
          <table className="grid">
            <thead>
              <tr>
                {["Expiry", "DTE", "Best strike", "r_lock", "Pickup vs USDC", "Net of forgone yield", "USD at expiry", "Option fees"].map((h) => (
                  <th key={h} className={h === "Expiry" ? undefined : "num"}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.expiry_ts}>
                  <td data-label="Expiry">{fmtExpiry(r.expiry_ts)}</td>
                  <td data-label="DTE" className="num">{fmtDte(r.dte)}</td>
                  <td data-label="Best strike" className="num">{fmtSpot(underlying, r.strike)}</td>
                  <td data-label="r_lock" className="num">{fmtRateAnn(r.r_lock)}</td>
                  <td data-label="Pickup vs USDC" className={`num ${r.pickup_vs_usdc > 0 ? "good" : "poor"}`}>{r.pickup_vs_usdc > 0 ? "+" : ""}{fmtRateAnn(r.pickup_vs_usdc)}</td>
                  <td data-label="Net of forgone yield" className={`num ${r.pickup_net > 0 ? "good" : "poor"}`}>{r.pickup_net > 0 ? "+" : ""}{fmtRateAnn(r.pickup_net)}</td>
                  <td data-label="USD at expiry" className="num">{fmtUsdTotal(r.usd_at_expiry)}</td>
                  <td data-label="Option fees" className="num">{r.fees === null ? "—" : inverse ? `${formatNumber(r.fees, 6)} ${underlying}` : fmtUsdTotal(r.fees)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="chart-footer dim">
        r_lock is the continuous USD rate locked on the spot index after taker option fees and delivery. Pickup subtracts your rates; the
        conversion only beats holding {underlying} when the last column is positive. USD at expiry compounds the holding's value at r_lock.
        Inputs stay in this browser.
      </div>
    </section>
  );
}
