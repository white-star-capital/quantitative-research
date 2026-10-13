// Chain tab (guide §7.4): T-layout per expiry, column presets, unit and quantity toggles.
// Non-good nodes are dimmed; ITM rows priced off the smile show their IV in italics.
import { Fragment, useMemo, useState } from "react";
import type { Composite } from "../../api/payloads.ts";
import { fmtDelta, fmtDte, fmtExpiry, fmtGamma, fmtGreekUsd, fmtIv, fmtPct, fmtPremium, fmtSpot, fmtVp } from "../../format/pricer.ts";
import { MISSING, formatNumber } from "../../format/units.ts";
import {
  COIN_ASSETS, chainOf, convertPrice, defaultExpiry, expiriesOf, loadPref, savePref, strikeRows,
  type ChainRow, type PriceUnit, type Quantity,
} from "./logic.ts";
import type { Prefill } from "./PricerTab.tsx";
import { contextOf, num, storage } from "./shared.ts";

type Preset = "trader" | "greeks" | "yield" | "vol";

interface Col {
  label: string;
  cell: (r: ChainRow) => string;
}

export function ChainTab({ underlying, payload, onPrice }: { underlying: string; payload: Composite; onPrice: (p: Prefill) => void }) {
  const expiries = expiriesOf(payload);
  const chain = useMemo(() => chainOf(payload), [payload]);
  const spot = num(contextOf(payload).spot) ?? 0;
  const [expiry, setExpiry] = useState<string | null>(() => defaultExpiry(expiries));
  const [preset, setPreset] = useState<Preset>(() => loadPref("chain_preset", "trader", storage()));
  const [unit, setUnit] = useState<PriceUnit>("usd");
  const [qty, setQty] = useState<Quantity>("unit");
  const [size, setSize] = useState("1");
  const current = expiries.find((e) => e.expiry_ts === expiry) ?? expiries[0];
  if (!current) return <div className="notice">NO EXPIRIES IN THIS RUN</div>;
  const rows = strikeRows(chain, current.expiry_ts);
  const coinOk = COIN_ASSETS.has(underlying);

  const price = (r: ChainRow, key: string): string => {
    const v = convertPrice(r[key] as number | null, unit, qty, {
      spot,
      forward: current.forward,
      contractSize: (r.contract_size as number | null) ?? 1,
      positionSize: Number(size) || 0,
    });
    if (v === null) return MISSING;
    if (unit === "pct") return `${formatNumber(v * 100, 2)}%`;
    if (unit === "coin") return formatNumber(v, 4);
    return fmtPremium(v);
  };
  const n = (r: ChainRow, k: string) => (typeof r[k] === "number" ? (r[k] as number) : null);
  const spread = (r: ChainRow) => {
    const b = n(r, "iv_bid"), a = n(r, "iv_ask");
    return b === null || a === null ? MISSING : fmtVp(a - b);
  };

  const PRESETS: Record<Preset, Col[]> = {
    trader: [
      { label: "Bid", cell: (r) => price(r, "venue_bid_usd") },
      { label: "Fair", cell: (r) => price(r, "fair_usd") },
      { label: "Ask", cell: (r) => price(r, "venue_ask_usd") },
      { label: "IV", cell: (r) => fmtIv(n(r, "iv_mark")) },
      { label: "Δ", cell: (r) => fmtDelta(n(r, "delta")) },
    ],
    greeks: [
      { label: "Δ spot", cell: (r) => fmtDelta(n(r, "delta_spot")) },
      { label: "Γ 1%", cell: (r) => fmtGamma(n(r, "gamma_1pct")) },
      { label: "Vega", cell: (r) => fmtGreekUsd(n(r, "vega_1pt")) },
      { label: "Θ/day", cell: (r) => fmtGreekUsd(n(r, "theta_1d")) },
    ],
    yield: [
      { label: "Fair", cell: (r) => price(r, "fair_usd") },
      { label: "% spot", cell: (r) => fmtPct(n(r, "fair_pct_spot")) },
      { label: "Yield ann.", cell: (r) => fmtPct(n(r, "yield_ann")) },
      { label: "P(ITM)", cell: (r) => fmtPct(n(r, "p_itm")) },
      { label: "BE", cell: (r) => fmtSpot(underlying, n(r, "breakeven")) },
    ],
    vol: [
      { label: "IV bid", cell: (r) => fmtIv(n(r, "iv_bid")) },
      { label: "IV mark", cell: (r) => fmtIv(n(r, "iv_mark")) },
      { label: "IV ask", cell: (r) => fmtIv(n(r, "iv_ask")) },
      { label: "Spread", cell: spread },
      { label: "Quality", cell: (r) => String(r.node_quality ?? MISSING).toUpperCase() },
    ],
  };
  const cols = PRESETS[preset];

  const side = (r: ChainRow | null, isCall: boolean, strike: number) =>
    cols.map((c) => {
      if (!r) return <td key={c.label} className="num dim">{MISSING}</td>;
      const itm = isCall ? strike < current.forward : strike > current.forward;
      const quality = String(r.node_quality);
      const cls = ["num", itm ? "itm" : "", r.is_node === true && quality !== "good" ? "weak" : "", quality === "interpolated" && c.label.startsWith("IV") ? "interp" : ""];
      const open = () => onPrice({ optionType: isCall ? "call" : "put", expiry: current.expiry_ts, strike });
      return (
        <td
          key={c.label}
          className={[...cls, "clickable-cell"].filter(Boolean).join(" ")}
          title={r.is_node === true ? `${quality} node · spread ${spread(r)}` : "priced off the smile (not a node)"}
          data-label={`${isCall ? "C" : "P"} ${c.label}`}
          tabIndex={0}
          role="button"
          onClick={open}
          onKeyDown={(e) => {
            if (e.key === "Enter") open();
          }}
        >
          {c.cell(r)}
        </td>
      );
    });

  let markerDone = false;

  return (
    <div className="panel">
      <div className="panel-body">
        <div className="chips" role="group" aria-label="Expiry">
          {expiries.map((e) => (
            <button key={e.expiry_ts} className={e.expiry_ts === current.expiry_ts ? "active" : ""} onClick={() => setExpiry(e.expiry_ts)}>
              {fmtExpiry(e.expiry_ts).slice(0, 9)} · {fmtDte(e.dte)} · {fmtIv(e.atm_iv)}
            </button>
          ))}
        </div>
        <div className="toolbar">
          <label className="label">Preset{" "}
            <select value={preset} onChange={(e) => { const v = e.target.value as Preset; setPreset(v); savePref("chain_preset", v, storage()); }}>
              <option value="trader">Trader</option>
              <option value="greeks">Greeks</option>
              <option value="yield">Yield</option>
              <option value="vol">Vol</option>
            </select>
          </label>
          <label className="label">Units{" "}
            <select value={unit} onChange={(e) => setUnit(e.target.value as PriceUnit)}>
              <option value="usd">USD</option>
              <option value="pct">% of spot</option>
              {coinOk && <option value="coin">Coin ({underlying})</option>}
            </select>
          </label>
          <label className="label">Quantity{" "}
            <select value={qty} onChange={(e) => setQty(e.target.value as Quantity)}>
              <option value="unit">Per unit</option>
              <option value="contract">Per contract</option>
              <option value="position">Per position</option>
            </select>
          </label>
          {qty === "position" && (
            <label className="label">Size{" "}<input value={size} onChange={(e) => setSize(e.target.value)} style={{ width: 80 }} /></label>
          )}
          <span className="dim">FORWARD {fmtSpot(underlying, current.forward)} · {fmtExpiry(current.expiry_ts)}</span>
        </div>
        <div className="table-wrap">
          <table className="grid">
            <thead>
              <tr>
                {cols.map((c) => <th key={`c${c.label}`} className="num">C {c.label}</th>)}
                <th style={{ textAlign: "center" }}>Strike</th>
                {cols.map((c) => <th key={`p${c.label}`} className="num">P {c.label}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const marker = !markerDone && row.strike > current.forward;
                if (marker) markerDone = true;
                return (
                  <Fragment key={row.strike}>
                    {marker && (
                      <tr className="fwd"><td colSpan={cols.length * 2 + 1}>FORWARD {fmtSpot(underlying, current.forward)}</td></tr>
                    )}
                    <tr>
                      {side(row.call, true, row.strike)}
                      <td className="strike num" data-label="Strike">{fmtSpot(underlying, row.strike)}</td>
                      {side(row.put, false, row.strike)}
                    </tr>
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="chart-footer dim">
          Fair = Black-76 on the expiry forward at the node's mark IV (ITM rows use the smile, so a call and put at one strike share an IV).
          Shaded cells are in the money; dimmed cells are nodes without a tight two-sided quote. Click a price to open it in the pricer.
        </div>
      </div>
    </div>
  );
}

