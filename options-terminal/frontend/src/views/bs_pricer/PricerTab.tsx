// Pricer & OTC tab (guide §7.6). Prices any strike/expiry against the run on screen (pinned by
// run_id) and evaluates OTC quotes. The counterparty label and cost basis never leave the browser
// (P-8); bands are display-only and use the viewer's own thresholds.
import { useState, type FormEvent } from "react";
import { fetchJson, problemSlug } from "../../api/client.ts";
import { API_BASE_URL } from "../../config.ts";
import {
  fmtDelta, fmtDte, fmtExpiry, fmtGamma, fmtGreekUsd, fmtIv, fmtPct, fmtPremium, fmtSpot,
  fmtUsdTotal, fmtVpPoints,
} from "../../format/pricer.ts";
import { MISSING } from "../../format/units.ts";
import { LineChart } from "../../renderers/chart/LineChart.tsx";
import { AMBER, CYAN } from "../../renderers/chart/palette.ts";
import { useWidth } from "../../renderers/chart/useWidth.ts";
import {
  BAND_LABEL, DEFAULT_THRESHOLDS, band, evaluateParams, loadPref, payoff, query, savePref, toCsv,
  type Band, type QuoteInput, type Thresholds,
} from "./logic.ts";
import { storage } from "./shared.ts";

export interface Prefill {
  optionType: "call" | "put";
  expiry: string;
  strike: number;
}

interface Greeks { delta: number; delta_spot: number; gamma_1pct: number; vega_1pt: number; theta_1d: number; p_itm?: number }
interface EvalData {
  run_id: string;
  spot: number;
  contract: { option_type: string; expiry: string; strike: number; size: number; dte: number };
  market: { forward: number; iv_mark: number; iv_bid: number | null; iv_ask: number | null; interpolation: string; screen_bid_usd: number | null; screen_ask_usd: number | null };
  fair: { per_unit_usd: number; total_usd: number; pct_spot: number; greeks_per_unit: Greeks; greeks_position: Greeks; breakeven: number; yield_ann: number };
  quote: null | {
    side: "sell" | "buy"; premium_per_unit_usd: number; premium_total_usd: number; iv_quote: number | null; iv_quote_status: string;
    pct_of_fair: number | null; edge_per_unit_usd: number; edge_total_usd: number; vol_edge_vp: number | null;
    screen_position: string; screen_position_frac: number | null; fair_at_spot_ref: number; yield_ann: number;
  };
  distribution: { basis: string; sigma: number; x: number[]; pdf: number[] };
  flags: { code: string; message: string }[];
}

const FIELD_OF: Record<string, keyof QuoteInput | "general"> = {
  invalid_expiry: "expiry", expiry_in_past: "expiry", tenor_out_of_range: "expiry",
  invalid_strike: "strike", strike_out_of_range: "strike", invalid_size: "size",
  invalid_premium: "premium", premium_requires_side: "side", premium_requires_unit: "premiumUnit",
  pct_spot_requires_spot_ref: "spotRef", invalid_spot_ref: "spotRef", invalid_quoted_at: "quotedAt",
  invalid_option_type: "optionType",
};

const REASON_TEXT: Record<string, string> = {
  below_intrinsic: "premium is at or below intrinsic value",
  above_max_price: "premium is above the maximum possible option value",
  below_min_vol: "implied vol would be below 1%",
  above_max_vol: "implied vol would be above 500%",
};

interface BlotterRow { id: number; label: string; data: EvalData }

function splitExpiry(iso: string): [string, string] {
  return [iso.slice(0, 10), iso.slice(11, 16)];
}

function isEval(body: unknown): body is { data: EvalData } {
  const d = (body as { data?: EvalData })?.data;
  return !!d && typeof d.fair?.per_unit_usd === "number" && Array.isArray(d.flags) && Array.isArray(d.distribution?.x);
}

export function PricerTab(props: { slotId: string; underlying: string; selection: Record<string, string>; runId: string; spot: number | null; prefill: Prefill | null }) {
  const { slotId, underlying, runId, prefill } = props;
  const [date0, time0] = prefill ? splitExpiry(prefill.expiry) : ["", "08:00"];
  const [date, setDate] = useState(date0);
  const [time, setTime] = useState(time0);
  const [input, setInput] = useState<QuoteInput>({
    optionType: prefill?.optionType ?? "call", expiry: "", strike: prefill ? String(prefill.strike) : "", size: "1",
    side: "", premium: "", premiumUnit: "usd_per_unit", spotRef: "", quotedAt: "", counterparty: "",
  });
  const [thresholds, setThresholds] = useState<Thresholds>(() => loadPref("thresholds", DEFAULT_THRESHOLDS, storage()));
  const [costBasis, setCostBasis] = useState("");
  const [result, setResult] = useState<EvalData | null>(null);
  const [errors, setErrors] = useState<Partial<Record<keyof QuoteInput | "general", string>>>({});
  const [busy, setBusy] = useState(false);
  const [blotter, setBlotter] = useState<BlotterRow[]>([]);

  const set = <K extends keyof QuoteInput>(k: K, v: QuoteInput[K]) => setInput((s) => ({ ...s, [k]: v }));

  const run = async (): Promise<EvalData | null> => {
    const params = evaluateParams(underlying, { ...input, expiry: `${date}T${time}:00Z` }, { run_id: runId });
    setBusy(true);
    setErrors({});
    try {
      const res = await fetchJson(API_BASE_URL, `/v1/slots/${encodeURIComponent(slotId)}/evaluate${query(params)}`);
      if (res.kind === "ok" && isEval(res.body)) return res.body.data;
      if (res.kind === "problem") {
        const slug = problemSlug(res.problem) ?? "general";
        setErrors({ [FIELD_OF[slug] ?? "general"]: res.problem?.detail ?? res.problem?.title ?? `ERROR ${res.status}` });
      } else {
        setErrors({ general: res.kind === "network_error" ? `NETWORK ERROR: ${res.message}` : "UNEXPECTED RESPONSE FROM THE API" });
      }
      return null;
    } finally {
      setBusy(false);
    }
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const data = await run();
    if (data) setResult(data);
  };

  const addToBlotter = async () => {
    const data = await run();
    if (!data) return;
    setResult(data);
    setBlotter((b) => [...b, { id: Date.now(), label: input.counterparty, data }]);
  };

  const err = (k: keyof QuoteInput) => errors[k] && <span className="field-error" role="alert">{errors[k]}</span>;
  const q = result?.quote ?? null;
  const verdict: Band | null = q ? band(q.side, q.pct_of_fair, q.screen_position, thresholds) : null;

  return (
    <div className="panel">
      <div className="panel-body">
        <form onSubmit={submit}>
          <div className="form-grid">
            <label><span className="label">Type</span>
              <select value={input.optionType} onChange={(e) => set("optionType", e.target.value as "call" | "put")}>
                <option value="call">Call</option><option value="put">Put</option>
              </select>{err("optionType")}
            </label>
            <label><span className="label">Expiry date (UTC)</span>
              <input type="date" value={date} onChange={(e) => setDate(e.target.value)} required />{err("expiry")}
            </label>
            <label><span className="label">Expiry time (UTC)</span>
              <span className="toolbar" style={{ padding: 0 }}>
                <input type="time" value={time} onChange={(e) => setTime(e.target.value)} required />
                <button type="button" className="btn" onClick={() => setTime("08:00")}>08:00</button>
                <button type="button" className="btn" onClick={() => setTime("23:59")}>23:59</button>
              </span>
            </label>
            <label><span className="label">Strike</span><input inputMode="decimal" value={input.strike} onChange={(e) => set("strike", e.target.value)} required />{err("strike")}</label>
            <label><span className="label">Size ({underlying})</span><input inputMode="decimal" value={input.size} onChange={(e) => set("size", e.target.value)} />{err("size")}</label>
            <label><span className="label">Side (ours)</span>
              <select value={input.side} onChange={(e) => set("side", e.target.value as QuoteInput["side"])}>
                <option value="">— price only</option><option value="sell">We sell</option><option value="buy">We buy</option>
              </select>{err("side")}
            </label>
            <label><span className="label">Premium</span><input inputMode="decimal" value={input.premium} onChange={(e) => set("premium", e.target.value)} placeholder="optional" />{err("premium")}</label>
            <label><span className="label">Premium unit</span>
              <select value={input.premiumUnit} onChange={(e) => set("premiumUnit", e.target.value as QuoteInput["premiumUnit"])}>
                <option value="usd_per_unit">USD per unit</option><option value="usd_total">USD total</option>
                <option value="pct_spot">% of spot ref</option><option value="coin_per_unit">Coin per unit</option>
              </select>{err("premiumUnit")}
            </label>
            <label><span className="label">Spot ref</span><input inputMode="decimal" value={input.spotRef} onChange={(e) => set("spotRef", e.target.value)} placeholder="counterparty's spot" />{err("spotRef")}</label>
            <label><span className="label">Quoted at (UTC)</span><input value={input.quotedAt} onChange={(e) => set("quotedAt", e.target.value)} placeholder="2026-10-13T16:30:00Z" />{err("quotedAt")}</label>
            <label><span className="label">Counterparty (this tab only)</span><input value={input.counterparty} onChange={(e) => set("counterparty", e.target.value)} autoComplete="off" placeholder="never sent" /></label>
          </div>
          <div className="toolbar">
            <button className="btn primary" type="submit" disabled={busy}>{busy ? "PRICING…" : "PRICE <GO>"}</button>
            <button className="btn" type="button" disabled={busy || !input.premium} onClick={addToBlotter}>ADD TO BLOTTER</button>
            <span className="dim">Priced against run {runId.slice(0, 8)}…; the counterparty label stays in this browser tab.</span>
          </div>
          {errors.general && <div className="notice bad" role="alert">{errors.general}</div>}
        </form>

        {result && (
          <>
            {q && (
              <div className="verdict">
                <Big label="Vol edge (good for us +)" value={fmtVpPoints(q.vol_edge_vp)} tone={q.vol_edge_vp === null ? "" : q.vol_edge_vp >= 0 ? "good" : "poor"} />
                <Big label="Edge total" value={fmtUsdTotal(q.edge_total_usd, true)} tone={q.edge_total_usd >= 0 ? "good" : "poor"} />
                <Big label="% of fair" value={fmtPct(q.pct_of_fair)} tone="" />
                <Big label="Band (your thresholds)" value={verdict ? BAND_LABEL[verdict] : MISSING} tone={verdict === "at_or_above_floor" ? "good" : verdict === "negotiate" ? "mid" : verdict ? "poor" : ""} />
              </div>
            )}
            {result.flags.length > 0 && (
              <ul className="notice">
                {result.flags.map((f) => <li key={f.code}><span className="badge">{f.code.replaceAll("_", " ").toUpperCase()}</span> {f.message}</li>)}
              </ul>
            )}
            {q && q.iv_quote === null && (
              <div className="notice bad">QUOTE IV UNSOLVABLE · {REASON_TEXT[q.iv_quote_status] ?? q.iv_quote_status}</div>
            )}
            <Ladder bid={result.market.screen_bid_usd} ask={result.market.screen_ask_usd} fair={result.fair.per_unit_usd} quote={q?.premium_per_unit_usd ?? null} />
            <div className="two-col">
              <dl className="tiles">
                <Kv k="Fair / unit" v={fmtPremium(result.fair.per_unit_usd)} />
                <Kv k="Fair total" v={fmtUsdTotal(result.fair.total_usd)} />
                <Kv k="IV mark" v={`${fmtIv(result.market.iv_mark)} ${result.market.interpolation === "extrapolated" ? "· OFF-GRID" : ""}`} />
                <Kv k="IV bid / ask" v={`${fmtIv(result.market.iv_bid)} / ${fmtIv(result.market.iv_ask)}`} />
                <Kv k="Forward" v={fmtSpot(underlying, result.market.forward)} />
                <Kv k="Expiry" v={`${fmtExpiry(result.contract.expiry)} · ${fmtDte(result.contract.dte)}`} />
                <Kv k="Breakeven" v={fmtSpot(underlying, result.fair.breakeven)} />
                <Kv k="Yield (APR)" v={fmtPct(q ? q.yield_ann : result.fair.yield_ann)} />
                {q && <Kv k="Fair at spot ref" v={fmtPremium(q.fair_at_spot_ref)} />}
              </dl>
              <dl className="tiles">
                <Kv k="Δ fwd / spot" v={`${fmtDelta(result.fair.greeks_per_unit.delta)} / ${fmtDelta(result.fair.greeks_per_unit.delta_spot)}`} />
                <Kv k="Γ 1%" v={fmtGamma(result.fair.greeks_per_unit.gamma_1pct)} />
                <Kv k="Vega (1 vp)" v={fmtGreekUsd(result.fair.greeks_per_unit.vega_1pt)} />
                <Kv k="Θ / day" v={fmtGreekUsd(result.fair.greeks_per_unit.theta_1d)} />
                <Kv k="P(ITM)" v={fmtPct(result.fair.greeks_per_unit.p_itm)} />
                <Kv k="Position Δ spot" v={fmtDelta(result.fair.greeks_position.delta_spot)} />
                <Kv k="Position vega" v={fmtGreekUsd(result.fair.greeks_position.vega_1pt)} />
                <Kv k="Position Θ / day" v={fmtGreekUsd(result.fair.greeks_position.theta_1d)} />
              </dl>
            </div>
            <Payoff data={result} underlying={underlying} costBasis={Number(costBasis) || null} />
            <div className="toolbar">
              <label className="label">Cost basis (this tab only){" "}<input inputMode="decimal" value={costBasis} onChange={(e) => setCostBasis(e.target.value)} style={{ width: 100 }} /></label>
              <label className="label">Floor % of fair{" "}
                <input type="number" value={Math.round(thresholds.floor * 100)} style={{ width: 70 }} onChange={(e) => { const t = { ...thresholds, floor: Number(e.target.value) / 100 }; setThresholds(t); savePref("thresholds", t, storage()); }} />
              </label>
              <label className="label">Walk-away % of fair{" "}
                <input type="number" value={Math.round(thresholds.walkAway * 100)} style={{ width: 70 }} onChange={(e) => { const t = { ...thresholds, walkAway: Number(e.target.value) / 100 }; setThresholds(t); savePref("thresholds", t, storage()); }} />
              </label>
            </div>
          </>
        )}

        {blotter.length > 0 && <Blotter rows={blotter} thresholds={thresholds} underlying={underlying} onClear={() => setBlotter([])} />}
      </div>
    </div>
  );
}

function Big({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className={`tile-value num ${tone}`}>{value}</div>
    </div>
  );
}

function Kv({ k, v }: { k: string; v: string }) {
  return (
    <div className="tile">
      <dt className="label">{k}</dt>
      <dd className="num" style={{ margin: 0, textAlign: "left" }}>{v}</dd>
    </div>
  );
}

/** Screen ladder: bid→ask bar with fair and quote markers. "Inside the market or not" at a glance. */
function Ladder({ bid, ask, fair, quote }: { bid: number | null; ask: number | null; fair: number; quote: number | null }) {
  const vals = [bid, ask, fair, quote].filter((v): v is number => v !== null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.15 || Math.abs(fair) * 0.05 || 1;
  const [ref, W] = useWidth<HTMLDivElement>();
  const H = 44;
  const x = (v: number) => 10 + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (W - 20);
  const mark = (v: number, label: string, color: string, row: number) => (
    <g key={label}>
      <line x1={x(v)} x2={x(v)} y1={12} y2={30} stroke={color} strokeWidth={2} />
      <text className="chart-text" x={x(v)} y={row} textAnchor="middle" fill={color}>{label} {fmtPremium(v)}</text>
    </g>
  );
  return (
    <div className="chart" ref={ref}>
      <div className="label">Screen ladder</div>
      {W > 0 && (
      <svg width={W} height={H} role="img" aria-label="Screen bid, ask, fair and quote">
        {bid !== null && ask !== null ? (
          <rect x={x(bid)} y={18} width={Math.max(1, x(ask) - x(bid))} height={6} fill="#3a3a3a" />
        ) : (
          <text className="chart-text" x={10} y={24}>NO TWO-SIDED SCREEN FOR THIS CONTRACT</text>
        )}
        {bid !== null && mark(bid, "BID", "#8d887c", 42)}
        {ask !== null && mark(ask, "ASK", "#8d887c", 42)}
        {mark(fair, "FAIR", AMBER, 10)}
        {quote !== null && mark(quote, "QUOTE", CYAN, 42)}
      </svg>
      )}
    </div>
  );
}

function Payoff({ data, underlying, costBasis }: { data: EvalData; underlying: string; costBasis: number | null }) {
  const q = data.quote;
  const isCall = data.contract.option_type === "call";
  const side = q?.side ?? "buy";
  const premium = q?.premium_per_unit_usd ?? data.fair.per_unit_usd;
  const xs = data.distribution.x;
  const opt = payoff(xs, { isCall, strike: data.contract.strike, premium, size: data.contract.size, side });
  const series = [{ key: "option", label: side === "sell" ? "Short option" : "Long option", values: opt as (number | null)[] }];
  if (costBasis && isCall && side === "sell") {
    const hold = xs.map((s) => (s - costBasis) * data.contract.size);
    series.push({ key: "hold", label: "Hold only", values: hold });
    series.push({ key: "covered", label: "Covered call", values: hold.map((h, i) => h + (opt[i] as number)) });
  }
  return (
    <LineChart
      title="Payoff at expiry"
      x={xs}
      xLabel="Spot at expiry"
      series={series}
      formatX={(v) => fmtSpot(underlying, v)}
      formatY={(v) => fmtUsdTotal(v, true)}
      markers={[{ x: data.spot, label: "SPOT" }, { x: data.contract.strike, label: "K" }, { x: data.fair.breakeven, label: "BE" }].filter((m) => m.x >= (xs[0] ?? 0) && m.x <= (xs[xs.length - 1] ?? 0))}
      silhouette={{ values: data.distribution.pdf, label: "lognormal at ATM vol" }}
      pnlFill={series.length === 1}
      footer={`P&L per position at expiry, from our side, using ${q ? "the quoted" : "the fair"} premium. Grey shape: lognormal distribution at the expiry's ATM vol (${fmtIv(data.distribution.sigma)}), scaled to fit; it ignores skew.`}
    />
  );
}

function Blotter({ rows, thresholds, underlying, onClear }: { rows: BlotterRow[]; thresholds: Thresholds; underlying: string; onClear: () => void }) {
  const ranked = [...rows].sort((a, b) => (b.data.quote?.vol_edge_vp ?? -Infinity) - (a.data.quote?.vol_edge_vp ?? -Infinity));
  const line = (r: BlotterRow) => {
    const q = r.data.quote;
    const b = q ? band(q.side, q.pct_of_fair, q.screen_position, thresholds) : null;
    return {
      label: r.label || MISSING, expiry: fmtExpiry(r.data.contract.expiry), strike: fmtSpot(underlying, r.data.contract.strike),
      type: r.data.contract.option_type, size: r.data.contract.size, premium: q ? fmtUsdTotal(q.premium_total_usd) : MISSING,
      fair: fmtUsdTotal(r.data.fair.total_usd), pct: fmtPct(q?.pct_of_fair), edge: fmtVpPoints(q?.vol_edge_vp),
      screen: q?.screen_position ?? MISSING, band: b ? BAND_LABEL[b] : MISSING, flags: r.data.flags.map((f) => f.code).join(" "),
      raw: { premium: q?.premium_total_usd ?? null, fair: r.data.fair.total_usd, pct: q?.pct_of_fair ?? null, edge: q?.vol_edge_vp ?? null },
    };
  };
  const download = () => {
    const header = ["label", "type", "expiry", "strike", "size", "premium_total_usd", "fair_total_usd", "pct_of_fair", "vol_edge_vp", "screen_position", "band", "flags"];
    const body = ranked.map((r) => {
      const l = line(r);
      return [r.label, l.type, r.data.contract.expiry, r.data.contract.strike, l.size, l.raw.premium, l.raw.fair, l.raw.pct, l.raw.edge, l.screen, l.band, l.flags];
    });
    const url = URL.createObjectURL(new Blob([toCsv(header, body)], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `otc_blotter_${underlying}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <section className="subsection">
      <div className="title sub-title">Blotter · ranked by vol edge</div>
      <div className="table-wrap">
        <table className="grid">
          <thead><tr>{["Label", "Type", "Expiry", "Strike", "Size", "Premium", "Fair", "% fair", "Vol edge", "Screen", "Band", "Flags"].map((h) => <th key={h}>{h}</th>)}</tr></thead>
          <tbody>
            {ranked.map((r) => {
              const l = line(r);
              return (
                <tr key={r.id}>
                  <td data-label="Label">{l.label}</td><td data-label="Type">{l.type.toUpperCase()}</td><td data-label="Expiry">{l.expiry}</td>
                  <td data-label="Strike" className="num">{l.strike}</td><td data-label="Size" className="num">{l.size}</td>
                  <td data-label="Premium" className="num">{l.premium}</td><td data-label="Fair" className="num">{l.fair}</td>
                  <td data-label="% fair" className="num">{l.pct}</td><td data-label="Vol edge" className="num">{l.edge}</td>
                  <td data-label="Screen">{l.screen.toUpperCase()}</td><td data-label="Band">{l.band}</td><td data-label="Flags" className="dim">{l.flags}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="toolbar">
        <button className="btn" onClick={download}>EXPORT CSV</button>
        <button className="btn" onClick={onClear}>CLEAR</button>
        <span className="dim">The blotter lives in this tab only; labels appear only in the CSV you export.</span>
      </div>
    </section>
  );
}
