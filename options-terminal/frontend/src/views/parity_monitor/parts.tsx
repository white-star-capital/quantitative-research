// Pieces shared by the parity_monitor view and the pricer view (guide §7.1, §7.2): the monitor
// columns and the Parity tab. The parity slot is found in the catalog by view id (F-2).
import { useMemo } from "react";
import type { Composite } from "../../api/payloads.ts";
import type { SlotSummary } from "../../api/types.ts";
import { usePolled, type Polled } from "../../api/usePolled.ts";
import { checkLatest, checkSlotDetail } from "../../api/validate.ts";
import type { LatestResponse } from "../../api/types.ts";
import { fmtBp, fmtRateAnn } from "../../format/parity.ts";
import { formatAge, formatUtc } from "../../format/units.ts";
import { runSelection } from "../bs_pricer/logic.ts";
import { latestPath, payloadOf } from "../bs_pricer/shared.ts";
import { ErrorIndicator, Loading, NoOutputYet, Unsupported } from "../states.tsx";
import { at30d, context, perpOnly, table } from "./logic.ts";
import { ParityPanels } from "./ParityPanels.tsx";

export const PARITY_COLUMNS = [
  "Implied basis 30d", "Future basis 30d", "Options vs futures", "Funding (ann.)", "Best net edge (ann.)", "Confirmed opps", "Inventory r_lock 30d",
];

/** The parity slot's run selection (production, else its first shadow), or null while loading. */
export function useParitySelection(slot: SlotSummary | null): Record<string, string> | null {
  const detail = usePolled(slot ? `/v1/slots/${encodeURIComponent(slot.slot_id)}` : null, checkSlotDetail);
  return useMemo(() => (detail.data ? runSelection(detail.data.implementations) : null), [detail.data]);
}

export function useParityLatest(slot: SlotSummary | null, underlying: string, selection: Record<string, string> | null): Polled<LatestResponse> {
  return usePolled(slot && selection ? latestPath(slot.slot_id, underlying, selection) : null, checkLatest);
}

/** The seven monitor cells for one asset; one dim cell spanning them when unavailable. */
export function ParityCells({ payload, why }: { payload: Composite | null; why: string }) {
  if (!payload) return <td data-label="Parity" colSpan={PARITY_COLUMNS.length} className="dim">{why}</td>;
  const ctx = context(payload);
  const expiries = table(payload, "expiries");
  const perp = perpOnly(expiries);
  const confirmed = typeof ctx.n_confirmed === "number" ? ctx.n_confirmed : null;
  return (
    <>
      <td data-label="Implied basis 30d" className="num">{fmtRateAnn(at30d(expiries, "implied_basis_ann"))}</td>
      <td data-label="Future basis 30d" className="num">{perp ? <span className="dim">perp only</span> : fmtRateAnn(at30d(expiries, "future_basis_ann"))}</td>
      <td data-label="Options vs futures" className="num">{perp ? <span className="dim">perp only</span> : fmtBp(at30d(expiries, "options_vs_future_bps"))}</td>
      <td data-label="Funding (ann.)" className="num">{fmtRateAnn(typeof ctx.funding_ann === "number" ? ctx.funding_ann : null)}</td>
      <td data-label="Best net edge (ann.)" className="num">{fmtRateAnn(typeof ctx.best_net_edge_ann === "number" ? ctx.best_net_edge_ann : null)}</td>
      <td data-label="Confirmed opps" className={`num ${confirmed ? "good" : "dim"}`}>{confirmed ?? "—"}</td>
      <td data-label="Inventory r_lock 30d" className="num">{fmtRateAnn(at30d(expiries, "r_lock_best"))}</td>
    </>
  );
}

export function whyMissing(latest: Polled<LatestResponse>): string {
  if (latest.loading) return "LOADING…";
  if (latest.problem?.slug === "no-output-yet") return "NO PARITY OUTPUT YET";
  return latest.error ?? latest.problem?.detail ?? "PARITY UNAVAILABLE";
}

/** Pricer asset view → Parity tab (guide §7.2). */
export function ParityTab({ slot, underlying }: { slot: SlotSummary; underlying: string }) {
  const selection = useParitySelection(slot);
  const latest = useParityLatest(slot, underlying, selection);
  const { data, unsupported, problem, error, loading } = latest;
  if (!data) {
    if (problem?.slug === "no-output-yet") return <NoOutputYet />;
    if (unsupported) return <Unsupported raw={unsupported.raw} reason={unsupported.reason} />;
    if (loading || !selection) return <Loading what={`${underlying} PARITY RUN`} />;
    return <ErrorIndicator error={error ?? problem?.detail ?? "NO DATA"} lastGoodAsOf={null} />;
  }
  const env = data.data;
  const payload = payloadOf(data);
  return (
    <>
      <div className="panel-meta">
        {slot.title} · {env.implementation_id} {env.implementation_version}{data.meta.role === "shadow" ? " · SHADOW" : ""} · AS OF{" "}
        <span className="num">{formatUtc(env.as_of)}</span> · {formatAge(data.meta.age_seconds)}{" "}
        <span className={`status ${data.meta.stale ? "bad" : "ok"}`}>{data.meta.stale ? "STALE" : "FRESH"}</span>
      </div>
      {env.status === "failed" ? (
        <div className="notice bad">PARITY RUN FAILED · {env.error.code} · {env.error.message}</div>
      ) : (
        payload && <ParityPanels env={env} payload={payload} underlying={underlying} />
      )}
    </>
  );
}
