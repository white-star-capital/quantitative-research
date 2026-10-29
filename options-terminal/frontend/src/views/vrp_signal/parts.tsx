// Pieces shared by the vrp_signal view and the pricer view (guide §12): the monitor columns and
// the VRP tab. The VRP slot is found in the catalog by view id (F-2); both disappear without it.
import { useMemo } from "react";
import type { Composite } from "../../api/payloads.ts";
import type { LatestResponse, SlotSummary } from "../../api/types.ts";
import { usePolled, type Polled } from "../../api/usePolled.ts";
import { checkLatest, checkSlotDetail } from "../../api/validate.ts";
import { fmtVp } from "../../format/pricer.ts";
import { fmtZ } from "../../format/vrp.ts";
import { formatAge, formatUtc } from "../../format/units.ts";
import { runSelection } from "../bs_pricer/logic.ts";
import { latestPath, payloadOf } from "../bs_pricer/shared.ts";
import { ErrorIndicator, Loading, NoOutputYet, Unsupported } from "../states.tsx";
import { chips, contextItems, ctxNum, gate, rowAt, signalRows, VRP_COLUMNS, zTitle } from "./logic.ts";
import { ChipList, StatusWord, VrpPanels } from "./VrpPanels.tsx";

/** The VRP slot's run selection (production, else its first shadow), or null while loading. */
export function useVrpSelection(slot: SlotSummary | null): Record<string, string> | null {
  const detail = usePolled(slot ? `/v1/slots/${encodeURIComponent(slot.slot_id)}` : null, checkSlotDetail);
  return useMemo(() => (detail.data ? runSelection(detail.data.implementations) : null), [detail.data]);
}

export function useVrpLatest(slot: SlotSummary | null, underlying: string, selection: Record<string, string> | null): Polled<LatestResponse> {
  return usePolled(slot && selection ? latestPath(slot.slot_id, underlying, selection) : null, checkLatest);
}

/** The three monitor cells for one asset; one dim cell spanning them when unavailable. */
export function VrpCells({ payload, stale, why }: { payload: Composite | null; stale: boolean; why: string }) {
  const r30 = rowAt(signalRows(payload), 30);
  if (!payload || !r30) return <td data-label="VRP" colSpan={VRP_COLUMNS.length} className="dim">{payload ? "NO 30D ROW" : why}</td>;
  const ctx = contextItems(payload);
  const g = gate(stale, ctx);
  return (
    <>
      <td data-label="VRP fcst 30d" className={`num ${g.open ? "" : "dim"}`}>{fmtVp(r30.vrp)}</td>
      <td data-label="VRP z 30d" className={`num ${g.open ? "" : "dim"}`} title={zTitle(r30)}>{fmtZ(r30.vrp_z)}</td>
      <td data-label="VRP read 30d" title={g.reason ?? undefined}>
        <ChipList list={chips(r30, g, ctxNum(ctx, "event_hump_share"))} /> <StatusWord row={r30} g={g} />
      </td>
    </>
  );
}

export function whyMissing(latest: Polled<LatestResponse>): string {
  if (latest.loading) return "LOADING…";
  if (latest.problem?.slug === "no-output-yet") return "NO VRP OUTPUT YET";
  if (latest.data?.data.status === "failed") return "VRP RUN FAILED";
  return latest.error ?? latest.problem?.detail ?? "VRP UNAVAILABLE";
}

/** Pricer asset view → VRP tab. */
export function VrpTab({ slot, underlying }: { slot: SlotSummary; underlying: string }) {
  const selection = useVrpSelection(slot);
  const latest = useVrpLatest(slot, underlying, selection);
  const { data, unsupported, problem, error, loading } = latest;
  if (!data) {
    if (problem?.slug === "no-output-yet") return <NoOutputYet />;
    if (unsupported) return <Unsupported raw={unsupported.raw} reason={unsupported.reason} />;
    if (loading || !selection) return <Loading what={`${underlying} VRP RUN`} />;
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
        <div className="notice bad">VRP RUN FAILED · {env.error.code} · {env.error.message}</div>
      ) : (
        payload && <VrpPanels payload={payload} stale={data.meta.stale} />
      )}
    </>
  );
}
