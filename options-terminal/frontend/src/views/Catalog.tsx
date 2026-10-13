// Catalog (FRONTEND.md §4.1): slot list from GET /v1/slots, never hard-coded (F-2).
import { useMemo, useState } from "react";
import type { SlotSummary } from "../api/types.ts";
import { defaultScope, formatHash } from "../app/router.ts";
import { ErrorIndicator, Loading } from "./states.tsx";
import type { Polled } from "../api/usePolled.ts";
import { hasBespoke } from "./registry.ts";

export function sortSlots(slots: SlotSummary[]): SlotSummary[] {
  return [...slots].sort(
    (a, b) =>
      (a.display?.order ?? Number.MAX_SAFE_INTEGER) - (b.display?.order ?? Number.MAX_SAFE_INTEGER) ||
      a.title.localeCompare(b.title),
  );
}

export function slotHref(slot: SlotSummary): string {
  // Bespoke views may have a scope-less landing view (e.g. the cross-asset monitor).
  const scope = hasBespoke(slot) ? {} : (defaultScope(slot) ?? {});
  return formatHash({ name: "slot", slotId: slot.slot_id, scope });
}

export function Catalog({ slots }: { slots: Polled<SlotSummary[]> }) {
  const [filter, setFilter] = useState("");
  const rows = useMemo(() => {
    const q = filter.trim().toLowerCase();
    const all = sortSlots(slots.data ?? []);
    return q ? all.filter((s) => s.title.toLowerCase().includes(q)) : all;
  }, [slots.data, filter]);

  return (
    <section>
      <div className="section-title">
        <span>Slot catalog</span>
        <span>FROM GET /V1/SLOTS</span>
      </div>
      <div className="scope-bar">
        <label className="label" htmlFor="catalog-filter">
          Filter
        </label>
        <input
          id="catalog-filter"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="TITLE"
        />
      </div>
      {slots.error && <ErrorIndicator error={slots.error} lastGoodAsOf={null} />}
      {slots.problem && (
        <div className="notice bad">
          API ERROR {slots.problem.status} · {slots.problem.detail ?? slots.problem.slug ?? "NO DETAIL"}
        </div>
      )}
      {slots.loading && <Loading what="CATALOG" />}
      {slots.unsupported && (
        <div className="notice bad">CATALOG RESPONSE INVALID · {slots.unsupported.reason}</div>
      )}
      {!slots.loading && slots.data === null && !slots.unsupported && !slots.error && !slots.problem && (
        <div className="notice">NO CATALOG · THE API HAS NOT ANSWERED</div>
      )}
      {slots.data !== null && rows.length === 0 && <div className="notice">NO SLOTS</div>}
      {rows.length > 0 && (
        <div className="table-wrap">
          <table className="grid">
            <thead>
              <tr>
                <th>#</th>
                <th>Slot</th>
                <th>Kind</th>
                <th>Implementation</th>
                <th>Scope</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((s, i) => (
                <tr
                  key={s.slot_id}
                  className="clickable"
                  onClick={() => (window.location.hash = slotHref(s))}
                >
                  <td data-label="#" className="row-code">
                    {i + 1})
                  </td>
                  <td data-label="Slot">
                    <a href={slotHref(s)}>{s.title}</a>
                    <div className="dim">{s.description}</div>
                  </td>
                  <td data-label="Kind">{s.payload_kind.toUpperCase()}</td>
                  <td data-label="Implementation">
                    {s.production_impl ? `${s.production_impl} ${s.production_version ?? ""}` : "—"}
                  </td>
                  <td data-label="Scope">{s.scope.dimensions.join(", ").toUpperCase()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
