// Data tab (guide §7.7): the audit trail and raw downloads.
import type { Composite, Curve, Table } from "../../api/payloads.ts";
import type { SuccessEnvelope, FailedEnvelope } from "../../api/types.ts";
import { formatUtc } from "../../format/units.ts";
import { DataTable } from "../../renderers/DataTable.tsx";
import { toCsv, type ChainRow } from "./logic.ts";

function download(name: string, type: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function sectionCsv(kind: string, payload: unknown): string | null {
  if (kind === "table") {
    const t = payload as Table;
    return toCsv(t.columns.map((c) => c.key), t.rows);
  }
  if (kind === "curve") {
    const c = payload as Curve;
    return toCsv([c.x.label, ...c.series.map((s) => s.key)], c.x.values.map((x, i) => [x, ...c.series.map((s) => s.values[i] ?? null)]));
  }
  return null;
}

export function DataTab({ envelope, payload, nodeRows }: { envelope: SuccessEnvelope | FailedEnvelope; payload: Composite; nodeRows: ChainRow[] }) {
  const quality = nodeRows.reduce<Record<string, number>>((acc, r) => {
    const q = String(r.node_quality);
    acc[q] = (acc[q] ?? 0) + 1;
    return acc;
  }, {});
  const expiries = payload.sections.find((s) => s.key === "expiries");
  return (
    <div className="panel">
      <div className="panel-body">
        <dl className="three">
          <div><dt className="label">Run</dt><dd className="num">{envelope.run_id}</dd></div>
          <div><dt className="label">Implementation</dt><dd>{envelope.implementation_id} {envelope.implementation_version}</dd></div>
          <div><dt className="label">Code</dt><dd className="num">{envelope.code.git_sha}</dd></div>
          <div><dt className="label">As of</dt><dd className="num">{formatUtc(envelope.as_of)}</dd></div>
          <div><dt className="label">Inputs</dt><dd>{envelope.inputs.sources.map((s) => `${s.name} @ ${formatUtc(s.snapshot_at)}`).join(" · ") || "—"}</dd></div>
          <div><dt className="label">Input hash</dt><dd className="num">{envelope.inputs.hash.slice(0, 19)}…</dd></div>
        </dl>
        <div className="notice">
          NODE QUALITY · {Object.entries(quality).map(([q, n]) => `${q.toUpperCase()} ${n}`).join(" · ") || "—"}
        </div>
        {envelope.warnings.length > 0 && (
          <ul className="notice">{envelope.warnings.map((w, i) => <li key={i}>{w.code}: {w.message}</li>)}</ul>
        )}
        {expiries && <DataTable payload={expiries.payload as Table} caption="Expiries" />}
        <div className="toolbar">
          <button className="btn" onClick={() => download(`${envelope.slot_id}_${envelope.scope.underlying}_${envelope.run_id}.json`, "application/json", JSON.stringify(envelope, null, 2))}>
            DOWNLOAD JSON (FULL RUN)
          </button>
          {payload.sections.map((s) => {
            const csv = sectionCsv(s.payload_kind, s.payload);
            return csv ? (
              <button key={s.key} className="btn" onClick={() => download(`${envelope.slot_id}_${s.key}.csv`, "text/csv", csv)}>CSV · {s.label}</button>
            ) : null;
          })}
        </div>
      </div>
    </div>
  );
}
