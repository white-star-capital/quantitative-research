import { useMemo, useState } from "react";
import type { Cell, Table } from "../api/payloads.ts";
import { formatCell, unitLabel } from "../format/units.ts";

const MAX_ROWS = 500;

function cmp(a: Cell, b: Cell): number {
  if (a === b) return 0;
  if (a === null) return 1; // nulls last
  if (b === null) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b));
}

/** table: sortable, unit-formatted, numbers right-aligned; scrolls inside its panel. */
export function DataTable({ payload, caption }: { payload: Table; caption?: string }) {
  const [sort, setSort] = useState<{ col: number; dir: 1 | -1 } | null>(null);
  const rows = useMemo(() => {
    if (!sort) return payload.rows;
    return [...payload.rows].sort((a, b) => sort.dir * cmp(a[sort.col] ?? null, b[sort.col] ?? null));
  }, [payload.rows, sort]);
  const shown = rows.slice(0, MAX_ROWS);

  return (
    <div className="table-wrap">
      <table className="grid">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr>
            {payload.columns.map((c, i) => (
              <th
                key={c.key}
                className={c.type === "number" ? "num" : undefined}
                aria-sort={sort?.col === i ? (sort.dir === 1 ? "ascending" : "descending") : "none"}
              >
                <button
                  className="th-sort"
                  onClick={() => setSort((s) => (s?.col === i ? { col: i, dir: s.dir === 1 ? -1 : 1 } : { col: i, dir: 1 }))}
                >
                  {c.unit ? unitLabel(c.label, c.unit) : c.label}
                  {sort?.col === i ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((row, r) => (
            <tr key={r}>
              {payload.columns.map((c, i) => (
                <td key={c.key} data-label={c.label} className={c.type === "number" ? "num" : undefined}>
                  {formatCell(row[i] ?? null, c.type, c.unit)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > MAX_ROWS && (
        <div className="notice">SHOWING {MAX_ROWS} OF {rows.length} ROWS · USE JSON EXPORT FOR ALL</div>
      )}
    </div>
  );
}
