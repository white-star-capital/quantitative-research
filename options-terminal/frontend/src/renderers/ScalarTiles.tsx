import type { ScalarSet } from "../api/payloads.ts";
import { MISSING, formatValue } from "../format/units.ts";

const FLAG_BADGE: Record<string, string> = {
  tenor_extrapolated_short: "OFF-GRID",
  tenor_extrapolated_long: "OFF-GRID",
  strike_extrapolated: "OFF-GRID",
};

/** scalar_set: metric tiles. Flags show as a word badge with the code on hover. */
export function ScalarTiles({ payload }: { payload: ScalarSet }) {
  return (
    <div className="tiles">
      {payload.items.map((it) => (
        <div key={it.key} className="tile">
          <div className="label">{it.label}</div>
          <div className="tile-value num">
            {typeof it.value === "boolean" ? (it.value ? "YES" : "NO") : it.value === null ? MISSING : formatValue(it.value, it.unit)}
          </div>
          {(it.flags ?? []).map((f) => (
            <span key={f} className="badge" title={f.replaceAll("_", " ")}>{FLAG_BADGE[f] ?? f.toUpperCase()}</span>
          ))}
        </div>
      ))}
    </div>
  );
}
