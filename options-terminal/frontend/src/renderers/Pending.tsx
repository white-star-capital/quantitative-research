import type { PayloadKind } from "../api/types.ts";

/** Placeholder until the M3 renderer exists. Shows no numbers: nothing is drawn as zero. */
export function Pending({ kind }: { kind: PayloadKind }) {
  return (
    <div className="notice">
      RENDERER PENDING (M3) · {kind.toUpperCase()} · USE JSON EXPORT FOR THE RAW PAYLOAD
    </div>
  );
}
