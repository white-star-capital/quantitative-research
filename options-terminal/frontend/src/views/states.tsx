// Slot view states (FRONTEND.md §5, front_end_design.md §6). Status is never colour alone.
import { useEffect, useMemo } from "react";
import { formatUtc } from "../format/units.ts";

export function Loading({ what }: { what: string }) {
  return <div className="notice">LOADING {what}…</div>;
}

export function FreshnessBadge({ stale }: { stale: boolean }) {
  return stale ? <span className="status bad">STALE</span> : <span className="status ok">FRESH</span>;
}

export function NoOutputYet() {
  return (
    <div className="notice">
      NO OUTPUT YET · THIS SLOT HAS NOT PRODUCED DATA FOR THIS SCOPE
    </div>
  );
}

export function SlotRetired({ slotId }: { slotId: string }) {
  return (
    <div className="notice">
      SLOT {slotId.toUpperCase()} IS RETIRED · RUN HISTORY STAYS AVAILABLE VIA THE API ·{" "}
      <a href="#/">BACK TO CATALOG</a>
    </div>
  );
}

export function UnknownSlot({ slotId }: { slotId: string }) {
  return (
    <div className="notice bad">
      UNKNOWN SLOT {slotId.toUpperCase()} · <a href="#/">BACK TO CATALOG</a>
    </div>
  );
}

export function InvalidScope({ problems }: { problems: string[] }) {
  return (
    <div className="notice bad">
      INVALID SCOPE · {problems.join(" · ")}
    </div>
  );
}

/** Non-blocking error shown with the last good data (F-4, design §6). */
export function ErrorIndicator({ error, lastGoodAsOf }: { error: string; lastGoodAsOf: string | null }) {
  return (
    <div className="notice bad">
      {lastGoodAsOf ? `SHOWING LAST GOOD FROM ${formatUtc(lastGoodAsOf)} · ` : ""}
      {error}
    </div>
  );
}

export function Unsupported({ raw, reason }: { raw: unknown; reason: string }) {
  const href = useMemo(
    () => URL.createObjectURL(new Blob([JSON.stringify(raw, null, 2)], { type: "application/json" })),
    [raw],
  );
  useEffect(() => () => URL.revokeObjectURL(href), [href]);
  return (
    <div className="notice bad">
      THIS OUTPUT CAN&apos;T BE DISPLAYED IN THIS VERSION OF THE TERMINAL · {reason} ·{" "}
      <a href={href} download="output.json">
        DOWNLOAD JSON
      </a>
    </div>
  );
}
