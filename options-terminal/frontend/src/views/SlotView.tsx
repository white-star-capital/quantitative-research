// Slot view (FRONTEND.md §4.2): scope selectors, header with freshness, warnings,
// payload body, audit footer.
import { useEffect, useMemo } from "react";
import { encodeScope } from "../api/client.ts";
import type { LatestResponse, Scope, SlotSummary } from "../api/types.ts";
import { usePolled, type Polled } from "../api/usePolled.ts";
import { checkLatest } from "../api/validate.ts";
import { checkScope, defaultScope, formatHash } from "../app/router.ts";
import { formatAge, formatUtc } from "../format/units.ts";
import { rendererFor } from "../renderers/index.tsx";
import { PanelBoundary } from "./PanelBoundary.tsx";
import {
  ErrorIndicator,
  FreshnessBadge,
  InvalidScope,
  Loading,
  NoOutputYet,
  UnknownSlot,
  Unsupported,
} from "./states.tsx";

interface Props {
  slotId: string;
  scope: Scope;
  slots: Polled<SlotSummary[]>;
  onAsOf: (asOf: string | null) => void;
}

export function SlotView({ slotId, scope, slots, onAsOf }: Props) {
  const slot = slots.data?.find((s) => s.slot_id === slotId) ?? null;
  const scopeCheck = useMemo(() => (slot ? checkScope(slot, scope) : null), [slot, scope]);
  const path =
    slot && scopeCheck?.ok ? `/v1/slots/${encodeURIComponent(slotId)}/latest${encodeScope(scope)}` : null;
  const latest = usePolled(path, checkLatest);
  const asOf = latest.data?.data.as_of ?? null;

  useEffect(() => {
    onAsOf(asOf);
    return () => onAsOf(null);
  }, [asOf, onAsOf]);

  if (!slot) {
    if (slots.loading) return <Loading what="CATALOG" />;
    if (slots.data === null) {
      return <ErrorIndicator error={slots.error ?? "CATALOG UNAVAILABLE"} lastGoodAsOf={null} />;
    }
    return <UnknownSlot slotId={slotId} />;
  }

  return (
    <section>
      <div className="section-title">
        <span>{slot.title}</span>
        <span>{slot.slot_id.toUpperCase()}</span>
      </div>
      <ScopeBar slot={slot} scope={scope} />
      {scopeCheck && !scopeCheck.ok ? (
        <>
          <InvalidScope problems={scopeCheck.problems} />
          {defaultScope(slot) && (
            <div className="notice">
              <a href={formatHash({ name: "slot", slotId, scope: defaultScope(slot) ?? {} })}>
                RESET TO DEFAULT SCOPE
              </a>
            </div>
          )}
        </>
      ) : (
        <LatestPanel slotId={slotId} latest={latest} />
      )}
    </section>
  );
}

function ScopeBar({ slot, scope }: { slot: SlotSummary; scope: Scope }) {
  const go = (key: string, value: string) => {
    window.location.hash = formatHash({ name: "slot", slotId: slot.slot_id, scope: { ...scope, [key]: value } });
  };
  return (
    <div className="scope-bar">
      {slot.scope.dimensions.map((d) => {
        const allowed = slot.scope.allowed?.[d];
        const id = `scope-${d}`;
        return (
          <span key={d}>
            <label className="label" htmlFor={id}>
              {d}
            </label>{" "}
            {allowed ? (
              <select id={id} value={scope[d] ?? ""} onChange={(e) => go(d, e.target.value)}>
                {!scope[d] && <option value="">—</option>}
                {allowed.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            ) : (
              <input
                id={id}
                defaultValue={scope[d] ?? ""}
                onKeyDown={(e) => {
                  if (e.key === "Enter") go(d, e.currentTarget.value.trim());
                }}
              />
            )}
          </span>
        );
      })}
    </div>
  );
}

function LatestPanel({ slotId, latest }: { slotId: string; latest: Polled<LatestResponse> }) {
  const { data, problem, error, unsupported, loading } = latest;

  if (problem?.slug === "unknown-slot") return <UnknownSlot slotId={slotId} />;
  if (problem?.status === 400) return <InvalidScope problems={[problem.detail ?? "REJECTED BY API"]} />;
  if (problem?.slug === "no-output-yet" && !data) return <NoOutputYet />;
  if (unsupported && !data) return <Unsupported raw={unsupported.raw} reason={unsupported.reason} />;
  if (loading) return <Loading what="LATEST OUTPUT" />;
  if (!data) {
    return <ErrorIndicator error={error ?? problem?.detail ?? "NO DATA"} lastGoodAsOf={null} />;
  }

  const env = data.data;
  const meta = data.meta;
  const Renderer = env.status === "success" ? rendererFor(env) : null;

  return (
    <div className="panel">
      <div className="panel-title">
        <span className="title">
          1) {env.implementation_id} {env.implementation_version}
          {meta.role === "shadow" ? " · SHADOW" : ""}
        </span>
        <FreshnessBadge stale={meta.stale} />
      </div>
      <div className="panel-meta">
        AS OF{" "}
        <span className="num" title={new Date(env.as_of).toLocaleString()}>
          {formatUtc(env.as_of)}
        </span>{" "}
        · {formatAge(meta.age_seconds)}
      </div>
      {(error || unsupported) && (
        <ErrorIndicator error={error ?? `UNSUPPORTED UPDATE: ${unsupported?.reason}`} lastGoodAsOf={env.as_of} />
      )}
      {env.warnings.length > 0 && (
        <div className="notice">
          <span className="status mid">WARN</span>{" "}
          {env.warnings.map((w) => `${w.code}: ${w.message}`).join(" · ")}
        </div>
      )}
      <div className="panel-body">
        <PanelBoundary>
          {env.status === "failed" ? (
            <div className="notice bad">
              RUN FAILED · {env.error.code} · {env.error.message}
            </div>
          ) : Renderer ? (
            <Renderer payload={env.payload} />
          ) : (
            <Unsupported raw={env} reason={`${env.payload_kind} v${env.payload_version}`} />
          )}
        </PanelBoundary>
      </div>
      <dl className="three dim">
        <div>
          <dt className="label">Run</dt>
          <dd className="num">{env.run_id}</dd>
        </div>
        <div>
          <dt className="label">Code</dt>
          <dd className="num">{env.code.git_sha}</dd>
        </div>
        <div>
          <dt className="label">Inputs</dt>
          <dd>
            {env.inputs.sources.map((s) => `${s.name} @ ${formatUtc(s.snapshot_at)}`).join(" · ") || "—"}
          </dd>
        </div>
      </dl>
    </div>
  );
}
