// One renderer per payload kind (FRONTEND.md §4.3). Renderers are pure functions of the payload
// (F-18) and never know which slot they draw. Unknown kind/version -> Unsupported.
import type { ComponentType } from "react";
import type { Composite, Curve, ScalarSet, SurfacePayload, Table, Timeseries } from "../api/payloads.ts";
import type { PayloadKind, SuccessEnvelope } from "../api/types.ts";
import { PanelBoundary } from "../views/PanelBoundary.tsx";
import { CurveChart, SurfaceHeatmap, TimeseriesChart } from "./Charts.tsx";
import { DataTable } from "./DataTable.tsx";
import { ScalarTiles } from "./ScalarTiles.tsx";

export interface RendererProps {
  payload: unknown;
  title: string;
}

interface Renderer {
  versions: readonly string[];
  component: ComponentType<RendererProps>;
}

const R = (component: Renderer["component"]): Renderer => ({ versions: ["1"], component });

function CompositeSections({ payload }: { payload: unknown }) {
  return (
    <>
      {(payload as Composite).sections.map((s) => {
        const r = RENDERERS[s.payload_kind];
        const Section = r && r.versions.includes(s.payload_version) ? r.component : null;
        return (
          <section key={s.key} className="subsection">
            <div className="title sub-title">{s.label}</div>
            <PanelBoundary>
              {Section ? <Section payload={s.payload} title={s.label} /> : <div className="notice bad">UNSUPPORTED SECTION {s.payload_kind}</div>}
            </PanelBoundary>
          </section>
        );
      })}
    </>
  );
}

const RENDERERS: Record<PayloadKind, Renderer> = {
  scalar_set: R(({ payload }) => <ScalarTiles payload={payload as ScalarSet} />),
  curve: R(({ payload, title }) => <CurveChart payload={payload as Curve} title={title} />),
  surface: R(({ payload, title }) => <SurfaceHeatmap payload={payload as SurfacePayload} title={title} />),
  timeseries: R(({ payload, title }) => <TimeseriesChart payload={payload as Timeseries} title={title} />),
  table: R(({ payload, title }) => <DataTable payload={payload as Table} caption={title} />),
  composite: R(({ payload }) => <CompositeSections payload={payload} />),
};

export function rendererFor(envelope: SuccessEnvelope): ComponentType<RendererProps> | null {
  const r = RENDERERS[envelope.payload_kind];
  // Stable component identity, so polling updates don't remount (sort/hover state survives).
  return r && r.versions.includes(envelope.payload_version) ? r.component : null;
}
