// One renderer per payload kind (FRONTEND.md §4.3). Renderers are pure functions of the
// payload (F-18) and never know which slot they draw. Unknown kind/version -> Unsupported.
import type { ComponentType } from "react";
import type { PayloadKind, SuccessEnvelope } from "../api/types.ts";
import { Pending } from "./Pending.tsx";

export interface RendererProps {
  payload: Record<string, unknown>;
}

interface Renderer {
  versions: readonly string[];
  component: ComponentType<RendererProps>;
}

const pending = (kind: PayloadKind): Renderer => ({
  versions: ["1"],
  component: () => <Pending kind={kind} />,
});

// M3 replaces each entry with the real renderer (SVG per front_end_design.md §5).
const RENDERERS: Record<PayloadKind, Renderer> = {
  scalar_set: pending("scalar_set"),
  curve: pending("curve"),
  surface: pending("surface"),
  timeseries: pending("timeseries"),
  table: pending("table"),
  composite: pending("composite"),
};

export function rendererFor(envelope: SuccessEnvelope): ComponentType<RendererProps> | null {
  const r = RENDERERS[envelope.payload_kind];
  return r && r.versions.includes(envelope.payload_version) ? r.component : null;
}
