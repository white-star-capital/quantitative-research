// Bespoke slot views (FRONTEND §1): an additive layer keyed by the slot's `display.default_view`.
// The generic renderers must still render every slot; users can always switch to them.
import type { ComponentType } from "react";
import type { LatestResponse, Scope, SlotSummary } from "../api/types.ts";
import type { Polled } from "../api/usePolled.ts";
import type { SlotDetail } from "../api/validate.ts";

export interface BespokeProps {
  slot: SlotSummary;
  detail: SlotDetail;
  /** Empty when the view is opened without a scope (e.g. a cross-asset landing view). */
  scope: Scope;
  /** run selection params: {} for production, or role/implementation in a shadow period. */
  selection: Record<string, string>;
  latest: Polled<LatestResponse>;
}

const VIEWS = new Map<string, ComponentType<BespokeProps>>();

export function registerView(id: string, view: ComponentType<BespokeProps>): void {
  VIEWS.set(id, view);
}

export function bespokeFor(slot: SlotSummary): ComponentType<BespokeProps> | null {
  const id = slot.display?.default_view;
  return (id && VIEWS.get(id)) || null;
}

export function hasBespoke(slot: SlotSummary): boolean {
  return bespokeFor(slot) !== null;
}
