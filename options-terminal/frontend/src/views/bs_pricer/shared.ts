import type { Composite, ScalarItem, ScalarSet } from "../../api/payloads.ts";
import { scalarMap, section } from "../../api/payloads.ts";
import type { LatestResponse } from "../../api/types.ts";
import { formatHash } from "../../app/router.ts";
import { query } from "./logic.ts";

export function storage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function latestPath(slotId: string, underlying: string, selection: Record<string, string>): string {
  return `/v1/slots/${encodeURIComponent(slotId)}/latest${query({ underlying, ...selection })}`;
}

export function payloadOf(latest: LatestResponse | null): Composite | null {
  const env = latest?.data;
  return env && env.status === "success" ? (env.payload as unknown as Composite) : null;
}

export function contextOf(payload: Composite | null): Record<string, ScalarItem> {
  const set = payload ? section<ScalarSet>(payload, "context") : null;
  return set ? scalarMap(set) : {};
}

export function num(item: ScalarItem | undefined): number | null {
  return typeof item?.value === "number" ? item.value : null;
}

export function assetHref(slotId: string, underlying: string | null): string {
  return formatHash({ name: "slot", slotId, scope: underlying ? { underlying } : {} });
}
