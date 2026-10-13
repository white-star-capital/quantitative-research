// Hash routing, no router library (front_end_design.md §9.5). Deep links are validated
// before use (FRONTEND.md F-24): `#/slots/<slot_id>?<dimension>=<value>&…`.
import type { Scope, SlotSummary } from "../api/types.ts";

export type Route =
  | { name: "catalog" }
  | { name: "slot"; slotId: string; scope: Scope }
  | { name: "not_found"; hash: string };

const IDENTIFIER = /^[a-z][a-z0-9_]{2,63}$/;

export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#/, "");
  if (raw === "" || raw === "/") return { name: "catalog" };

  const [path = "", query = ""] = raw.split("?", 2);
  const parts = path.split("/").filter(Boolean);
  if (parts.length === 2 && parts[0] === "slots" && IDENTIFIER.test(parts[1] ?? "")) {
    const scope: Scope = {};
    for (const [k, v] of new URLSearchParams(query)) {
      if (!IDENTIFIER.test(k) || k in scope) return { name: "not_found", hash };
      scope[k] = v;
    }
    return { name: "slot", slotId: parts[1] as string, scope };
  }
  return { name: "not_found", hash };
}

export function formatHash(route: Route): string {
  if (route.name !== "slot") return "#/";
  const params = new URLSearchParams();
  for (const k of Object.keys(route.scope).sort()) params.set(k, route.scope[k] ?? "");
  const q = params.toString();
  return `#/slots/${route.slotId}${q ? `?${q}` : ""}`;
}

export type ScopeCheck = { ok: true; scope: Scope } | { ok: false; problems: string[] };

/** Check a deep-linked scope against the slot's declared dimensions and allowlist. */
export function checkScope(slot: SlotSummary, scope: Scope): ScopeCheck {
  const problems: string[] = [];
  const dims = slot.scope.dimensions;
  for (const k of Object.keys(scope)) {
    if (!dims.includes(k)) problems.push(`UNKNOWN PARAMETER ${k.toUpperCase()}`);
  }
  for (const d of dims) {
    const value = scope[d];
    const allowed = slot.scope.allowed?.[d];
    if (value === undefined || value === "") problems.push(`MISSING ${d.toUpperCase()}`);
    else if (allowed && !allowed.includes(value)) {
      problems.push(`${d.toUpperCase()} MUST BE ONE OF ${allowed.join(", ")}`);
    }
  }
  return problems.length ? { ok: false, problems } : { ok: true, scope };
}

/** First allowed value per dimension; null when a dimension has no allowlist to pick from. */
export function defaultScope(slot: SlotSummary): Scope | null {
  const scope: Scope = {};
  for (const d of slot.scope.dimensions) {
    const first = slot.scope.allowed?.[d]?.[0];
    if (first === undefined) return null;
    scope[d] = first;
  }
  return scope;
}
