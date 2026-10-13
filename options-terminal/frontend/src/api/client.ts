// Native fetch wrapper (building_secure_front_ends.md §2: no HTTP client libraries).
import { asProblem } from "./validate.ts";
import type { Problem } from "./types.ts";

export type FetchResult =
  | { kind: "ok"; body: unknown; etag: string | null }
  | { kind: "not_modified" }
  | { kind: "problem"; status: number; problem: Problem | null; retryAfter: string | null }
  | { kind: "missing" } // non-JSON response, e.g. a proxy serving index.html (design §9.4)
  | { kind: "network_error"; message: string };

export async function fetchJson(
  baseUrl: string,
  path: string,
  etag: string | null = null,
  fetchImpl: typeof fetch = fetch,
): Promise<FetchResult> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (etag) headers["If-None-Match"] = etag;

  let res: Response;
  try {
    res = await fetchImpl(`${baseUrl}${path}`, { cache: "no-store", headers });
  } catch (err) {
    return { kind: "network_error", message: err instanceof Error ? err.message : String(err) };
  }
  if (res.status === 304) return { kind: "not_modified" };

  const type = res.headers.get("content-type") ?? "";
  const isJson = type.includes("application/json") || type.includes("application/problem+json");
  if (!isJson) return { kind: "missing" };

  let body: unknown;
  try {
    body = await res.json();
  } catch {
    return { kind: "missing" };
  }
  if (!res.ok) return problem(res, asProblem(body));
  return { kind: "ok", body, etag: res.headers.get("etag") };
}

function problem(res: Response, p: Problem | null): FetchResult {
  return { kind: "problem", status: res.status, problem: p, retryAfter: res.headers.get("retry-after") };
}

/** Problem type slug, e.g. ".../problems/no-output-yet" -> "no-output-yet" (API.md A-6). */
export function problemSlug(p: Problem | null): string | null {
  if (!p) return null;
  return p.type.split("/").pop() || null;
}

export function encodeScope(scope: Record<string, string>): string {
  const params = new URLSearchParams();
  for (const k of Object.keys(scope).sort()) params.set(k, scope[k] ?? "");
  const q = params.toString();
  return q ? `?${q}` : "";
}
