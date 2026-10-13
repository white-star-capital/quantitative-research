// Poll timing (front_end_design.md §9.4 takes precedence over FRONTEND.md F-3):
// every 60 s and on tab focus; errors back off exponentially, capped at 60 s (F-4).

export const POLL_INTERVAL_MS = 60_000;
const BACKOFF_BASE_MS = 2_000;
const BACKOFF_CAP_MS = 60_000;

export function backoffMs(failures: number): number {
  if (failures <= 0) return POLL_INTERVAL_MS;
  return Math.min(BACKOFF_BASE_MS * 2 ** (failures - 1), BACKOFF_CAP_MS);
}

/** Retry-After as delta-seconds or HTTP-date; null if absent or unparseable. */
export function retryAfterMs(header: string | null, now: number = Date.now()): number | null {
  if (!header) return null;
  const trimmed = header.trim();
  if (/^\d+$/.test(trimmed)) return Number(trimmed) * 1000;
  const at = Date.parse(trimmed);
  return Number.isNaN(at) ? null : Math.max(0, at - now);
}
