import { useEffect, useRef, useState } from "react";
import { API_BASE_URL } from "../config.ts";
import { fetchJson, problemSlug } from "./client.ts";
import { POLL_INTERVAL_MS, backoffMs, retryAfterMs } from "./polling.ts";
import type { Checked } from "./validate.ts";

export interface Polled<T> {
  /** Last good, validated response. Kept through later errors (FRONTEND.md F-4). */
  data: T | null;
  loading: boolean;
  /** 4xx answer from the API (unknown slot, invalid scope, no output yet). */
  problem: { status: number; slug: string | null; detail: string | null } | null;
  /** Network, 5xx or non-JSON failure. Shown alongside last good data. */
  error: string | null;
  /** Response that failed validation (F-5), kept for the JSON download. */
  unsupported: { raw: unknown; reason: string } | null;
}

const INITIAL = { data: null, loading: true, problem: null, error: null, unsupported: null };

/**
 * GET `path` now, then every 60 s while the tab is visible, plus immediately on focus.
 * ETag/304 keeps unchanged polls cheap; errors back off up to 60 s; 429 honours Retry-After.
 */
export function usePolled<T>(path: string | null, check: (body: unknown) => Checked<T>): Polled<T> {
  const [state, setState] = useState<Polled<T>>(INITIAL);
  const checkRef = useRef(check);
  checkRef.current = check;

  useEffect(() => {
    setState(INITIAL);
    if (path === null) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let etag: string | null = null;
    let failures = 0;

    const schedule = (ms: number) => {
      clearTimeout(timer);
      if (!cancelled && document.visibilityState === "visible") timer = setTimeout(tick, ms);
    };

    const tick = async () => {
      const res = await fetchJson(API_BASE_URL, path, etag);
      if (cancelled) return;

      if (res.kind === "ok") {
        const checked = checkRef.current(res.body);
        failures = 0;
        if (checked.ok) {
          etag = res.etag;
          setState({ data: checked.value, loading: false, problem: null, error: null, unsupported: null });
        } else {
          setState((s) => ({ ...s, loading: false, unsupported: { raw: res.body, reason: checked.error } }));
        }
        schedule(POLL_INTERVAL_MS);
      } else if (res.kind === "not_modified") {
        failures = 0;
        setState((s) => ({ ...s, loading: false, error: null }));
        schedule(POLL_INTERVAL_MS);
      } else if (res.kind === "problem" && res.status >= 400 && res.status < 500 && res.status !== 429) {
        failures = 0;
        const problem = {
          status: res.status,
          slug: problemSlug(res.problem),
          detail: res.problem?.detail ?? null,
        };
        setState((s) => ({ ...s, loading: false, problem }));
        schedule(POLL_INTERVAL_MS);
      } else {
        failures += 1;
        const message =
          res.kind === "network_error"
            ? `NETWORK ERROR: ${res.message}`
            : res.kind === "missing"
              ? "API RETURNED NON-JSON"
              : `API ERROR ${res.status}`;
        setState((s) => ({ ...s, loading: false, error: message }));
        const wait = res.kind === "problem" ? retryAfterMs(res.retryAfter) : null;
        schedule(wait ?? backoffMs(failures));
      }
    };

    const onVisible = () => {
      if (document.visibilityState === "visible") void tick();
      else clearTimeout(timer);
    };

    void tick();
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", onVisible);
    return () => {
      cancelled = true;
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("focus", onVisible);
    };
  }, [path]);

  return state;
}
