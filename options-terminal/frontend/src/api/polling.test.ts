import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { POLL_INTERVAL_MS, backoffMs, retryAfterMs } from "./polling.ts";

describe("polling", () => {
  it("backs off exponentially and caps at 60s", () => {
    assert.equal(backoffMs(0), POLL_INTERVAL_MS);
    assert.equal(backoffMs(1), 2_000);
    assert.equal(backoffMs(3), 8_000);
    assert.equal(backoffMs(20), 60_000);
  });

  it("reads Retry-After seconds and dates", () => {
    assert.equal(retryAfterMs("30"), 30_000);
    const now = Date.parse("2026-10-13T17:00:00Z");
    assert.equal(retryAfterMs("Tue, 13 Oct 2026 17:00:10 GMT", now), 10_000);
    assert.equal(retryAfterMs(null), null);
    assert.equal(retryAfterMs("soon"), null);
  });
});
