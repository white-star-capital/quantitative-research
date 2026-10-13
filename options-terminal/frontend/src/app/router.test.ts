import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type { SlotSummary } from "../api/types.ts";
import { checkScope, defaultScope, formatHash, parseHash } from "./router.ts";

const slot: SlotSummary = {
  slot_id: "smile",
  title: "Smile",
  description: "",
  payload_kind: "curve",
  payload_version: "1",
  scope: { dimensions: ["underlying", "expiry"], allowed: { underlying: ["BTC", "ETH"] } },
  max_age_seconds: 900,
  production_impl: null,
  production_version: null,
};

describe("parseHash / formatHash", () => {
  it("parses the catalog", () => {
    assert.deepEqual(parseHash(""), { name: "catalog" });
    assert.deepEqual(parseHash("#/"), { name: "catalog" });
  });

  it("round-trips a slot route with scope", () => {
    const route = { name: "slot", slotId: "smile", scope: { underlying: "BTC", expiry: "2026-12-25" } } as const;
    const hash = formatHash(route);
    assert.equal(hash, "#/slots/smile?expiry=2026-12-25&underlying=BTC");
    assert.deepEqual(parseHash(hash), route);
  });

  it("encodes values", () => {
    const hash = formatHash({ name: "slot", slotId: "smile", scope: { underlying: "a&b=<c>" } });
    assert.deepEqual(parseHash(hash), { name: "slot", slotId: "smile", scope: { underlying: "a&b=<c>" } });
  });

  it("rejects malformed slot ids and keys", () => {
    for (const bad of ["#/slots/Vol-Surface", "#/slots/<script>", "#/slots/vol_surface/x", "#/other",
                       "#/slots/smile?Bad-Key=1", "#/slots/smile?underlying=BTC&underlying=ETH"]) {
      assert.equal(parseHash(bad).name, "not_found", bad);
    }
  });
});

describe("checkScope", () => {
  it("accepts a complete, allowed scope", () => {
    assert.equal(checkScope(slot, { underlying: "BTC", expiry: "2026-12-25" }).ok, true);
  });

  it("lists every problem", () => {
    const r = checkScope(slot, { underlying: "DOGE", strike: "1" });
    assert.equal(r.ok, false);
    if (!r.ok) {
      assert.deepEqual(r.problems, [
        "UNKNOWN PARAMETER STRIKE",
        "UNDERLYING MUST BE ONE OF BTC, ETH",
        "MISSING EXPIRY",
      ]);
    }
  });

  it("only defaults when every dimension has an allowlist", () => {
    assert.equal(defaultScope(slot), null);
    const one = { ...slot, scope: { dimensions: ["underlying"], allowed: { underlying: ["BTC"] } } };
    assert.deepEqual(defaultScope(one), { underlying: "BTC" });
  });
});
