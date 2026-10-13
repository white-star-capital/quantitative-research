# bs_pricer_black76

Fills slot `bs_pricer` (shadow-only until promoted; guide §9). Full methodology: [`BS_PRICER_IMPLEMENTATION_GUIDE.md`](../../../../BS_PRICER_IMPLEMENTATION_GUIDE.md) §3.

**Method.**
- Black-76 on each expiry's forward, r = `usd_rate` (default 0, so fair reproduces Deribit marks).
- Nodes are OTM listed options with a mark IV. Smiles are linear in total variance in log-moneyness; across tenors, linear in total variance at fixed k.
- Off-grid values are held flat and always flagged. Display grids never extrapolate.

**Inputs (Deribit):** joined chain, spot index, 31 days of hourly perp candles, DVOL (BTC/ETH).

**Output:** composite `context`, `term_structure`, `smile`, `surface`, `expiries`, `chain` (guide §4.3).

**Fails visibly, never silently:**
- venue down → `venue_unavailable`;
- input older than 120 s → `stale_input`;
- no expiry with enough nodes → `insufficient_market_data`;
- more than 20% of nodes missing the venue mark by ≥ 1% → `mark_reproduction` (a units bug);
- an expiry below `min_nodes_per_expiry` → dropped with a warning.

## Known limits (guide §11)

- SOL/HYPE books are thin; marks there are mostly Deribit's model. Node quality is reported everywhere.
- Single venue: an outage makes all four assets stale (the last good run is served, marked stale).
- Sticky-strike, piecewise-linear surface. SVI is a backlog shadow candidate.
- The IV solver's absolute price tolerance (1e-10) can't pin bid/ask IVs for options worth ~1e-12. That doesn't occur at real USD prices, but it does on synthetic test chains.
- The distribution overlay in evaluation is lognormal at ATM vol (`lognormal_atm`) and ignores skew.

## Tests

- `tests/test_model.py`: synthetic chains priced from a known smile (`tests/synthetic.py`, test-only). Covers sections, node selection, drops, failure paths, units-bug hard fail, determinism ×10, subprocess isolation and `evaluate`.
- `tests/test_golden.py`: golden runs on recorded Deribit snapshots per asset (`tests/fixtures/`). **Pending live access to Deribit.**
