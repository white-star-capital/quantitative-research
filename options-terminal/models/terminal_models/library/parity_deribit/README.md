# parity_deribit

Fills slot `parity_monitor`. Full methodology: [`PARITY_MONITOR_IMPLEMENTATION_GUIDE.md`](../../../../PARITY_MONITOR_IMPLEMENTATION_GUIDE.md) §2–3 and its amendments.

**Not registered yet.** `params.fees` is null until the fee table is filled from Deribit's current schedule and reviewed by trading. `validate-registry` refuses any registered manifest with a null fee, and the runner validates the whole registry each pass, so registering it early would also stop `bs_pricer`. Once the fees are filled, add it to `models/registry.yaml` as shadow (`production: null`).

**Method.**
- Parity runs on executable bid/ask, never marks. Inverse books (BTC/ETH) use the coin formulas, hedged with inverse futures at K USD notional per coin. Linear books (SOL/HYPE) use the USD formulas.
- Hedge per expiry: the same-expiry dated future (`locked_future`), else the perp (`perp_funding`, unlocked; shown with breakeven funding).
- Implied forward per expiry: the median of per-strike forwards within `atm_band`, weighted by 1/band width. Expiries with fewer than `min_strikes_for_forward` strikes are `insufficient_strikes`.
- Fees: taker on all legs, plus delivery when held to expiry (guide A-4). Net edge, USD value, bp of the K notional, and simple annualized come from `core/quant/parity.py:net_metrics`.
- The top `max_confirmations` candidates are confirmed on their legs' order books (follow-up inputs, M-5a) and recomputed. Size, hedge contracts, rounding residual and leg time spread come from those books.

**Inputs (Deribit):** option chain (native bid/ask), spot index, futures book (dated + perp), perp funding (ticker), plus order books for the screened candidates.

**Output:** composite `context`, `carry_curve`, `expiries`, `bands`, `opportunities`, `fees` (guide §4.2, A-2).

**Fails visibly, never silently:**
- null fees → `fees_unavailable` (and validate-registry refuses the manifest);
- no two-sided call/put pair → `insufficient_market_data`;
- no expiry with an implied forward → `insufficient_market_data`;
- crossed parity band → `parity_band_ordering` (bad data);
- option vs futures books more than `max_snapshot_skew_s` apart → `snapshot_skew` warning, and no unconfirmed row is ever executable;
- a missing or empty order book → `confirmation_unavailable` warning, row stays unconfirmed.

## To confirm on first live access (guide M1, §2.2)

- [ ] `funding_8h` semantics: annualized here as `funding_8h × 3 × 365`.
- [ ] Order-book `amount` units: inverse futures in USD and options in contracts × `contract_size` (assumed); linear futures in coin.
- [ ] Same-expiry options and futures settle to the same delivery price. If not, `locked_future` rows are downgraded to `basis_risk`.

## Tests

`tests/` run the model on a TEST-ONLY synthetic chain (`tests/synthetic.py`) with a TEST-ONLY fee table. Golden tests on recorded Deribit snapshots run once a recording is committed (`tests/test_golden.py`).
