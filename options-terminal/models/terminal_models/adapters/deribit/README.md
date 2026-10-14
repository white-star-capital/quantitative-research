# Deribit adapter

Public, keyless JSON-RPC over HTTPS (`https://www.deribit.com/api/v2/public/*`). Spec: `BS_PRICER_IMPLEMENTATION_GUIDE.md` §2.

| Underlying | Book (`currency`) | `base_currency` | Settlement | Spot index | Candles (perp) | DVOL |
|---|---|---|---|---|---|---|
| BTC | `BTC` (inverse) | BTC | BTC | `btc_usd` | discovered (`BTC-PERPETUAL`) | `btcdvol_usdc` |
| ETH | `ETH` (inverse) | ETH | ETH | `eth_usd` | discovered (`ETH-PERPETUAL`) | `ethdvol_usdc` |
| SOL | `USDC` (linear) | SOL | USDC | `sol_usdc` | discovered (none in the shapes seen so far) | — |
| HYPE | `USDC` (linear) | HYPE | USDC | `hype_usdc` | discovered (none in the shapes seen so far) | — |

## Conventions

- Rows join book summary to instrument metadata on `instrument_name`. Names are never constructed (D-2), and `contract_size` comes from metadata (D-3).
- USD per 1 unit (D-5): inverse marks/bids/asks × the row's `underlying_price` (the forward). For linear USDC books, USDC is taken as USD (assumption).
- `mark_iv` arrives in percent and is stored as a decimal. A null or zero price/IV is *missing*, never zero.
- Inverse vs linear comes from metadata `instrument_type` (`reversed` / `linear`); if that's absent, coin settlement means inverse.
- The candle perp is discovered from `get_instruments kind=future` metadata: `base_currency` = asset, `settlement_period` = `perpetual`, active, preferring settlement in the book's currency. If none is listed, the run fails with `no_perpetual`. RV never comes from a guessed instrument name (D-2).
- The book summary's `bid_iv`/`ask_iv` are recorded raw but not used. The model computes bid/ask IV from prices (guide §3.1).
- `snapshot_at` is Deribit's `usOut` server time (D-4).
- The USDC book summary is cached for 60 s, so SOL and HYPE share one fetch per tick. Instrument metadata is cached for 1 h.
- Any failure after retries → `venue_unavailable` (D-8). There is no fallback data (D-10).

## parity_monitor inputs (PARITY_MONITOR_IMPLEMENTATION_GUIDE.md §2)

- `futures_book`: `get_book_summary_by_currency kind=future`, joined to `get_instruments kind=future` metadata on `instrument_name` (D-15). `contract_size` comes from metadata: USD per contract for inverse, coin for linear.
- `funding`: `ticker` on the discovered perpetual. Annualized downstream as `funding_8h × 3 × 365`; the field semantics are confirmed in M1.
- `order_book`: `get_order_book depth=5`. `snapshot_at` is the book's own `timestamp`, so leg time spread is measurable (D-13). Requested as optional follow-up inputs (MODELS M-5a) and never cached.
- Option rows also carry native `bid_native`/`ask_native` (coin or USDC), `expiration_ms` and `underlying_index`, because parity runs on native prices.
- Tick cache (`snapshot_cache.py`, D-11): one cache per `run-due` pass. Books and tickers are reused for at most 20 s; metadata for 1 h.

## To confirm on first live access (guide §2, M1)

- [x] `btc_usd` / `eth_usd` / `sol_usdc` / `hype_usdc` spot index names (confirmed live 2026-10-13).
- [x] `btcdvol_usdc` / `ethdvol_usdc` index names (confirmed live 2026-10-13).
- [ ] Whether Deribit lists a SOL/HYPE perpetual. Until one is listed, SOL/HYPE runs fail with `no_perpetual`; the first recording will show it.
- [ ] parity: `funding_8h` semantics; order-book `amount` units (inverse futures USD, options in contracts, linear futures coin); same-expiry option and future delivery prices match (else `locked_future` → `basis_risk`).
- [ ] Depth of the inverse BTC/ETH books vs linear `BTC_USDC`/`ETH_USDC`; record the choice here (both books' summaries are in the first recording).

## Recording snapshots (fixtures, golden tests, parity vectors)

Run this anywhere `www.deribit.com` is reachable:

```sh
cd options-terminal/models && uv sync
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
uv run python -m terminal_models.tools.record_deribit \
  --out terminal_models/library/bs_pricer_black76/tests/fixtures/deribit_$STAMP
```

The recorder fetches exactly what `bs_pricer_black76` and `parity_deribit` runs fetch, through this adapter and the runner's fetch path, plus BTC/ETH DVOL history. parity's follow-up order books are recorded only once its fee table is filled, because it screens on net edge; the manifest's `notes` say when they were skipped. It saves each response body byte-for-byte (gzip, `calls/NNN_<method>.body.gz`), with the request URL/params and the local receive time (`.meta.json`). `manifest.json` lists every call with its sha256. If a required call fails, no manifest is written and `FAILED.txt` explains why. `--allow-partial` keeps the assets that succeeded. Replay refuses a recording whose bodies don't match the manifest.

Then, from `options-terminal/models`:

```sh
uv run python -m terminal_models.tools.build_fixtures --recording <recording dir>
uv run python -m terminal_models.tools.gen_eval_vectors \
  --run ../contracts/fixtures/valid/composite/bs_pricer_btc.json \
  --run ../contracts/fixtures/valid/composite/bs_pricer_hype.json
```

Commit the recording, the fixtures and the vectors together. `build_fixtures` reports each asset's status, its node-quality mix and the guide §4.4 HYPE coverage. A run that fails on real data is a finding to report, not something to tune away.

The whole path (record → replay → fixtures → vectors → `runner replay` → Rust parity) runs offline in `models/tests/test_replay_pipeline.py`. It uses Deribit-shaped bodies built from the test-only synthetic chain, so a plumbing bug shows up there and not as a data finding on the first real recording. CI (`.github/workflows/options-terminal.yml`) runs it along with the golden tests.
