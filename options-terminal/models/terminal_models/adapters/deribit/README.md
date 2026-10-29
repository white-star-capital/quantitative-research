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

## vrp_signal inputs (RV_FORECAST_IMPLEMENTATION_GUIDE.md VRP-A-8, §12)

- `rv_daily`: `get_tradingview_chart_data` on `params.instrument`, or the discovered perpetual, at `params.resolution` (minutes). Paged in whole UTC days of at most `CHART_ROWS_PER_CALL` bars, reduced to completed days by `core/quant/rv.daily_rv`; days below `min_coverage` are null. `CHART_ROWS_PER_CALL` is an upper bound, not the venue's cap: a page that comes back with some bars but short days is halved and asked again (at most `MAX_SPLIT_DEPTH` deep; it stops at a response with fewer bars than one covered day, or a half that adds no bar to a short day and returns fewer bars than its parent), so a lower cap costs calls, never coverage. Shared with Hyperliquid in `adapters/rv_daily.py`, including the completed-day cache under `SNAPSHOT_DIR/rv_daily/` (runner only; recordings and replay don't use it). `snapshot_at` is the last page's `usOut`, or `get_time` when every day was cached.
- `dvol_history`: `get_volatility_index_data` at resolution `1D`, paged back via `continuation`. Each completed day's close, in vol points ÷ 100 (decimal). BTC/ETH only; other assets are `bad_input_spec`. A candle opening at `ts` is keyed to day `ceil_day(ts)`: its own day when candles open at 00:00 UTC, otherwise the day its close falls in, so a day's value is the latest DVOL known at that day's close and never a candle still open. `close_offset_h` in the data is the candles' UTC close hour (0 when aligned); candles at mixed hours fail.
- BTC/ETH `rv_daily` stays on hourly bars in v1: it matches bs_pricer's RV basis, HAR on log RV gains little from 5-minute sampling at 7–60-day horizons, and switching resolution mid-series shifts the level. Moving to 5-minute bars is a later, versioned change, gated on `make probe` (5-minute history depth) and a `validate_rv` comparison.

## To confirm on first live access (guide §2, M1)

- [x] `btc_usd` / `eth_usd` / `sol_usdc` / `hype_usdc` spot index names (confirmed live 2026-10-13).
- [x] `btcdvol_usdc` / `ethdvol_usdc` index names (confirmed live 2026-10-13).
- [ ] Whether Deribit lists a SOL/HYPE perpetual. Until one is listed, SOL/HYPE runs fail with `no_perpetual`; the first recording will show it.
- [ ] parity: `funding_8h` semantics; order-book `amount` units (inverse futures USD, options in contracts, linear futures coin); same-expiry option and future delivery prices match (else `locked_future` → `basis_risk`).
- [ ] vrp_signal: the `get_tradingview_chart_data` row cap (`CHART_ROWS_PER_CALL` 5,000 is an upper bound; a lower cap is split around) and 5-min history depth; the `1D` DVOL candles' open hour (any hour works; non-zero shows as `close_offset_h`) and how `continuation` pages. `make probe` (tools/probe_venues.py) answers all of these and writes `library/rv_har/validation/venue_probe.json`.
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
