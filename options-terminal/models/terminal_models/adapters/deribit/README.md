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

## To confirm on first live access (guide §2, M1)

- [x] `btc_usd` / `eth_usd` / `sol_usdc` / `hype_usdc` spot index names (confirmed live 2026-10-13).
- [x] `btcdvol_usdc` / `ethdvol_usdc` index names (confirmed live 2026-10-13).
- [ ] Whether Deribit lists a SOL/HYPE perpetual. Until one is listed, SOL/HYPE runs fail with `no_perpetual`; the first recording will show it.
- [ ] Depth of the inverse BTC/ETH books vs linear `BTC_USDC`/`ETH_USDC`; record the choice here (both books' summaries are in the first recording).

## Recording snapshots (fixtures, golden tests, parity vectors)

Run this anywhere `www.deribit.com` is reachable:

```sh
cd options-terminal/models && uv sync
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
uv run python -m terminal_models.tools.record_deribit \
  --out terminal_models/library/bs_pricer_black76/tests/fixtures/deribit_$STAMP
```

The recorder fetches exactly what a `bs_pricer_black76` run fetches, through this adapter, plus BTC/ETH DVOL history. It saves each response body byte-for-byte (gzip, `calls/NNN_<method>.body.gz`), with the request URL/params and the local receive time (`.meta.json`). `manifest.json` lists every call with its sha256. If a required call fails, no manifest is written and `FAILED.txt` explains why. `--allow-partial` keeps the assets that succeeded. Replay refuses a recording whose bodies don't match the manifest.

Then, from `options-terminal/models`:

```sh
uv run python -m terminal_models.tools.build_fixtures --recording <recording dir>
uv run python -m terminal_models.tools.gen_eval_vectors \
  --run ../contracts/fixtures/valid/composite/bs_pricer_btc.json \
  --run ../contracts/fixtures/valid/composite/bs_pricer_hype.json
```

Commit the recording, the fixtures and the vectors together. `build_fixtures` reports each asset's status, its node-quality mix and the guide §4.4 HYPE coverage. A run that fails on real data is a finding to report, not something to tune away.
