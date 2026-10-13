# Deribit adapter

Public, keyless JSON-RPC over HTTPS (`https://www.deribit.com/api/v2/public/*`). Spec: `BS_PRICER_IMPLEMENTATION_GUIDE.md` §2.

| Underlying | Book (`currency`) | `base_currency` | Settlement | Spot index | Candles | DVOL |
|---|---|---|---|---|---|---|
| BTC | `BTC` (inverse) | BTC | BTC | `btc_usd` | `BTC-PERPETUAL` | `btcdvol_usdc` |
| ETH | `ETH` (inverse) | ETH | ETH | `eth_usd` | `ETH-PERPETUAL` | `ethdvol_usdc` |
| SOL | `USDC` (linear) | SOL | USDC | `sol_usdc` | `SOL_USDC-PERPETUAL` | — |
| HYPE | `USDC` (linear) | HYPE | USDC | `hype_usdc` | `HYPE_USDC-PERPETUAL` | — |

## Conventions

- Rows join book summary to instrument metadata on `instrument_name`. Names are never constructed (D-2), and `contract_size` comes from metadata (D-3).
- USD per 1 unit (D-5): inverse marks/bids/asks × the row's `underlying_price` (the forward). For linear USDC books, USDC is taken as USD (assumption).
- `mark_iv` arrives in percent and is stored as a decimal. A null or zero price/IV is *missing*, never zero.
- `snapshot_at` is Deribit's `usOut` server time (D-4).
- The USDC book summary is cached for 60 s, so SOL and HYPE share one fetch per tick. Instrument metadata is cached for 1 h.
- Any failure after retries → `venue_unavailable` (D-8). There is no fallback data (D-10).

## To confirm on first live access (guide §2, M1)

- [ ] Perpetual names for SOL/HYPE candles.
- [ ] `btcdvol_usdc` / `ethdvol_usdc` index names.
- [ ] Depth of the inverse BTC/ETH books vs linear `BTC_USDC`/`ETH_USDC`; record the choice here.
