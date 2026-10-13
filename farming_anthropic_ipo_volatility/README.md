# Farming Anthropic IPO Volatility

Python implementation of the two-phase Anthropic IPO book across Hyperliquid `io:ANTH` and Strike cash-settled ANTH options.

**Core identities:** Black–Scholes (r = 0, 365-day year) for ATM marks; ATM + 25Δ risk reversal + 25Δ butterfly pin each smile; calendar interpolation is linear (or PCHIP) in total variance `w(T) = σ²T`.

## Architecture

```
selling_the_wait_pack_v2/
├── app.py                          # Streamlit interactive dashboard
├── requirements.txt
├── .streamlit/config.toml          # Paper theme
├── src/
│   ├── anth_ipo_book.py            # Snapshot constants, BS, event/fwd vol, phases, capital, returns
│   ├── anth_vol_surface.py         # Wing reconstruction, k-quadratic smile, interpolate_iv, PCHIP, BL density
│   ├── dashboard_charts.py         # Live Plotly figures for the dashboard
│   ├── render_smiles.py            # PNG: smiles, solid only inside 25Δ
│   ├── render_vol_surface_3d.py    # PNG: mesh interpolate_iv (K/S × time) + 2D heatmap
│   └── render_vol_surface_3d_smooth.py  # PNG: PCHIP sheet in call-delta × time
├── tests/
│   ├── test_anth_ipo_book.py       # 20 pins: event, fwd, greeks, capital, 60/30/10
│   └── test_anth_vol_surface.py    # 21 pins: wings, smile, interpolant, calendar hole
├── documentation/
│   ├── IMPLEMENTATION_IPO_BOOK.md  # Identities, API, test pins, surface rules
│   └── DEV_SURFACE_3D.md           # 3D mesh checklist
└── charts/ipo/                     # Note figures 01–13
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Interactive Dashboard

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`.

### 3. Kernel tests

```bash
pytest tests -q
```

41 passed. Do not loosen a pin to hide an API change.

### 4. Static note figures (optional)

```bash
python src/render_smiles.py
python src/render_vol_surface_3d.py
python src/render_vol_surface_3d_smooth.py
```

---

## Book Construction Process

The book is determined from the **16 Sep 2026 Strike snapshot** plus Black–Scholes at flat IV. Here's the flow:

### 1. **Snapshot quotes** → `src/anth_ipo_book.py`

`SURFACE_2026_09_16` is nine listed expiries: label, calendar days, ATM IV, 25Δ RR, 25Δ fly.

```text
SPOT = 2170          # HIP-3 units; 1 = $1B implied cap → $2.17T
PERP_NOTIONAL = 1e6
PHASE1_IV = 0.65
FUNDING_ANNUAL = 0.24
```

### 2. **Event extraction** → `event_move`, `forward_vol`

```text
total_variance(iv, days)  = iv² × days / 365
event_move(iv, days, b)   = sqrt( max(iv² − b², 0) × T )
forward_vol(near, far)    = sqrt( (w_far − w_near) / ΔT )
```

Oct 17 → Nov 21 forward vol is **83.8%**. Sep 20 → 21 on the wings is the calendar hole (do not smooth it).

### 3. **Surface reconstruction** → `src/anth_vol_surface.py` (the MAGIC)

Three quotes pin a smile. They do not pin 10Δ wings, and this surface does not invent them.

```python
# Step 1: wings from RR / fly (vol points)
IV_25c = ATM + fly + ½ RR
IV_25p = ATM + fly − ½ RR

# Step 2: 25Δ → strike (r = 0, F = S)
K = S · exp( −N⁻¹(Δ) · σ √τ + ½ σ² τ )

# Step 3: k-quadratic, exact at K = S and both 25Δ strikes
IV(k) = ATM + α k + β k² ,   k = ln(K / S)

# Step 4: calendar interpolation at fixed strike
w(T) = σ(T)² · T
interpolate_iv          # linear in w(T)
interpolate_iv_pchip    # Fritsch–Carlson PCHIP in w(T)
```

**What happens:**
- Front end is a put-skew market (Sep 23 RR = −5.9 vol points).
- Back end is cheap-wing call skew (negative flies from Oct 17).
- RR crosses zero between Sep 26 and Oct 17.
- Solid smiles stop at the 25Δ dots. Past that band is flagged, not drawn as if quoted.
- `quoted_moneyness_band` clips every tenor to its own 25Δ put → 25Δ call. A shared rectangle `0.86–1.16` is what made the yellow wall.

### 4. **Phase-one book** → `phase_one_greeks`

Short 1 unit of `io:ANTH`. Sell 2 ATM Strike puts. The puts’ +0.5 delta each cancel the short perp, so by put-call parity the book is a short straddle financed on Hyperliquid.

```python
n_perp = −notional / spot
n_puts = −2 × |n_perp|

dollar_γ / 1%² = 0.5 × n_puts × bs_gamma × (0.01 S)²
BE_daily       = sqrt( −theta / dollar_γ )
BE_hourly      = BE_daily / sqrt(24)

Strike IM/unit = mark + max(shock × S − OTM, floor × K)
HL IM          = notional / leverage          # ≤ 2x isolated
capital @ 2x   = 500k + Strike IM  ≈  $970k
```

At note defaults: 922 puts, $71.8k premium, +$5,125/day theta, −$1,104 vega/vol pt, BE 3.40% / 0.69%, funding +$658/day.

**Kill:** no new short-vol week after the S-1 is public, or if 72-hour realized exceeds implied.

### 5. **Phase-two book** → `phase_two_book`, `apply_early_listing_rule`

Keep the short perp. Neutralize it with a calendar: long Nov 21 ATM calls + short Oct 17 ATM puts. That buys the listing window at 84% forward vol against SPCX realized 105–130%.

```python
listing < 30d  → swap_near_to_far   # buy back Oct puts, replace with Nov calls
listing > 65d  → exit_after_far
else           → hold
```

### 6. **Expected return** → `weighted_expected_return`

Full book, 16 Sep → Nov 21, $1M notional, 60 / 30 / 10 on in-window / slip / early:

```text
0.60 × in_window + 0.30 × slip + 0.10 × early
= +10.6% at 24% funding
= +6.8% at 0% funding
```

Never quote the 10.6% without the weights and the funding-on/off pair.

### 7. **Dashboard** → `app.py` & `src/dashboard_charts.py`

Sidebar knobs (spot, notional, sold-put IV, tenor, funding, leverage, listing day) recompute greeks, capital, and the timing rule.

| Section | What it shows |
|---------|----------------|
| **Overview** | Fwd vol, weekend hole, weighted return, phase cards, kill switches |
| **Term** | ATM term structure, event-variance vs SPCX, 25Δ RR/fly, listed quotes |
| **Surface** | Smiles, heatmap, Breeden–Litzenberger density, 3D `interpolate_iv` mesh, PCHIP delta sheet, calendar violations |
| **Book** | Live phase-one greeks, phase-two legs, HL + Strike capital, timing rule |
| **Returns** | Scenario bars, attribution, 60/30/10 with and without funding |

3D is a mesh of `interpolate_iv`, not a triangulation of the nine ATM quotes. Scatter listed ATM at `(1.0, days, ATM)` in the same units as Z — if those dots float, the interpolant is broken.

---

| Component | What it does |
|-----------|-------------|
| **Snapshot** | Nine expiries: ATM, 25Δ RR, 25Δ fly |
| **wings_from_rr_fly** | Three quotes → 25Δ call/put IV |
| **iv_from_strike** | k-quadratic smile, exact at ATM and both wings |
| **interpolate_iv** | **Linear in w(T) at fixed strike** ← **THE SURFACE** |
| **interpolate_iv_pchip_delta** | PCHIP in w(T) on the call-delta rectangle |
| **phase_one_greeks** | Short perp + 2× short ATM puts → premium, θ, ν, γ, BE, IM |
| **phase_two_book** | Calendar + timing rule |
| **weighted_expected_return** | 60/30/10, funding on/off |

**The key insight:** a listed expiry is not an ATM vol. It is ATM + 25Δ RR + 25Δ fly. Those three numbers pin a smile. Calendar interpolation is in **total variance**, not in IV. The weekend trench and the Sep 20→21 wing hole are the snapshot — do not “fix” them.

---

## References

- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637–654.
- Breeden, D. T., & Litzenberger, R. H. (1978). Prices of state-contingent claims implicit in option prices. *Journal of Business*, 51(4), 621–651.
- Fritsch, F. N., & Carlson, R. E. (1980). Monotone piecewise cubic interpolation. *SIAM Journal on Numerical Analysis*, 17(2), 238–246.
- White Star Capital Liquid Fund (16 September 2026). *Selling the wait, buying the listing.* Internal research note.

---

*Nothing in this implementation is investment advice.*
