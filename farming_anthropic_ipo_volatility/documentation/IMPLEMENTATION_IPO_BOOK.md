# Selling the wait — Python implementation spec

**Audience:** engineering  
**Research:** `RESEARCH_SELLING_THE_WAIT.md`, `HIP3_IPO_Selling_the_Wait.pdf`  
**Kernel:** `src/anth_ipo_book.py`  
**Tests:** `tests/test_anth_ipo_book.py` + `tests/test_anth_vol_surface.py` — 41 passed.  
**3D surface:** `src/render_vol_surface_3d.py` → `charts/ipo/12_surface_3d.png` + `09_surface.png`. Smooth delta sheet: `src/render_vol_surface_3d_smooth.py` → `13_surface_3d_delta.png`. Step list: `DEV_SURFACE_3D.md`.

**Dashboard:** `app.py` — live Plotly figures from the kernel.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sidebar knobs recompute phase-one greeks, capital, the timing rule, and the 60/30/10 return. Surface charts mesh `interpolate_iv` / PCHIP as in `src/render_vol_surface_3d.py`.

Strike and Hyperliquid adapters are out of scope for v1. ATM marks are Black–Scholes at flat IV, r = 0, 365-day year, multiplier 1 USD per HIP-3 price unit. Off-ATM marks go through `src/anth_vol_surface.py`.

---

## Snapshot constants

```text
SPOT              = 2170          # $2.17T
PERP_NOTIONAL     = 1_000_000
PHASE1_IV         = 0.65
PHASE1_PUTS/UNIT  = 2
FUNDING_ANNUAL    = 0.24
DAYS_PER_YEAR     = 365
```

Surface: `SURFACE_2026_09_16` in the kernel (label, days, ATM IV, 25Δ RR, 25Δ fly).

---

## Identities to implement exactly

```text
total_variance(iv, days)     = iv² × days / 365
event_move(iv, days, b)      = sqrt( max(iv² − b², 0) × days / 365 )
forward_vol(iv_n, d_n, iv_f, d_f)
                             = sqrt( (tv_f − tv_n) / ((d_f − d_n)/365) )

phase one qty                n_perp = −notional / spot
                             n_puts = −2 × |n_perp|

dollar gamma per 1%²         = 0.5 × n_puts × bs_gamma × (0.01 × spot)²
daily breakeven              = sqrt( −theta / dollar_gamma )
hourly breakeven             = daily / sqrt(24)

Strike short margin / unit   = mark + max(shock × spot − OTM, floor × strike)
entry shock 20% floor 10%    maintenance shock 12%

HL isolated margin           = notional / leverage
phase one capital @ 2x       = 500_000 + Strike entry IM   ≈ 970k

timing rule                  listing < 30d  → swap Oct puts for Nov calls
                             listing > 65d  → exit phase two
                             else hold

weighted return              0.60×in_window + 0.30×slip + 0.10×early
```

---

## API

```python
event_move(iv, days, baseline) -> float
forward_vol(iv_near, days_near, iv_far, days_far) -> float
phase_one_greeks(spot=2170, notional=1e6, iv=0.65, days=7) -> PhaseOneGreeks
phase_two_book(spot, notional, near=Oct17, far=Nov21) -> PhaseTwoLegs
apply_early_listing_rule(listing_days) -> "swap_near_to_far" | "hold" | "exit_after_far"
phase_one_capital(notional, leverage=2.0) -> float
weighted_expected_return(funding_on=True) -> float
```

`PhaseOneGreeks` fields used by tests: `premium`, `theta_per_day`, `vega_per_vol_point`, `dollar_gamma_per_pct2`, `breakeven_daily`, `breakeven_hourly`, `funding_per_day`, `margin_entry`, `margin_maint`, `delta`, `n_puts`.

Surface API (`src/anth_vol_surface.py`):

```python
wings_from_rr_fly(atm, rr_pts, fly_pts) -> (iv_25c, iv_25p)
build_surface(spot=2170) -> tuple[SliceQuote, ...]
iv_from_strike(slice, strike) -> float          # k-quadratic, exact at ATM and 25Δ K
interpolate_iv(slices, days, strike) -> float   # linear in total variance
interpolate_iv_pchip(slices, days, strike) -> float  # PCHIP in w(T), fixed strike
interpolate_iv_pchip_delta(slices, days, Δ) -> float # PCHIP in w(T), fixed call delta
quoted_moneyness_band(slices, days) -> (lo, hi) # 25Δ K/S at that tenor
calendar_violations(slices, moneyness) -> list
digital_and_density(slice, strike) -> (digital, pdf)
```

Surface tests: `tests/test_anth_vol_surface.py`. Together with the book tests: **41 passed**.

---

## Tests that must stay green

`pytest tests/test_anth_ipo_book.py -q` → 20 passed.

| Bucket | Pin |
|---|---|
| Wings | Sep 18 25Δc/p = 78.15 / 79.25; Oct 17 = 71.2 / 70.4 |
| Event | Oct 17 @ 50% → 15.4%; Nov 21 @ 45/50/60 → 27.5 / 25.9 / 21.8% |
| Fwd | Oct 17 → Nov 21 → 83.8%; Sep 20 → 21 → 26.3% |
| Phase one | 922 puts; premium $71.8k; theta +$5,125; vega −$1,104; γ −$443; BE 3.40% / 0.69%; funding +$658; IM/MM $472k / $312k; \|delta\| < 20 |
| Phase two | fwd 83.8%; timing rule Oct 12 / Nov 9 / day 70 |
| Capital | 6x IM $167k; 2x $500k; phase-one capital ≈ $970k |
| Returns | components sum to total; 60/30/10 = 10.6% with funding, 6.8% without; 65d funding = 4.3% |

Tolerances are in the test file. Do not loosen a pin to hide an API change.

---

## Adapters to add later

| Adapter | Extra test |
|---|---|
| Strike chain | listed Sep 23 65.8% ATM maps onto `Expiry` |
| Strike fees | 40 bp taker, 10 bp maker, 50 bp of intrinsic at settlement — applied outside `bs_price` |
| HL marks | 2170 → $2.17T; isolated 2x liquidation fixture +38% |
| Funding print | 90×8h (or hourly) collapses to `FUNDING_ANNUAL` |
| Bootstrap | 24-hour block resample; do not replace paper fixtures with a new mean |

---

## Policy

- Phase one does not run a new short-vol week after the S-1 or after 72h realized > implied.
- Perp leverage ≤ 2x. Sweep Strike MTM to Hyperliquid daily.
- `strike_spot != live_spot` is an unhedged book, not a rounding error.
- Never quote the 10.6% without the 60/30/10 weights and the funding-on/off pair.

---

## 3D vol surface — mesh `interpolate_iv`

Do not triangulate the nine listed ATM quotes and call it a surface. Mesh the total-variance interpolant already in `src/anth_vol_surface.py`. Mask cells outside each tenor’s 25Δ strikes before `plot_surface`.

One line: *surface may look like this; the front-left spike may not.*

```bash
python src/render_vol_surface_3d.py
python src/render_vol_surface_3d.py --elev 24 --azim -58 --out charts/ipo/12_surface_3d.png
```

3D is for the note (`charts/ipo/12_surface_3d.png`). 2D heatmap is for checking pins (`charts/ipo/09_surface.png`).

### Steps

1. Build slices via `build_surface()`.
2. Mesh `interpolate_iv` on a 48×40 grid, days 2–65. Each row’s moneyness runs from that tenor’s 25Δ put to 25Δ call (`quoted_moneyness_band`).
3. Mask — do not plot outside 25Δ. A 2-day 25Δ put is about `K/S ≈ 0.97`, not 0.86. A single rectangle `0.86–1.16` for every tenor created the yellow wall and the near-right cliff.
4. `plot_surface` — never `plot_trisurf`, never Delaunay on nine points.
5. Scatter listed ATM at `(1.0, sl.days, sl.atm)` in the **same** units as Z. Assert `interpolate_iv(slices, sl.days, SPOT) == sl.atm` within `8e-4`. If a dot floats, the interpolant is broken — do not move the camera to hide it.
6. Confirm the dark trench near day 3–4 (weekend forward, IV ~59%). Do not “fix” it.
7. Confirm `calendar_violations()` still reports Sep 20→21 on the wings and does **not** report ATM after Sep 23. Smoothing the trench changes the book.
8. Write `charts/ipo/12_surface_3d.png` and `charts/ipo/09_surface.png`.

### Grid

```python
slices = build_surface()
days = np.linspace(2, 65, 48)
u    = np.linspace(0.0, 1.0, 40)            # 0 = 25Δ put, 1 = 25Δ call
for i, d in enumerate(days):
    lo, hi = quoted_moneyness_band(slices, float(d))
    mny = lo + u * (hi - lo)
    X[i, :], Y[i, :] = mny, d
    Z[i, :] = [interpolate_iv(slices, d, SPOT * m) for m in mny]
ax.plot_surface(X, Y, Z, cmap="magma", linewidth=0)
```

`quoted_moneyness_band` is in `anth_vol_surface.py`. Camera: `elev=24`, `azim=-58`. Full step list: `DEV_SURFACE_3D.md`.

---

## Smiles — solid only between the wing dots

`python src/render_smiles.py` → `charts/ipo/08_smiles.png`.

Dots and the curve between them are the snapshot. Solid lines past the outer 25Δ dots are not — that is the k-quadratic leaving the quoted band (Sep 23 25Δ call is `K/S ≈ 1.06`, not 1.25). **Solid only between the two wing dots; drop the line outside.** Do not add a 10Δ point to complete the smile. Three quotes pin three points, not the whole x-axis.

---

## Smooth sheet — PCHIP in `w(T)`, call delta × time

Do not Gaussian-blur `12_surface_3d.png`. The scarf is the 25Δ mask in strike space. Plot a rectangle in **call delta × time** instead.

```bash
python src/render_vol_surface_3d_smooth.py
```

Writes `charts/ipo/13_surface_3d_delta.png`. Calendar interpolation is PCHIP in total variance. Same listed ATMs, no crease at Sep 26 → Oct 17, weekend trench stays. Scatter ATM at `(0.50, sl.days, sl.atm)`.

Do not: blur, spline in IV, or invent 10Δ wings. After any change, `interpolate_iv_pchip(slices, sl.days, SPOT) == sl.atm` and the weekend trench must still be there.
