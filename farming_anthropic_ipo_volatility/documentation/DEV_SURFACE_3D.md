# 3D vol surface — mesh `interpolate_iv`

**Audience:** engineering  
**Kernel:** `src/anth_vol_surface.py` (`interpolate_iv`, `surface_grid`, `quoted_moneyness_band`, `calendar_violations`)  
**Renderer:** `src/render_vol_surface_3d.py`  
**Images:** `charts/ipo/12_surface_3d.png` (note) · `charts/ipo/09_surface.png` (pin-check)

Do not triangulate the nine listed ATM quotes and call it a surface. Mesh the total-variance interpolant. Mask cells outside each tenor’s 25Δ strikes before `plot_surface`.

One line: *surface may look like this; the front-left spike may not.*

```bash
python src/render_vol_surface_3d.py
python src/render_vol_surface_3d.py --elev 24 --azim -58 --out charts/ipo/12_surface_3d.png
```

---

## Steps

1. Build slices via `build_surface()`.
2. Mesh `surface_grid` / `interpolate_iv` on a 48×40 grid, days 2–65. Each row’s moneyness runs from that tenor’s 25Δ put to 25Δ call (`quoted_moneyness_band`).
3. **Mask** — do not plot outside 25Δ. A 2-day 25Δ put is about `K/S ≈ 0.97`, not 0.86. A single rectangle `0.86–1.16` for every tenor created the yellow wall and the near-right cliff.
4. `plot_surface` — never `plot_trisurf`, never Delaunay on nine points.
5. Scatter listed ATM at `(1.0, sl.days, sl.atm)` in the **same** units as Z. Assert `interpolate_iv(slices, sl.days, SPOT) == sl.atm` within `8e-4`. If a dot floats, the interpolant is broken — do not move the camera to hide it.
6. Confirm the dark trench near day 3–4 (weekend forward, IV ~59%). Do not “fix” it.
7. Confirm `calendar_violations()` still reports Sep 20→21 on the wings and does **not** report ATM after Sep 23. Smoothing the trench changes the book.
8. Write `charts/ipo/12_surface_3d.png` (note) and `charts/ipo/09_surface.png` (2D pin-check).

---

## Grid

```python
slices = build_surface()
days = np.linspace(2, 65, 48)
u    = np.linspace(0.0, 1.0, 40)            # 0 = 25Δ put, 1 = 25Δ call
for i, d in enumerate(days):
    lo, hi = quoted_moneyness_band(slices, float(d))
    mny = lo + u * (hi - lo)                # mask: only the quoted band
    X[i, :], Y[i, :] = mny, d
    Z[i, :] = [interpolate_iv(slices, d, SPOT * m) for m in mny]
ax.plot_surface(X, Y, Z, cmap="magma", linewidth=0)
```

Do not fill a rectangle `0.86–1.16` for every tenor and call it the surface. `quoted_moneyness_band` is in `anth_vol_surface.py`. Camera: `elev=24`, `azim=-58`.

---

## Keep (these are the snapshot)

- Weekend trench at ~3–4 days, IV down near 59%. That is Sep 20/21, not a plotting bug.
- Front-end put skew: low `K/S` hotter than high `K/S` on the short tenors. Sep 23 is 68.8 put / 62.9 call.
- Back ridge from ~30d to 65d around 73–79%, flatter in moneyness. Negative flies — event is in ATM, not the wings.
- White dots on `K/S = 1`. Those are listed ATMs. They must sit *on* the sheet.

## Do not keep (these are the interpolant lying)

- The yellow wall at short-dated low `K/S` (~0.86, day 2). A 2-day 25Δ put is about `K/S ≈ 0.97`, not 0.86. That spike is the k-quadratic running off the quoted band.
- The vertical purple cliff on the near-right edge. That is the grid boundary, not a traded expiry.

---

## Smooth sheet — call delta × time (do not blur 12)

The scarf on `12_surface_3d.png` is the 25Δ mask in strike space. A 2-day 25Δ put lives near `K/S ≈ 0.97`; a 65-day 25Δ put lives near `0.85`. Mask to that band and the mesh *has* to taper. Gaussian-smoothing Z would also break the weekend variance and the book pins.

Do this instead:

1. Plot the same interpolant in **call delta × time**, Δ ∈ [0.25, 0.75]. That domain is a rectangle on every expiry, so it reads as a sheet.
2. Replace piecewise-linear total variance with **PCHIP** in `w(T)`. Same listed ATMs, no crease at Sep 26 → Oct 17. The weekend trench stays.

```bash
python src/render_vol_surface_3d_smooth.py
```

That’s `charts/ipo/13_surface_3d_delta.png`. Pins still hit listed ATM exactly.

```python
days   = np.linspace(1, 65, 200)
deltas = np.linspace(0.25, 0.75, 120)
Z = np.array(surface_grid_delta_pchip(slices, days.tolist(), deltas.tolist()))
X, Y = np.meshgrid(deltas, days)
ax.plot_surface(X, Y, Z, ...)
ax.scatter([0.5]*9, [sl.days for sl in slices], [sl.atm for sl in slices], ...)
```

If they insist on `K/S` on the x-axis: keep a **rectangular** mesh and fade cells outside that tenor’s 25Δ. Deleting those cells is what draws the cliff.

Do not: blur, spline in IV, or invent 10Δ wings. After any change, `interpolate_iv_pchip(slices, sl.days, spot) == sl.atm` and `interpolate_iv_pchip_delta(slices, sl.days, 0.50) == sl.atm`. The weekend trench must still be there.

