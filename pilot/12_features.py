"""Week 2 - terrain feature engineering (terrain only; no building attributes).

Per building (sampled at the footprint centroid, or via its nearest flood-path cell):
  local terrain
    curv_profile   profile curvature (1/m), smoothed DEM; > 0 = convex (spur), < 0 = concave (hollow)
    curv_plan      plan curvature (1/m); < 0 = converging flow (hollow / channel), > 0 = diverging (spur)
    twi            topographic wetness index ln(a / tan(slope)); a = specific catchment area (m)
  valley / channel geometry (properties of the nearest flood-path cell)
    valley_w20     width (m) of valley floor with HAND <= 20 m, averaged over a 500 m reach
    valley_w40     same with HAND <= 40 m
    chan_slope     channel gradient over +-500 m along the path (m/m)
    spi            stream-power index = drainage area (km2) x channel slope
    bend           signed bend exposure (1/km): > 0 = building on the OUTER bank of a bend,
                   < 0 = inner bank; magnitude = curvature of the smoothed channel (~1 km scale)
Writes data/processed/features.csv (bid + features); clean.load_clean() merges it.
"""
import pathlib

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import ndimage

INTERIM = pathlib.Path("data/interim")
OUT = pathlib.Path("data/processed/features.csv")
CELL = 30.0
SMOOTH_SIGMA = 1.0
REACH_M = 500.0
SLOPE_HALF_M = 500.0
BEND_WIN_CELLS = 33

def read(name):
    with rasterio.open(INTERIM / f"{name}.tif") as r:
        return r.read(1).astype("float64"), r.profile

def curvatures(dem):
    """Zevenbergen-Thorne style profile and plan curvature, x = east, y = north."""
    z = ndimage.gaussian_filter(np.nan_to_num(dem, nan=np.nanmean(dem)), SMOOTH_SIGMA)
    dz_drow, dz_dcol = np.gradient(z, CELL)
    p, q = dz_dcol, -dz_drow
    r = np.gradient(p, CELL, axis=1)
    t = -np.gradient(q, CELL, axis=0)
    s = np.gradient(p, CELL, axis=0) * -1
    g2 = p ** 2 + q ** 2 + 1e-12
    profile = -(p ** 2 * r + 2 * p * q * s + q ** 2 * t) / (g2 * (1 + g2) ** 1.5)
    plan = -(q ** 2 * r - 2 * p * q * s + p ** 2 * t) / g2 ** 1.5

    plan = np.where(g2 < np.tan(np.radians(2.0)) ** 2, 0.0, plan)

    return profile, plan

def path_order(path_mask, along):
    rows, cols = np.nonzero(path_mask > 0.5)
    order = np.argsort(along[rows, cols])
    return rows[order], cols[order], along[rows, cols][order]

def moving_average(a, w):
    k = np.ones(w) / w
    pad = np.pad(a, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")

def main():
    dem, prof = read("dem")
    acc_km2, _ = read("acc_km2")
    hand, _ = read("hand")
    path, _ = read("flood_path")
    along, _ = read("along_km")
    slope_deg, _ = read("slope")
    transform = prof["transform"]

    curv_profile, curv_plan = curvatures(dem)
    sca = (acc_km2 * 1e6 / CELL ** 2) * CELL
    twi = np.log((sca + CELL) / (np.tan(np.radians(slope_deg)) + 1e-3))

    pr, pc, pa = path_order(path, along)
    n = len(pr)
    x = transform.c + (pc + 0.5) * transform.a
    y = transform.f + (pr + 0.5) * transform.e
    z = dem[pr, pc]
    area = acc_km2[pr, pc]
    s_m = pa * 1000.0

    lo = np.searchsorted(s_m, s_m - SLOPE_HALF_M)
    hi = np.clip(np.searchsorted(s_m, s_m + SLOPE_HALF_M), 0, n - 1)
    chan_slope = np.clip((z[lo] - z[hi]) / np.maximum(s_m[hi] - s_m[lo], CELL), 0, None)
    spi = area * chan_slope

    path_idx_grid = np.full(path.shape, -1, dtype=np.int64)
    path_idx_grid[pr, pc] = np.arange(n)
    _, (ir, ic) = ndimage.distance_transform_edt(path < 0.5, return_indices=True)
    nearest = path_idx_grid[ir, ic]

    widths = {}
    for h in (20, 40):
        floor = (hand <= h) & np.isfinite(hand)
        counts = np.bincount(nearest[floor], minlength=n).astype(float)
        bins = (s_m // REACH_M).astype(int)
        per_bin = pd.Series(counts).groupby(bins).sum()
        bin_len = pd.Series(s_m).groupby(bins).agg(lambda v: max(v.max() - v.min(), CELL))
        widths[h] = (per_bin * CELL ** 2 / bin_len).reindex(bins).to_numpy()

    xs, ys = moving_average(x, BEND_WIN_CELLS), moving_average(y, BEND_WIN_CELLS)
    dx, dy = np.gradient(xs, s_m), np.gradient(ys, s_m)
    ddx, ddy = np.gradient(dx, s_m), np.gradient(dy, s_m)
    kappa = (dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-9) * 1000.0

    b = pd.read_csv("data/processed/buildings_labelled.csv", usecols=["bid", "lon", "lat"])
    bx, by = Transformer.from_crs("EPSG:4326", prof["crs"], always_xy=True).transform(b["lon"].values, b["lat"].values)
    col, row = ~transform * (np.asarray(bx), np.asarray(by))
    row, col = row.astype(int), col.astype(int)
    k = nearest[row, col]

    cross = dx[k] * (by - y[k]) - dy[k] * (bx - x[k])
    bend = -np.sign(cross) * kappa[k]

    feats = pd.DataFrame({
        "bid": b["bid"],
        "curv_profile": curv_profile[row, col], "curv_plan": curv_plan[row, col], "twi": twi[row, col],
        "valley_w20": widths[20][k], "valley_w40": widths[40][k],
        "chan_slope": chan_slope[k], "spi": spi[k], "bend": bend,
    })
    feats.to_csv(OUT, index=False)
    print(feats.describe().T.round(4).to_string())
    print(f"path cells: {n}; buildings: {len(feats)}; NaNs: {int(feats.isna().sum().sum())}")

if __name__ == "__main__":
    main()
