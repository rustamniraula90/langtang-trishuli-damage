"""Step 2 - terrain rasters from the Copernicus GLO-30 DEM.

Two channel definitions are produced:
  * FLOOD PATH (main): the D8 flow path that the 2026 flood actually followed,
    traced downstream from the Nepal-China border at Rasuwagadhi (and a short
    distance upstream of it, along the main stem). Tributaries such as the
    Tadi Khola are NOT part of it.
  * ALL CHANNELS (pilot v1, kept for comparison): every cell draining >= CHANNEL_KM2.

Outputs (in data/interim, UTM 45N, 30 m):
  dem_raw.tif, dem.tif   raw / conditioned elevation
  acc_km2.tif            upstream drainage area (km2)
  hand.tif, dist.tif     HAND and distance (m) to the FLOOD PATH
  along_km.tif           along-river distance (km, from the border) of each cell's
                         nearest flood-path cell; negative = upstream of the border
  hand_all.tif, dist_all.tif   HAND and distance to ALL channels >= CHANNEL_KM2
  slope.tif, relief.tif  terrain slope (deg), local relief in a 510 m window
  channel.tif, flood_path.tif  channel masks
"""
import pathlib

import numpy as np
import rasterio
from pyproj import Transformer
from pysheds.grid import Grid
from pysheds.sview import Raster
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy import ndimage

RAW = pathlib.Path("data/raw/dem")
OUT = pathlib.Path("data/interim")
OUT.mkdir(parents=True, exist_ok=True)

CROP = (84.85, 27.70, 85.75, 28.99)
CHANNEL_KM2 = 500.0
DST_CRS = "EPSG:32645"
CELL = 30.0
ENTRY_LONLAT = (85.378, 28.276)
UPSTREAM_KM = 15.0

OFFSETS = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1)}

def mosaic_and_project():
    srcs = [rasterio.open(p) for p in sorted(RAW.glob("*.tif"))]
    arr, tr = merge(srcs, bounds=CROP, nodata=np.nan)
    arr = arr.astype("float32")
    print("DEM gap cells after mosaic:", int(np.isnan(arr).sum()))
    src_crs = srcs[0].crs
    h, w = arr.shape[1:]
    left, top = tr.c, tr.f
    right, bottom = left + w * tr.a, top + h * tr.e
    dtr, dw, dh = calculate_default_transform(src_crs, DST_CRS, w, h, left, bottom, right, top, resolution=CELL)
    dst = np.full((dh, dw), np.nan, dtype="float32")
    reproject(arr[0], dst, src_transform=tr, src_crs=src_crs,
              dst_transform=dtr, dst_crs=DST_CRS, resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    profile = dict(driver="GTiff", height=dh, width=dw, count=1, dtype="float32",
                   crs=DST_CRS, transform=dtr, nodata=np.nan, compress="deflate")
    with rasterio.open(OUT / "dem_raw.tif", "w", **profile) as f:
        f.write(dst, 1)
    return profile

def write(name, arr, profile):
    with rasterio.open(OUT / name, "w", **profile) as f:
        f.write(np.asarray(arr, dtype="float32"), 1)

def step_length(dr, dc):
    return CELL * (np.sqrt(2.0) if dr and dc else 1.0)

def trace_downstream(fdir, r, c):
    """Follow D8 directions from (r, c) to the grid edge; return cells and cumulative distance (m)."""
    cells, dist = [(r, c)], [0.0]
    while True:
        code = int(fdir[r, c])
        if code not in OFFSETS:
            break
        dr, dc = OFFSETS[code]
        r, c = r + dr, c + dc
        if not (0 <= r < fdir.shape[0] and 0 <= c < fdir.shape[1]):
            break
        dist.append(dist[-1] + step_length(dr, dc))
        cells.append((r, c))
    return cells, np.array(dist)

def trace_upstream(fdir, acc, r, c, max_m, min_cells):
    """Walk up the main stem: at each step move to the inflowing neighbour with the largest accumulation."""
    cells, dist = [], []
    walked = 0.0
    while walked < max_m:
        best = None
        for code, (dr, dc) in OFFSETS.items():
            rr, cc = r - dr, c - dc
            if 0 <= rr < fdir.shape[0] and 0 <= cc < fdir.shape[1] and int(fdir[rr, cc]) == code:
                if best is None or acc[rr, cc] > acc[best[0], best[1]]:
                    best = (rr, cc, dr, dc)
        if best is None or acc[best[0], best[1]] < min_cells:
            break
        walked += step_length(best[2], best[3])
        r, c = best[0], best[1]
        cells.append((r, c))
        dist.append(-walked)
    return cells[::-1], np.array(dist[::-1])

def as_mask(arr, like):
    """Boolean pysheds Raster (built via a float comparison so the NaN nodata stays valid)."""
    return Raster(np.asarray(arr, dtype="float32"), viewfinder=like.viewfinder) > 0.5

def hand_and_distance(grid, fdir, dem_c, mask):
    hand = grid.compute_hand(fdir, dem_c, as_mask(mask, dem_c))
    dist, idx = ndimage.distance_transform_edt(~mask, return_indices=True)
    return np.asarray(hand, dtype="float32"), (dist * CELL).astype("float32"), idx

def main():
    profile = mosaic_and_project()
    grid = Grid.from_raster(str(OUT / "dem_raw.tif"))
    dem = grid.read_raster(str(OUT / "dem_raw.tif"))

    dem_c = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    fdir = grid.flowdir(dem_c)
    acc = grid.accumulation(fdir)
    cell_km2 = CELL * CELL / 1e6
    acc_np = np.asarray(acc)
    channel = acc_np >= CHANNEL_KM2 / cell_km2
    print("all-channel cells:", int(channel.sum()), "max drainage km2:", round(float(acc_np.max() * cell_km2)))

    x, y = Transformer.from_crs("EPSG:4326", DST_CRS, always_xy=True).transform(*ENTRY_LONLAT)
    sx, sy = grid.snap_to_mask(as_mask(channel, dem_c), (x, y))
    col, row = ~dem_c.affine * (float(sx), float(sy))
    row, col = int(row), int(col)
    print(f"entry snapped to row {row}, col {col}, drainage {acc_np[row, col] * cell_km2:.0f} km2")
    fdir_np = np.asarray(fdir)
    up_cells, up_dist = trace_upstream(fdir_np, acc_np, row, col, UPSTREAM_KM * 1000, CHANNEL_KM2 / cell_km2)
    down_cells, down_dist = trace_downstream(fdir_np, row, col)
    cells = up_cells + down_cells
    along_m = np.concatenate([up_dist, down_dist])
    path = np.zeros_like(channel)
    along_on_path = np.full(channel.shape, np.nan, dtype="float32")
    for (r, c), d in zip(cells, along_m):
        path[r, c] = True
        along_on_path[r, c] = d / 1000.0
    print(f"flood path: {len(cells)} cells, {along_m[0] / 1000:.1f} km to {along_m[-1] / 1000:.1f} km from the border")

    hand_fp, dist_fp, idx = hand_and_distance(grid, fdir, dem_c, path)
    along_km = along_on_path[idx[0], idx[1]]
    hand_all, dist_all, _ = hand_and_distance(grid, fdir, dem_c, channel)

    demv = np.asarray(dem_c, dtype="float32")
    gy, gx = np.gradient(demv, CELL)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    relief = ndimage.maximum_filter(demv, size=17) - ndimage.minimum_filter(demv, size=17)

    for name, arr in [("dem.tif", demv), ("acc_km2.tif", acc_np * cell_km2),
                      ("hand.tif", hand_fp), ("dist.tif", dist_fp), ("along_km.tif", along_km),
                      ("hand_all.tif", hand_all), ("dist_all.tif", dist_all),
                      ("slope.tif", slope), ("relief.tif", relief),
                      ("channel.tif", channel), ("flood_path.tif", path)]:
        write(name, arr, profile)
    print("wrote terrain rasters to", OUT)

if __name__ == "__main__":
    main()
