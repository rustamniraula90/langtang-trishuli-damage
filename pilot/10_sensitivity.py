"""Week 1 - sensitivity of the pilot result to data-preparation choices.

Variants (each evaluated with the same logistic-regression transfer test as 04_model.py):
  channel   flood path (main) vs all rivers >= 200 / 500 / 1000 km2
  DEM       Copernicus GLO-30 (surface model, main) vs FABDEM (bare earth)
  labels    ambiguous rows dropped (main) vs relabelled as possibly damaged
  negatives all footprint sources (main) vs OpenStreetMap only

Outputs results/sensitivity.csv and results/dem_comparison.csv.
"""
import importlib
import pathlib
import zipfile

import numpy as np
import pandas as pd
import rasterio
from pysheds.grid import Grid
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clean import load_clean

terrain = importlib.import_module("02_terrain")
INTERIM = pathlib.Path("data/interim")
FAB_ZIP = pathlib.Path("data/raw/fabdem/N20E080-N30E090_FABDEM_V1-2.zip")
FAB_DIR = pathlib.Path("data/raw/fabdem/tiles")
FAB_TILES = ["N27E084", "N27E085", "N28E084", "N28E085"]
F4 = ["hand", "dist", "slope", "relief"]
F2 = ["hand", "dist"]

def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))

def transfer(df, feats, target):
    """Leave-one-region-out AUC for each region, plus the mean."""
    out = {}
    for reg in sorted(df["region"].unique()):
        tr, te = df[df.region != reg], df[df.region == reg]
        if te[target].nunique() < 2:
            continue
        p = logreg().fit(tr[feats], tr[target]).predict_proba(te[feats])[:, 1]
        out[reg.split(" ")[0]] = roc_auc_score(te[target], p)
    out["mean"] = np.mean(list(out.values()))
    return out

def sample(df, arr, profile):
    """Sample a raster (same grid as dem_raw.tif) at building centroids given as lon/lat."""
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", terrain.DST_CRS, always_xy=True).transform(df["lon"].values, df["lat"].values)
    col, row = ~profile["transform"] * (np.asarray(x), np.asarray(y))
    return arr[row.astype(int), col.astype(int)]

def routed_terrain(dem_path):
    """Condition a DEM, route flow, and return HAND/dist to all-river thresholds and to the flood path."""
    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))
    dem_c = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    fdir = grid.flowdir(dem_c)
    acc = np.asarray(grid.accumulation(fdir))
    cell_km2 = terrain.CELL ** 2 / 1e6
    out = {}
    for thr in [200, 500, 1000]:
        mask = acc >= thr / cell_km2
        out[f"all{thr}"] = terrain.hand_and_distance(grid, fdir, dem_c, mask)[:2]

    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", terrain.DST_CRS, always_xy=True).transform(*terrain.ENTRY_LONLAT)
    channel = acc >= terrain.CHANNEL_KM2 / cell_km2
    sx, sy = grid.snap_to_mask(terrain.as_mask(channel, dem_c), (x, y))
    col, row = ~dem_c.affine * (float(sx), float(sy))
    fd = np.asarray(fdir)
    up, upd = terrain.trace_upstream(fd, acc, int(row), int(col), terrain.UPSTREAM_KM * 1000, terrain.CHANNEL_KM2 / cell_km2)
    dn, dnd = terrain.trace_downstream(fd, int(row), int(col))
    path = np.zeros_like(channel)
    for r, c in up + dn:
        path[r, c] = True
    out["path"] = terrain.hand_and_distance(grid, fdir, dem_c, path)[:2]
    d = np.asarray(dem_c, dtype="float32")
    gy, gx = np.gradient(d, terrain.CELL)
    out["slope"] = np.degrees(np.arctan(np.hypot(gx, gy)))
    from scipy import ndimage
    out["relief"] = ndimage.maximum_filter(d, size=17) - ndimage.minimum_filter(d, size=17)
    return out

def fabdem_on_glo_grid():
    """Extract the four FABDEM tiles and resample them onto the exact dem_raw.tif grid."""
    dst_path = INTERIM / "fabdem_raw.tif"
    if dst_path.exists():
        return dst_path
    FAB_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(FAB_ZIP) as z:
        names = [n for n in z.namelist() if any(t in n for t in FAB_TILES) and n.endswith(".tif")]
        for n in names:
            if not (FAB_DIR / pathlib.Path(n).name).exists():
                (FAB_DIR / pathlib.Path(n).name).write_bytes(z.read(n))
    srcs = [rasterio.open(p) for p in sorted(FAB_DIR.glob("*.tif"))]
    arr, tr = merge(srcs, bounds=terrain.CROP, nodata=np.nan)
    with rasterio.open(INTERIM / "dem_raw.tif") as ref:
        prof = ref.profile.copy()
        dst = np.full((ref.height, ref.width), np.nan, dtype="float32")
        reproject(arr[0].astype("float32"), dst, src_transform=tr, src_crs=srcs[0].crs, dst_transform=ref.transform,
                  dst_crs=ref.crs, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    with rasterio.open(dst_path, "w", **prof) as f:
        f.write(dst, 1)
    return dst_path

def main():
    base, _ = load_clean(verbose=False)
    with rasterio.open(INTERIM / "dem_raw.tif") as r:
        prof = r.profile
    rows = []

    def record(name, df, feats=F4):
        for target in ["affected", "destroyed"]:
            res = transfer(df, feats, target)
            rows.append(dict(variant=name, features="+".join(feats), target=target, n=len(df), **res))

    record("main (flood path, GLO-30, R1-R3)", base)
    record("main (flood path, GLO-30, R1-R3)", base, F2)

    dems = {"GLO-30": INTERIM / "dem_raw.tif"}
    if FAB_ZIP.exists():
        dems["FABDEM"] = fabdem_on_glo_grid()
    dem_cmp = {}
    for dem_name, path in dems.items():
        t = routed_terrain(path)
        for ch in ["path", "all200", "all500", "all1000"]:
            df = base.copy()
            df["hand"], df["dist"] = sample(df, t[ch][0], prof), sample(df, t[ch][1], prof)
            df["slope"], df["relief"] = sample(df, t["slope"], prof), sample(df, t["relief"], prof)
            if ch == "path":
                dem_cmp[dem_name] = df[["bid", "hand", "affected"]].copy()
            if dem_name == "GLO-30" and ch == "path":
                continue
            record(f"{dem_name}, channel={ch}", df)
            record(f"{dem_name}, channel={ch}", df, F2)
    if "FABDEM" in dem_cmp:
        m = dem_cmp["GLO-30"].merge(dem_cmp["FABDEM"], on="bid", suffixes=("_glo", "_fab"))
        m["diff"] = m["hand_fab"] - m["hand_glo"]
        summ = dict(median_diff_m=m["diff"].median(), mean_abs_diff_m=m["diff"].abs().mean(),
                    share_abs_diff_gt_5m=(m["diff"].abs() > 5).mean(),
                    auc_hand_glo=roc_auc_score(m["affected_glo"], -m["hand_glo"]),
                    auc_hand_fab=roc_auc_score(m["affected_glo"], -m["hand_fab"]))
        pd.DataFrame([summ]).to_csv("results/dem_comparison.csv", index=False)
        m.to_csv("results/dem_comparison_buildings.csv", index=False)
        print("DEM comparison:", {k: round(v, 3) for k, v in summ.items()})

    record("ambiguous rows relabelled as possibly damaged", load_clean(ambiguous="affected", verbose=False)[0])
    record("negatives: OpenStreetMap footprints only", load_clean(negatives="osm", verbose=False)[0])

    res = pd.DataFrame(rows)
    res.to_csv("results/sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
