"""Week 1 - visual label audit: stratified sample + pre/post image chips.

Imagery: Vantor (formerly Maxar) Open Data, released for this event on OpenAerialMap
(CC BY-NC 4.0; pre-event 2021-2026, post-event 27 Aug - 12 Sep 2026, 0.35-0.7 m).
Windows are read directly from the cloud-optimised GeoTIFFs; nothing large is downloaded.

Sample per AOI (fixed seed): 20 Destroyed, 10 Damaged/Possibly damaged, 20 Not affected,
drawn from the CLEAN table (clean.load_clean). Outputs:
  results/audit/chips/<sample_id>.png      pre | post chip, footprint outlined
  results/audit/sheet_<AOI>.png            contact sheet for quick review
  results/audit/label_audit.csv            one row per sample; fill in 'judgement'
"""
import json
import os
import pathlib
import urllib.request

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from shapely.geometry import box

from clean import load_clean

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
OUT = pathlib.Path("results/audit")
(OUT / "chips").mkdir(parents=True, exist_ok=True)
INDEX = pathlib.Path("data/raw/oam_index.json")
UTM = "EPSG:32645"
HALF = 40.0
PX = 200
SEED = 42
STRATA = {"Destroyed": 20, "Damaged or possibly": 10, "Not affected": 20}

def oam_index():
    if not INDEX.exists():
        url = "https://api.openaerialmap.org/meta?bbox=84.94,27.79,85.45,28.30&limit=200"
        INDEX.write_text(urllib.request.urlopen(url, timeout=120).read().decode())
    res = json.loads(INDEX.read_text())["results"]
    rows = []
    for r in res:
        title = r.get("title", "")
        if "Vantor" not in (r.get("provider") or "") and "Maxar" not in (r.get("provider") or ""):
            continue
        phase = "post" if r.get("acquisition_start", "") >= "2026-08-26" else "pre"
        rows.append(dict(title=title, url=r["uuid"], date=r["acquisition_start"][:10], gsd=r.get("gsd", 9),
                         phase=phase, geometry=box(*r["bbox"])))
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")

def candidates(idx, lon, lat, phase, n=6):
    """Up to n images of the given phase whose footprint contains the point, finest first."""
    from shapely.geometry import Point
    c = idx[(idx.phase == phase) & idx.contains(Point(lon, lat))]
    if phase == "pre":
        c = c[c.date >= "2021-01-01"]
    return c.sort_values(["gsd", "date"]).head(n)

def texture(chip):
    """Mean gradient magnitude of the grey image: clouds, haze and no-data are smooth (low score)."""
    grey = chip.mean(axis=2)
    gy, gx = np.gradient(grey)
    return float(np.hypot(gx, gy).mean())

def best_chip(idx, lon, lat, phase, x, y):
    """Read chips from all candidate images and keep the clearest (highest texture)."""
    best, best_img, best_score = None, None, -1.0
    for _, img in candidates(idx, lon, lat, phase).iterrows():
        try:
            chip = read_chip(img.url, x, y)
        except Exception:
            continue
        if chip is None:
            continue
        s = texture(chip)
        if s > best_score:
            best, best_img, best_score = chip, img, s
    return best, best_img, best_score

def read_chip(url, x, y):
    with rasterio.open(url) as src, WarpedVRT(src, crs=UTM, resampling=Resampling.bilinear) as vrt:
        w = from_bounds(x - HALF, y - HALF, x + HALF, y + HALF, vrt.transform)
        arr = vrt.read(indexes=[1, 2, 3], window=w, out_shape=(3, PX, PX))
    arr = np.moveaxis(arr, 0, -1).astype("float32")
    valid = arr[arr > 0]
    if valid.size == 0:
        return None
    lo, hi = np.percentile(valid, [2, 98])
    return np.clip((arr - lo) / max(hi - lo, 1), 0, 1)

def sample(df):
    df = df.assign(stratum=np.where(df["grade"] == "Destroyed", "Destroyed",
                                    np.where(df["grade"] == "Not affected", "Not affected", "Damaged or possibly")))
    parts = []
    for (aoi, s), g in df.groupby(["aoi", "stratum"]):
        parts.append(g.sample(min(STRATA[s], len(g)), random_state=SEED))
    out = pd.concat(parts).reset_index(drop=True)
    out["sample_id"] = [f"{a}_{i:03d}" for i, a in enumerate(out["aoi"])]
    return out

def main():
    df, _ = load_clean(verbose=False)
    geo = gpd.read_file("data/processed/buildings_labelled.gpkg")[["bid", "geometry"]].to_crs(UTM)
    s = sample(df).merge(geo, on="bid", how="left")
    idx = oam_index()
    rows = []
    for _, r in s.iterrows():
        g = r["geometry"]
        x, y = g.centroid.x, g.centroid.y
        chips, scores = {}, {}
        chips["pre"], pre, scores["pre"] = best_chip(idx, r.lon, r.lat, "pre", x, y)
        chips["post"], post, scores["post"] = best_chip(idx, r.lon, r.lat, "post", x, y)
        fig, axes = plt.subplots(1, 2, figsize=(4, 2.1), dpi=100)
        for ax, name, img in [(axes[0], "pre", pre), (axes[1], "post", post)]:
            ax.set_xticks([]); ax.set_yticks([])
            if chips[name] is not None:
                ax.imshow(chips[name], extent=(x - HALF, x + HALF, y - HALF, y + HALF))
                gs = gpd.GeoSeries([g], crs=UTM)
                (gs.boundary if g.geom_type != "Point" else gs.buffer(3).boundary).plot(ax=ax, color="cyan", lw=1)
            ax.set_xlim(x - HALF, x + HALF); ax.set_ylim(y - HALF, y + HALF)
            ax.set_title(f"{name} {'' if img is None else img.date}", fontsize=6)
        fig.suptitle(f"{r.sample_id}  EMS: {r.grade}  HAND {r.hand:.0f} m", fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "chips" / f"{r.sample_id}.png")
        plt.close(fig)
        rows.append(dict(sample_id=r.sample_id, bid=r.bid, aoi=r.aoi, ems_grade=r.grade, stratum=r.stratum,
                         hand=round(r.hand, 1), dist=round(r.dist, 0), along_km=round(r.along_km, 1),
                         lon=round(r.lon, 6), lat=round(r.lat, 6),
                         pre_image=None if pre is None else f"{pre.title} ({pre.date})",
                         post_image=None if post is None else f"{post.title} ({post.date})",
                         post_texture=round(scores["post"], 4),
                         judgement="", note=""))
    sheet = pd.DataFrame(rows)
    out = OUT / "label_audit.csv"
    if out.exists():
        old = pd.read_csv(out)
        keep = [c for c in ["judgement", "note", "first_pass", "first_pass_note"] if c in old.columns]
        sheet = sheet.drop(columns=[c for c in keep if c in sheet.columns]).merge(old[["sample_id"] + keep], on="sample_id", how="left")
        for c in keep:
            sheet[c] = sheet[c].fillna("")
        print(f"kept {int((sheet['judgement'] != '').sum())} existing judgements from {out}")
    sheet.to_csv(out, index=False)

    for aoi, g in sheet.groupby("aoi"):
        ims = [plt.imread(OUT / "chips" / f"{sid}.png") for sid in g["sample_id"]]
        cols = 5
        rows_n = int(np.ceil(len(ims) / cols))
        h, w = ims[0].shape[:2]
        canvas = np.ones((rows_n * h, cols * w, ims[0].shape[2]))
        for i, im in enumerate(ims):
            canvas[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = im
        plt.imsave(OUT / f"sheet_{aoi}.png", canvas)
    print(sheet.groupby(["aoi", "stratum"]).size().unstack())
    print("chips without post-event image:", int(sheet["post_image"].isna().sum()),
          "| without pre-event image:", int(sheet["pre_image"].isna().sum()))

if __name__ == "__main__":
    main()
