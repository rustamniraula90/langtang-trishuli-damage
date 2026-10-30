"""High-resolution re-render of the label-audit chips for the researcher's review.

Re-reads the SAME pre/post images recorded in results/audit/label_audit.csv (so the HD chips match the
first-pass chips sample for sample), but at native resolution (0.2 m/px) over a wider window:
  top row    160 m x 160 m context, pre | post
  bottom row  80 m x 80 m zoom,     pre | post
Footprint outlined in cyan. Also writes results/audit/image_urls.csv with the COG URLs so any image can be
opened at full resolution in QGIS (Layer > Add Raster Layer > Protocol HTTP(S)). Needs internet; ~1 s per read.

Usage: .venv/bin/python 29_hd_chips.py [sample_id ...]   (no args = all 200)
Output: results/audit/chips_hd/<sample_id>.png
"""
import pathlib
import sys

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

import os
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
from importlib import import_module
m09 = import_module("09_label_audit")

UTM = m09.UTM
OUT = pathlib.Path("results/audit/chips_hd"); OUT.mkdir(parents=True, exist_ok=True)
MPP = 0.2
WINDOWS = [80.0, 40.0]

def read(url, x, y, half):
    px = int(round(2 * half / MPP))
    with rasterio.open(url) as src, WarpedVRT(src, crs=UTM, resampling=Resampling.cubic) as vrt:
        w = from_bounds(x - half, y - half, x + half, y + half, vrt.transform)
        arr = vrt.read(indexes=[1, 2, 3], window=w, out_shape=(3, px, px), resampling=Resampling.cubic)
    arr = np.moveaxis(arr, 0, -1).astype("float32")
    valid = arr[arr > 0]
    if valid.size == 0:
        return None
    lo, hi = np.percentile(valid, [1, 99])
    return np.clip((arr - lo) / max(hi - lo, 1), 0, 1)

def main():
    s = pd.read_csv("results/audit/label_audit.csv", dtype=str, keep_default_na=False)
    if len(sys.argv) > 1:
        s = s[s["sample_id"].isin(sys.argv[1:])]
    idx = m09.oam_index()
    key = idx["title"] + " (" + idx["date"] + ")"
    url_of = dict(zip(key, idx["url"]))
    pd.DataFrame({"sample_id": s["sample_id"], "pre_image": s["pre_image"], "pre_url": s["pre_image"].map(url_of),
                  "post_image": s["post_image"], "post_url": s["post_image"].map(url_of)}
                 ).to_csv("results/audit/image_urls.csv", index=False)
    geo = gpd.read_file("data/processed/buildings_labelled.gpkg")[["bid", "geometry"]].to_crs(UTM)
    geo = geo.set_index("bid")["geometry"]
    for _, r in s.iterrows():
        g = geo.get(r.bid)
        if g is None:
            print(r.sample_id, "no footprint; skipped"); continue
        x, y = g.centroid.x, g.centroid.y
        fig, axes = plt.subplots(2, 2, figsize=(10, 10.6), dpi=160)
        for row, half in zip(axes, WINDOWS):
            for ax, name in zip(row, ["pre", "post"]):
                ax.set_xticks([]); ax.set_yticks([])
                url = url_of.get(r[f"{name}_image"])
                chip = None
                if url:
                    try:
                        chip = read(url, x, y, half)
                    except Exception as e:
                        print(r.sample_id, name, "read failed:", e)
                if chip is not None:
                    ax.imshow(chip, extent=(x - half, x + half, y - half, y + half), interpolation="none")
                    gs = gpd.GeoSeries([g], crs=UTM)
                    (gs.boundary if g.geom_type != "Point" else gs.buffer(3).boundary).plot(ax=ax, color="cyan", lw=1.2)
                else:
                    ax.text(0.5, 0.5, "no image", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(x - half, x + half); ax.set_ylim(y - half, y + half)
                ax.set_title(f"{name}  {r[f'{name}_image']}  |  {int(2*half)} m window", fontsize=7)
        fig.suptitle(f"{r.sample_id}   EMS: {r.ems_grade}   HAND {float(r.hand):.0f} m   dist {float(r.dist):.0f} m   "
                     f"first pass: {r.first_pass}", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"{r.sample_id}.png")
        plt.close(fig)
        print(r.sample_id, "ok")

if __name__ == "__main__":
    main()
