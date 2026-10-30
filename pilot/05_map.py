"""Study-area map for the proposal: hillshade, main channel, graded buildings, AOI extents."""
import glob
import pathlib

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from clean import load_clean
from scipy import ndimage
from rasterio.windows import from_bounds

OUT = pathlib.Path("../proposal/figures")
b = gpd.read_file("data/processed/buildings_labelled.gpkg")
b = b[b["bid"].isin(load_clean(verbose=False)[0]["bid"])]
aois = gpd.GeoDataFrame(
    [{"aoi": pathlib.Path(f).name.split("_")[1], "geometry": gpd.read_file(f).to_crs(32645).geometry.union_all()}
     for f in sorted(glob.glob("data/raw/ems/EMSR927_AOI*_v*/*areaOfInterestA*.json"))], crs=32645)
x0, y0, x1, y1 = aois.total_bounds
pad = 4000
x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad

with rasterio.open("data/interim/dem_raw.tif") as r:
    w = from_bounds(x0, y0, x1, y1, r.transform)
    dem = r.read(1, window=w)
    dem[dem <= 0] = np.nan
    ext = rasterio.windows.bounds(w, r.transform)
with rasterio.open("data/interim/channel.tif") as r:
    ch = r.read(1, window=from_bounds(x0, y0, x1, y1, r.transform))
with rasterio.open("data/interim/flood_path.tif") as r:
    fp = r.read(1, window=from_bounds(x0, y0, x1, y1, r.transform))

gy, gx = np.gradient(dem, 30.0)
slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
aspect = np.arctan2(-gx, gy)
az, alt = np.radians(315), np.radians(45)
hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
hs = np.ma.masked_invalid(hs)

fig, ax = plt.subplots(figsize=(6.2, 7.6))
ax.set_facecolor("white")
extent = (ext[0], ext[2], ext[1], ext[3])
ax.imshow(hs, cmap="Greys_r", extent=extent, alpha=0.9)
ax.imshow(np.ma.masked_where(ch < 0.5, ch), cmap="Blues", vmin=0, vmax=3, extent=extent)
fp_d = ndimage.binary_dilation(fp > 0.5, iterations=2)
ax.imshow(np.ma.masked_where(~fp_d, fp_d.astype(float)), cmap="winter", vmin=0, vmax=1, extent=extent)
aois.boundary.plot(ax=ax, color="black", lw=0.8, ls="--")
for _, r in aois.iterrows():
    c = r.geometry.bounds
    name = {"AOI01": "Syaprubesi", "AOI02": "Timure", "AOI03": "Betrawati–Bidur", "AOI05": "Phosretar"}[r.aoi]
    ax.annotate(name, (c[2], c[3]), fontsize=8, xytext=(3, 3), textcoords="offset points",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
colours = {"Not affected": "#8d99ae", "Possibly damaged": "#f2c14e", "Damaged": "#f78154", "Destroyed": "#b3001b"}
for g in ["Not affected", "Possibly damaged", "Damaged", "Destroyed"]:
    s = b[b.grade == g]
    ax.scatter(s.geometry.centroid.x, s.geometry.centroid.y, s=1.5 if g == "Not affected" else 3,
               c=colours[g], label=g, lw=0)
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_xticks([]); ax.set_yticks([])
ax.plot([], [], color=plt.cm.winter(1.0), lw=2, label="Flood path (traced)")
ax.legend(loc="lower right", fontsize=7, markerscale=4, framealpha=0.9, title="EMSR927 grade", title_fontsize=7)
ax.plot([x0 + 2000, x0 + 12000], [y0 + 2000, y0 + 2000], color="k", lw=2)
ax.text(x0 + 7000, y0 + 2600, "10 km", ha="center", fontsize=8)
ax.annotate("N", xy=(x0 + 2500, y1 - 2000), xytext=(x0 + 2500, y1 - 5000), ha="center",
            arrowprops=dict(arrowstyle="-|>", color="k"), fontsize=10)
ax.set_title("Bhote Koshi–Trishuli corridor, Rasuwa and Nuwakot, Nepal", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "study_area.png", dpi=200)
print("saved")
