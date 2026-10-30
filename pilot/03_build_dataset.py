"""Step 3 - one row per building, with damage label and terrain features.

Labelling rules:
  * Analysed area = EMS area-of-interest polygon minus EMS 'notAnalysed' polygons.
  * Candidate buildings = pre-event Overture footprints (OSM + Google + Microsoft,
    release 2026-08-19) whose centroid lies in the analysed area.
  * An EMS graded point inside a footprint (or within MATCH_M metres of one) gives
    that footprint its grade; if several points hit one footprint, keep the worst.
  * EMS points with no footprint are kept as point-only rows ('ems_only').
  * Footprints with no EMS point are labelled 'Not affected' (EMS grading maps
    list affected buildings only).
"""
import glob
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely import wkb

RAW = pathlib.Path("data/raw")
INTERIM = pathlib.Path("data/interim")
OUT = pathlib.Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
UTM = "EPSG:32645"
MATCH_M = 10.0
GRADE_RANK = {"Not affected": 0, "Possibly damaged": 1, "Damaged": 2, "Destroyed": 3}

ENTRY_LONLAT = (85.378, 28.276)

def ems_layer(aoi_dir, layer):
    f = glob.glob(f"{aoi_dir}/*_{layer}_v*.json")
    return gpd.read_file(f[0]).to_crs(UTM) if f else None

def load_aoi(aoi_dir):
    aoi_id = pathlib.Path(aoi_dir).name.split("_")[1]
    area = ems_layer(aoi_dir, "areaOfInterestA")
    locality = area["locality"].iloc[0]
    analysed = area.geometry.union_all()
    na = ems_layer(aoi_dir, "notAnalysedA")
    if na is not None:
        analysed = analysed.difference(na.geometry.union_all())
    pts = ems_layer(aoi_dir, "builtUpP")[["damage_gra", "obj_type", "geometry"]]
    pts["aoi"] = aoi_id
    event = ems_layer(aoi_dir, "observedEventA")
    return aoi_id, locality, analysed, pts, event.geometry.union_all()

def label_aoi(aoi_id, locality, analysed, pts, event_geom, bld):
    b = bld[bld.centroid.within(analysed)].copy()
    b["aoi"], b["locality"] = aoi_id, locality
    pts = pts[pts.within(analysed.buffer(MATCH_M))].copy()
    pts["pid"] = np.arange(len(pts))
    hit = gpd.sjoin_nearest(pts, b[["bid", "geometry"]], how="left", max_distance=MATCH_M, distance_col="d")
    hit = hit.sort_values("d").drop_duplicates("pid")
    hit["rank"] = hit["damage_gra"].map(GRADE_RANK)
    worst = hit.dropna(subset=["bid"]).groupby("bid")["rank"].max()
    b["rank"] = b["bid"].map(worst).fillna(0).astype(int)
    b["kind"] = "footprint"
    orphan = hit[hit["bid"].isna()]
    o = gpd.GeoDataFrame({"bid": [f"ems_{aoi_id}_{i}" for i in orphan["pid"]],
                          "source": "EMS only", "rank": orphan["rank"].astype(int).values,
                          "aoi": aoi_id, "locality": locality, "kind": "ems_only"},
                         geometry=orphan.geometry.values, crs=UTM)
    out = pd.concat([b, o], ignore_index=True)
    out["in_event"] = out.centroid.within(event_geom)
    stats = dict(aoi=aoi_id, locality=locality, ems_points=len(pts),
                 matched=int(hit["bid"].notna().sum()), ems_only=len(o),
                 footprints=len(b), not_affected=int((b["rank"] == 0).sum()))
    return out, stats

def sample(gdf, name):
    with rasterio.open(INTERIM / f"{name}.tif") as r:
        xy = [(p.x, p.y) for p in gdf.centroid]
        return np.array([v[0] for v in r.sample(xy)], dtype="float32")

def main():
    raw = pd.read_parquet(RAW / "buildings" / "overture_2026-08-19.parquet")
    bld = gpd.GeoDataFrame(raw.drop(columns="geometry"),
                           geometry=[wkb.loads(bytes(g)) for g in raw["geometry"]], crs="EPSG:4326").to_crs(UTM)
    bld = bld.rename(columns={"id": "bid"})[["bid", "source", "geometry"]]
    bld["footprint_m2"] = bld.area

    rows, stats = [], []
    for d in sorted(glob.glob(str(RAW / "ems" / "EMSR927_AOI*_GRA_PRODUCT_v*"))):
        if pathlib.Path(d).suffix == ".zip":
            continue
        out, s = label_aoi(*load_aoi(d), bld)
        rows.append(out)
        stats.append(s)
    df = pd.concat(rows, ignore_index=True)
    print(pd.DataFrame(stats).to_string(index=False))

    for name in ["hand", "dist", "along_km", "hand_all", "dist_all", "slope", "relief", "dem"]:
        df[name] = sample(df, name)
    entry = gpd.GeoSeries(gpd.points_from_xy([ENTRY_LONLAT[0]], [ENTRY_LONLAT[1]]), crs="EPSG:4326").to_crs(UTM)[0]
    df["dist_entry_km"] = df.centroid.distance(entry) / 1000.0
    df["grade"] = df["rank"].map({v: k for k, v in GRADE_RANK.items()})
    df["affected"] = (df["rank"] >= 1).astype(int)
    df["destroyed"] = (df["rank"] == 3).astype(int)
    df["lon"] = df.centroid.to_crs("EPSG:4326").x
    df["lat"] = df.centroid.to_crs("EPSG:4326").y

    cols = ["bid", "aoi", "locality", "kind", "source", "grade", "rank", "affected", "destroyed", "in_event",
            "footprint_m2", "hand", "dist", "along_km", "hand_all", "dist_all", "slope", "relief", "dem",
            "dist_entry_km", "lon", "lat"]
    df[cols].to_csv(OUT / "buildings_labelled.csv", index=False)
    df[cols + ["geometry"]].to_file(OUT / "buildings_labelled.gpkg")
    print(df.groupby(["aoi", "grade"]).size().unstack(fill_value=0))
    print("missing HAND:", int(df["hand"].isna().sum()))

if __name__ == "__main__":
    main()
