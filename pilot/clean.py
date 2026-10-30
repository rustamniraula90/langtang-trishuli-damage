"""Shared cleaning rules (Week 1). Every modelling script loads data through load_clean().

Rules, applied in order to data/processed/buildings_labelled.csv:
  R1 ambiguous   drop 'not affected' footprints inside the EMS observed-event polygon
                 (mostly small ungraded sheds / split footprints; in_event is post-event
                 information and is used ONLY here, never as a feature)
  R2 tiny        drop UNAFFECTED footprints < MIN_M2 (slivers and sheds that EMS does not
                 grade separately). Affected rows are kept: an EMS grade means a real building,
                 and the error this rule guards against (ungraded sheds counted as undamaged)
                 only affects the negative class.
  R3 twin        drop 'not affected' footprints whose centroid is < TWIN_M from an
                 AFFECTED footprint from a different source (same house mapped twice)

Options for sensitivity analyses:
  ambiguous = "drop" | "affected"      R1 alternative: relabel as affected (possibly damaged)
  negatives = "all" | "osm"            keep only OpenStreetMap footprints as 'not affected'
"""
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd

MIN_M2 = 10.0
LOG_OFFSET_KM = 10.0
TWIN_M = 8.0
REGIONS = {"AOI01": "Upper (Syaprubesi+Timure)", "AOI02": "Upper (Syaprubesi+Timure)",
           "AOI03": "Middle (Betrawati-Bidur)", "AOI05": "Lower (Phosretar)"}

def twin_ids(path="data/processed/buildings_labelled.gpkg"):
    """Ids of unaffected footprints with an affected different-source footprint within TWIN_M."""
    g = gpd.read_file(path)
    g = g[g["kind"] == "footprint"]
    c = g.set_geometry(g.centroid)
    near = gpd.sjoin_nearest(
        c[["bid", "source", "rank", "geometry"]],
        c[["bid", "source", "rank", "geometry"]].rename(columns={"bid": "bid_n", "source": "source_n", "rank": "rank_n"}),
        max_distance=TWIN_M, distance_col="d", exclusive=True)
    near = near[(near["source"] != near["source_n"]) & (near["rank"] == 0) & (near["rank_n"] > 0)]
    return set(near["bid"])

def load_clean(ambiguous="drop", negatives="all", min_m2=MIN_M2, verbose=True):
    df = pd.read_csv("data/processed/buildings_labelled.csv")
    log = [("EMS points + pre-event footprints in analysed areas", len(df))]
    amb = (df["rank"] == 0) & df["in_event"]
    if ambiguous == "drop":
        df = df[~amb].copy()
        log.append(("R1 drop ambiguous unlabelled footprints inside the flood trace", len(df)))
    else:
        df.loc[amb, ["rank", "grade", "affected"]] = [1, "Possibly damaged", 1]
        log.append(("R1' relabel ambiguous footprints as possibly damaged", len(df)))
    tiny = (df["kind"] == "footprint") & (df["footprint_m2"] < min_m2) & (df["rank"] == 0)
    df = df[~tiny].copy()
    log.append((f"R2 drop unaffected footprints < {min_m2:.0f} m2", len(df)))
    twins = df["bid"].isin(twin_ids()) & (df["rank"] == 0)
    df = df[~twins].copy()
    log.append((f"R3 drop unaffected twins < {TWIN_M:.0f} m from an affected footprint", len(df)))
    if negatives == "osm":
        df = df[(df["rank"] > 0) | (df["source"] == "OpenStreetMap")].copy()
        log.append(("S  keep only OpenStreetMap footprints as negatives", len(df)))
    feats = pathlib.Path("data/processed/features.csv")
    if feats.exists():
        df = df.merge(pd.read_csv(feats), on="bid", how="left")
    df["region"] = df["aoi"].map(REGIONS)
    df["block"] = (df["lon"] // 0.01).astype(int).astype(str) + "_" + (df["lat"] // 0.01).astype(int).astype(str)
    df["valley_floor"] = (df["dist"] <= 1000) & (df["hand"] <= 100)
    df["hand_x_along"] = df["hand"] * df["along_km"]
    df["logd"] = np.log(df["along_km"] + LOG_OFFSET_KM)
    df["hand_x_logd"] = df["hand"] * df["logd"]
    if "valley_w20" in df:
        df["log_valley_w20"] = np.log(df["valley_w20"])
        df["log_spi"] = np.log(df["spi"] + 1)
    log = pd.DataFrame(log, columns=["step", "rows"])
    log["removed"] = (-log["rows"].diff()).fillna(0).astype(int)
    if verbose:
        print(log.to_string(index=False))
    return df, log
