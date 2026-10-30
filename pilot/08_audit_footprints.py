"""Week 1 - footprint and label audit (automatic checks).

1. EMS inventory vs our table: the EMS summary tables (xlsx in each GRA zip) count every
   building EMS inventoried in the AOI, including 'No visible damage' points that are NOT
   in the public builtUpP layer. Compare those totals with our footprints and labels.
2. Duplicate footprints: overlapping or near-coincident footprints from different
   sources (OSM / Google / Microsoft) that Overture did not merge.
3. Footprint size by source and by label (small sheds, slivers).
4. Label plausibility by AOI: destroyed buildings far above the flood path, image sources.

Writes results/audit_*.csv and prints a summary.
"""
import glob
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

OUT = pathlib.Path("results")
EMS = sorted(p for p in glob.glob("data/raw/ems/EMSR927_AOI*_v*") if not p.endswith(".zip"))
UTM = "EPSG:32645"

def ems_inventory():
    rows = []
    for d in EMS:
        aoi = pathlib.Path(d).name.split("_")[1]
        x = glob.glob(f"{d}/*summaryTable*.xlsx")[0]
        sheet = [s for s in pd.ExcelFile(x).sheet_names if s.startswith("_builtUpP") and s.endswith("_aoi")][0]
        t = pd.read_excel(x, sheet_name=sheet, header=None)
        hdr = t.index[t.iloc[:, 0].astype(str).str.strip() == "OID"][0]
        t = pd.read_excel(x, sheet_name=sheet, header=hdr)
        counts = t.groupby("damage_gra")["Count"].sum().to_dict()
        src = gpd.read_file(glob.glob(f"{d}/*.gpkg")[0],
                            layer=[l for l in pyogrio.list_layers(glob.glob(f"{d}/*.gpkg")[0])[:, 0]
                                   if l.startswith("source")][0])
        post = src[src["eventphase"] == "Post-event"]
        rows.append(dict(aoi=aoi, ems_destroyed=counts.get("Destroyed", 0), ems_damaged=counts.get("Damaged", 0),
                         ems_possibly=counts.get("Possibly damaged", 0),
                         ems_no_visible_damage=counts.get("No visible damage", 0),
                         ems_total=sum(counts.values()),
                         post_event_imagery="; ".join(f"{a} {b}" for a, b in zip(post["source_nam"], post["sensor_res"]))))
    return pd.DataFrame(rows)

def main():
    lab = gpd.read_file("data/processed/buildings_labelled.gpkg").to_crs(UTM)
    fp = lab[lab["kind"] == "footprint"].copy()

    inv = ems_inventory()
    ours = lab.groupby("aoi").agg(our_rows=("bid", "size"), our_footprints=("kind", lambda s: (s == "footprint").sum()),
                                  our_affected=("affected", "sum"),
                                  our_not_affected=("affected", lambda s: (s == 0).sum()))
    clean = lab[~((lab["rank"] == 0) & lab["in_event"])]
    ours["our_not_affected_clean"] = clean[clean.affected == 0].groupby("aoi").size()
    inv = inv.merge(ours, left_on="aoi", right_index=True)
    inv["ems_affected"] = inv[["ems_destroyed", "ems_damaged", "ems_possibly"]].sum(axis=1)
    inv["footprints_minus_ems_total"] = inv["our_footprints"] - inv["ems_total"]
    inv.to_csv(OUT / "audit_ems_inventory.csv", index=False)
    pd.set_option("display.width", 250)
    print("== 1. EMS inventory (summary tables) vs our table ==")
    print(inv[["aoi", "ems_affected", "ems_no_visible_damage", "ems_total", "our_footprints", "our_affected",
               "our_not_affected", "our_not_affected_clean", "footprints_minus_ems_total"]].to_string(index=False))
    print(inv[["aoi", "post_event_imagery"]].to_string(index=False))

    fp["cx"], fp["cy"] = fp.centroid.x, fp.centroid.y
    pairs = gpd.sjoin(fp[["bid", "source", "aoi", "rank", "footprint_m2", "geometry"]],
                      fp[["bid", "source", "rank", "footprint_m2", "geometry"]], predicate="intersects",
                      lsuffix="a", rsuffix="b")
    pairs = pairs[pairs["bid_a"] < pairs["bid_b"]]
    over = []
    for _, r in pairs.iterrows():
        ga, gb = fp.loc[fp.bid == r.bid_a, "geometry"].iloc[0], fp.loc[fp.bid == r.bid_b, "geometry"].iloc[0]
        over.append(ga.intersection(gb).area / min(ga.area, gb.area))
    pairs["overlap_of_smaller"] = over
    dup = pairs[pairs["overlap_of_smaller"] >= 0.5]
    near = gpd.sjoin_nearest(fp[["bid", "source", "aoi", "rank", "geometry"]].set_geometry(fp.centroid),
                             fp[["bid", "source", "rank", "geometry"]].set_geometry(fp.centroid).rename(
                                 columns={"bid": "bid_n", "source": "source_n", "rank": "rank_n"}),
                             max_distance=8, distance_col="d", exclusive=True)
    near = near[near["source"] != near["source_n"]]
    print("\n== 2. Possible duplicate footprints ==")
    print(f"overlapping pairs (overlap >= 50% of smaller footprint): {len(dup)}; "
          f"mixed-label pairs (one affected, one not): {int(((dup.rank_a > 0) != (dup.rank_b > 0)).sum())}")
    print(f"centroids < 8 m apart, different sources: {len(near)} footprints; "
          f"of which one affected and one not: {int(((near['rank'] > 0) != (near['rank_n'] > 0)).sum())}")
    print(near.groupby("aoi").size().rename("near_pairs").to_string())
    dup.drop(columns="geometry").to_csv(OUT / "audit_duplicate_pairs.csv", index=False)

    fp["size_class"] = pd.cut(fp["footprint_m2"], [0, 10, 20, 40, 80, 160, 1e9],
                              labels=["<10", "10-20", "20-40", "40-80", "80-160", ">160"])
    fp["label"] = np.where(fp["rank"] > 0, "affected", np.where(fp["in_event"], "unlabelled in event", "not affected"))
    size_tab = pd.crosstab([fp["source"]], fp["size_class"])
    print("\n== 3. Footprint size (m2) by source ==")
    print(size_tab.to_string())
    print("\nMedian footprint (m2) by label:", fp.groupby("label")["footprint_m2"].median().round(0).to_dict())
    print("Match rate of EMS points by footprint source (affected footprints / all footprints):")
    print(fp.groupby("source")["rank"].apply(lambda s: round((s > 0).mean(), 3)).to_string())
    size_tab.to_csv(OUT / "audit_footprint_size.csv")

    print("\n== 4. Label plausibility ==")
    for aoi, g in clean.groupby("aoi"):
        d = g[g["destroyed"] == 1]
        print(f"{aoi}: destroyed n={len(d)}, HAND median {d.hand.median():.1f} m, "
              f"90th pct {d.hand.quantile(.9):.1f} m, share > 60 m: {(d.hand > 60).mean():.1%}; "
              f"inside observed-event polygon: {d.in_event.mean():.1%}")
    odd = clean[(clean["destroyed"] == 1) & (clean["hand"] > 60)]
    odd[["bid", "aoi", "grade", "hand", "dist", "along_km", "in_event", "lon", "lat"]].to_csv(
        OUT / "audit_destroyed_high_hand.csv", index=False)
    print(f"destroyed buildings with HAND > 60 m written for visual check: {len(odd)}")

if __name__ == "__main__":
    main()
