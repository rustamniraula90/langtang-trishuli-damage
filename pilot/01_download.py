"""Step 1 - download raw inputs for the pilot.

Inputs:
  - Copernicus EMS EMSR927 grading products (already copied to data/raw/ems)
  - Copernicus GLO-30 DEM tiles (AWS open data)
  - Overture Maps buildings, release 2026-08-19.0 (pre-event; merges OSM,
    Microsoft and Google Open Buildings footprints)
  - OSM river lines (Overpass) for the channel network
"""
import json
import pathlib
import urllib.parse
import urllib.request
import time
import zipfile

import duckdb

RAW = pathlib.Path("data/raw")

BBOX = (84.90, 27.75, 85.45, 28.32)

DEM_TILES = [
    "Copernicus_DSM_COG_10_N28_00_E085_00_DEM",
    "Copernicus_DSM_COG_10_N27_00_E085_00_DEM",
    "Copernicus_DSM_COG_10_N27_00_E084_00_DEM",
    "Copernicus_DSM_COG_10_N28_00_E084_00_DEM",
]
OVERTURE = "s3://overturemaps-us-west-2/release/2026-08-19.0/theme=buildings/type=building/*"

def unzip_ems():
    for z in sorted((RAW / "ems").glob("*.zip")):
        out = RAW / "ems" / z.stem
        if not out.exists():
            zipfile.ZipFile(z).extractall(out)
        print("ems", z.stem, len(list(out.glob("*.json"))), "json layers")

def download_dem():
    for t in DEM_TILES:
        dst = RAW / "dem" / f"{t}.tif"
        if dst.exists():
            continue
        url = f"https://copernicus-dem-30m.s3.amazonaws.com/{t}/{t}.tif"
        urllib.request.urlretrieve(url, dst)
        print("dem", dst.name, dst.stat().st_size // 1_000_000, "MB")

def download_buildings():
    dst = RAW / "buildings" / "overture_2026-08-19.parquet"
    if dst.exists():
        return
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
    x0, y0, x1, y1 = BBOX
    con.execute(f"""
        COPY (
          SELECT id,
                 sources[1].dataset AS source,
                 height, num_floors, class,
                 ST_AsWKB(geometry) AS geometry
          FROM read_parquet('{OVERTURE}', hive_partitioning=1)
          WHERE bbox.xmin > {x0} AND bbox.xmax < {x1}
            AND bbox.ymin > {y0} AND bbox.ymax < {y1}
        ) TO '{dst}' (FORMAT PARQUET);
    """)
    n = con.execute(f"SELECT count(*), count(DISTINCT source) FROM '{dst}'").fetchone()
    print("buildings", n)

def download_rivers():
    dst = RAW / "osm" / "rivers.json"
    if dst.exists():
        return
    x0, y0, x1, y1 = BBOX
    q = f"[out:json][timeout:180];way[waterway~\"^(river|stream)$\"]({y0},{x0},{y1},{x1});out tags geom;"
    data = urllib.parse.urlencode({"data": q}).encode()
    for attempt in range(5):
        try:
            r = urllib.request.urlopen("https://overpass.kumi.systems/api/interpreter", data, timeout=300).read()
            dst.write_bytes(r)
            print("rivers", len(json.loads(r)["elements"]), "ways")
            return
        except Exception as e:
            print("overpass retry", attempt, e)
            time.sleep(20)
    raise RuntimeError("Overpass download failed")

if __name__ == "__main__":
    unzip_ems()
    download_dem()
    download_rivers()
    download_buildings()
