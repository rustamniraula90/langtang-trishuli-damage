# Data dictionary

All paths are relative to `research/pilot/`. `data/raw/` is never edited; everything else is rebuilt by `run_all.sh`.

## 1. Raw inputs (`data/raw/`)

| Path | Content | Source / licence | Notes |
|---|---|---|---|
| `ems/EMSR927_AOI0x_GRA_PRODUCT_vN.zip` (+ unzipped folder) | Copernicus EMS grading product per area | Copernicus EMS, EMSR927; free with attribution | See §1.1 |
| `ems/…/Maps/*.pdf` | Official EMS map sheet per area | same | Used in the visual audit |
| `ems/…/*summaryTable*.xlsx` | EMS counts per class, including "No visible damage" buildings | same | The public point layer omits undamaged buildings; the table has their counts |
| `ems/…/*.gpkg` layer `source_vN` | Imagery EMS used | same | AOI01 WorldView-3 0.3 m; AOI02 Legion 0.3 m; AOI03 ALOS 0.8–1.0 m; AOI05 Pléiades 0.5 m / Pléiades Neo 0.3 m (all post-event) |
| `buildings/overture_2026-08-19.parquet` | 405,975 footprints in the study bbox | Overture Maps release 2026-08-19.0 (ODbL / CDLA) | OSM 232,585; Google 127,727; Microsoft 45,663 |
| `dem/Copernicus_DSM_COG_10_N2{7,8}_00_E08{4,5}_00_DEM.tif` | GLO-30 elevation, 4 tiles | ESA Copernicus DEM, AWS Open Data | A surface model (includes trees and roofs) |
| `fabdem/N20E080-N30E090_FABDEM_V1-2.zip` | Bare-earth GLO-30 (FABDEM v1-2) | University of Bristol, CC BY-NC-SA 4.0 | For the DEM comparison |
| `osm/rivers.json` | OSM waterways | OpenStreetMap (ODbL) | Not used: the download was truncated at 2,000 ways |
| `oam_index.json` | Metadata of OpenAerialMap images over the study area | OpenAerialMap API | Vantor/Maxar Open Data pre/post imagery (CC BY-NC 4.0) for the label audit |

### 1.1 EMS layers used
| Layer | Geometry | Key fields | Use |
|---|---|---|---|
| `builtUpP` | point per affected building | `damage_gra` ∈ {Destroyed, Damaged, Possibly damaged}; `obj_type`; `det_method` = Photo-interpretation | Labels |
| `areaOfInterestA` | polygon | `locality` | Defines the analysed area |
| `notAnalysedA` | polygons | – | Removed from the analysed area |
| `observedEventA` | polygon(s) | `event_type` = Landslide | Cleaning only (rule R1); never a feature |

## 2. Terrain rasters (`data/interim/`, UTM 45N / EPSG:32645, 30 m)

| File | Meaning | Unit |
|---|---|---|
| `dem_raw.tif` | GLO-30 mosaic, reprojected (gaps = NaN, never 0) | m |
| `dem.tif` | Conditioned DEM (pits filled, depressions filled, flats resolved) | m |
| `acc_km2.tif` | Upstream drainage area (D8) | km² |
| `channel.tif` | All rivers with drainage ≥ 500 km² | 0/1 |
| `flood_path.tif` | Flood path: D8 path traced from Rasuwagadhi (−15 to +103 km) | 0/1 |
| `hand.tif` | Height above the flood path, along the flow route | m |
| `dist.tif` | Straight-line distance to the flood path | m |
| `along_km.tif` | Along-river distance (from the border) of the nearest flood-path cell | km |
| `hand_all.tif`, `dist_all.tif` | Same as `hand` / `dist` but relative to all rivers ≥ 500 km² (pilot v1; sensitivity only) | m |
| `slope.tif` | Terrain slope | degrees |
| `relief.tif` | Max − min elevation in a 17 × 17-cell (510 m) window | m |

## 3. Building table (`data/processed/buildings_labelled.csv` / `.gpkg`)
One row per candidate building (18,908 before cleaning).

| Column | Meaning | Feature? |
|---|---|---|
| `bid` | Overture id, or `ems_AOIxx_n` for EMS points without a footprint | id |
| `aoi`, `locality` | EMS area (AOI01 Syaprubesi, AOI02 Timure, AOI03 Betrawati–Bidur, AOI05 Phosretar) | grouping only |
| `kind` | `footprint` or `ems_only` | no |
| `source` | OpenStreetMap / Google Open Buildings / Microsoft ML Buildings / EMS only | no (sensitivity) |
| `grade` | Not affected / Possibly damaged / Damaged / Destroyed | label |
| `rank` | 0–3 version of `grade` | label |
| `affected` | 1 if any damage grade | target (a) |
| `destroyed` | 1 if Destroyed | target (b) |
| `in_event` | Centroid inside the EMS observed-event polygon | no (post-event; cleaning only) |
| `footprint_m2` | Footprint area | no, leaks the label (EMS-only rows have none) |
| `hand`, `dist`, `along_km` | See §2, sampled at the centroid | yes |
| `hand_all`, `dist_all` | See §2 | sensitivity only |
| `slope`, `relief` | See §2 | yes |
| `dem` | Elevation | no: identifies the reach |
| `dist_entry_km` | Straight-line distance from Rasuwagadhi | check on `along_km` only (correlation 0.994) |
| `lon`, `lat` | Centroid (WGS84) | no (identifies location); used for spatial blocks |

Derived in `clean.py`: `region` (Upper / Middle / Lower), `block` (0.01° cells for spatial CV), `valley_floor`, `hand_x_along`, `logd` = log(along_km + 10), `hand_x_logd`.

## 4. Cleaning rules (`clean.py`) and log (`results/cleaning_log.csv`)
| Step | Rows after | Removed |
|---|---|---|
| EMS points + pre-event footprints in analysed areas | 18,908 | – |
| R1 drop unlabelled footprints inside the flood trace | 17,626 | 1,282 |
| R2 drop unaffected footprints < 10 m² | 17,520 | 106 |
| R3 drop unaffected footprints < 8 m from an affected footprint of another source | 17,503 | 17 |

All 3,360 affected buildings are kept. Sensitivity options: `ambiguous="affected"` (relabel R1 rows as possibly damaged) and `negatives="osm"` (only OSM footprints as negatives).
