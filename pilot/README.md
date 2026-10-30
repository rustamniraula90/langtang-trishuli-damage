# Pilot: does valley geography predict building damage in the 26 Aug 2026 Trishuli cascade?

First run 2026-09-24; flood-path channel added 2026-09-27. Everything below is reproducible from the scripts.

> Superseded numbers (30 Oct 2026). The pilot notes below are kept as a record. The paper (`../paper/paper.tex`) uses the final cleaned table of 17,503 buildings (R1-R3 in `cleaning_log.csv`) and the final attenuation fit of `17_attenuation_curve.py`: HAND50 falls by 19.5 m (affected) and 23.0 m (destroyed) per doubling of distance, from about 87/63 m near the border to about 16/11 m at 90 km (fitted; last observed bin 19.6/15.3 m). Where a number below differs, the paper and `results/*.csv` are authoritative.

## Run order

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The environment is pinned (Python 3.12.16). Steps, in order:

- `.venv/bin/python 01_download.py`: EMS labels, Copernicus GLO-30 DEM (4 tiles), Overture buildings (2026-08-19), OSM rivers
- `.venv/bin/python 02_terrain.py`: DEM conditioning, flow routing, flood-path tracing, HAND / distance / along-river distance
- `.venv/bin/python 03_build_dataset.py`: label join (EMS points -> footprints), feature sampling
- `.venv/bin/python 04_model.py`: cleaning, 9 model variants, leave-region-out + spatial-block CV, figures
- `.venv/bin/python 05_map.py`: study-area map (with traced flood path) for the proposal
- `.venv/bin/python 06_hard_checks.py`: near-channel subset; destroyed vs damaged among affected
- `.venv/bin/python 07_log_attenuation.py`: linear vs log vs normalised-HAND attenuation; attenuation-curve figure
- Later steps: see STEPS in run_all.sh (needs imbalanced-learn). Week 4 add-ons (2026-10-02):
- `.venv/bin/python 23_ismote.py`: ISMOTE (Song & Liu 2024) vs SMOTE / class weights / none
- `.venv/bin/python 24_knndm.py`: kNNDM vs 1-km block vs random inner CV: optimism gap (helpers in spatial_tools.py)
- `.venv/bin/python 25_aoa.py`: area of applicability; AUC inside vs outside (needs 20_final_eval output)
- `.venv/bin/python 26_calibration.py`: held-out calibration + intercept-only recalibration with k labels
- `.venv/bin/python 27_damaged_share.py`: observed vs predicted share of affected buildings per reach / 3-km bin
- `.venv/bin/python 28_audit_summary.py`: tabulates the researcher's judgement column of results/audit/label_audit.csv and prints the paper's audit numbers
- `.venv/bin/python 30_nearchannel_ci.py`: near-channel (HAND <= 30 m, dist <= 300 m) AUC with block-bootstrap CIs from the stored held-out predictions; the paper's headline test
- `.venv/bin/python 29_hd_chips.py`: HD (0.2 m/px) 160 m + 80 m pre/post chips of the same audit sample -> results/audit/chips_hd/ (internet, ~30 min; run_all.sh --audit)

Or simply `bash run_all.sh`: steps 01-30 in order (+ the WorldCover variant of 21); ~1 h. Add --audit to rebuild the audit chips (09, 11, 29; needs internet) before 28; the recorded judgements are kept.

## Data (all open)
| Input | Source | Note |
|---|---|---|
| Damage labels | Copernicus EMS **EMSR927** grading, AOIs 01, 02, 03, 05 | 3,816 photo-interpreted building points |
| Buildings (pre-event) | Overture Maps **2026-08-19** (OSM 233k + Google 128k + Microsoft 46k in bbox) | 18,492 footprints inside analysed AOIs |
| Terrain | Copernicus GLO-30 DEM, 4 tiles, reprojected to UTM 45N | This is a **DSM**: it includes trees and roofs |
| Channel | **Flood path** traced on the DEM (see below) | OSM river download was truncated at 2,000 ways, so it is not used |

## The flood-path channel (02_terrain.py)
1. Snap the border point. Rasuwagadhi (85.378 E, 28.276 N) is snapped to the nearest river draining ≥ 500 km². It lands on the Bhote Koshi, at 3,044 km² drainage.
2. Trace downstream along the D8 flow directions to the edge of the DEM.
3. Trace 15 km upstream along the main stem, always stepping to the inflowing cell with the largest accumulation.
4. Result: 3,353 cells, from 15 km upstream to 103 km downstream of the border. It follows the Bhote Koshi–Trishuli through all four areas and excludes tributaries such as the Tadi Khola (see `../proposal/figures/study_area.png`).
5. Features measured to the path:
   - `hand` (via `compute_hand`)
   - `dist` (Euclidean distance)
   - `along_km`: the along-river distance, from the border, of the building's nearest path cell
6. The pilot-v1 definition (all rivers ≥ 500 km²) is kept as `hand_all` / `dist_all` for comparison.

Effect of the fix: HAND changes by more than 2 m for only 0.6% of buildings, mostly at Syaprubesi (7%), where tributaries join. Single-feature AUCs:

| Feature | Flood path | All channels ≥ 500 km² (v1) |
|---|---|---|
| HAND | 0.932 | 0.934 |
| Distance | 0.914 | 0.921 |

The physically correct definition changes little here.

Along-river distances by area:

| Area | Along-river distance from the border | Straight-line distance |
|---|---|---|
| Timure | −0.3–4.5 km | |
| Syaprubesi | 11.6–15.9 km | |
| Betrawati–Bidur | 41.5–65.7 km | 35–54 km |
| Phosretar | 65.3–91.0 km | 53–67 km |

Along-river and straight-line distance correlate at 0.994.

## Log-distance attenuation (07_log_attenuation.py, 2026-09-27)
Logistic regression, leave-one-region-out. The table shows mean ROC-AUC over the test reaches (full tables in `results/log_attenuation_log.txt`):

| Task | base | linear | log (c = 10 km) | normalised HAND (c = 10 km) |
|---|---|---|---|---|
| Affected (all) | 0.961 | 0.963 | **0.967** | 0.946 |
| Destroyed (all) | 0.947 | 0.968 | **0.971** | 0.953 |
| Affected, near-channel (middle, lower) | **0.856** | 0.778 | 0.810 | 0.751 |
| Destroyed vs damaged | 0.754 | 0.827 | 0.834 | **0.839** |

- Log beats linear on all four tasks at c = 10 km.
- c (unknown collapse-to-border distance) matters little: at most about 0.02 AUC between 5 and 40 km.
- Near-channel, lower reach: any attenuation term still hurts (0.83 base, 0.66 linear, 0.72 log). In that subset along-river distance mostly separates reaches with very different damage shares (upper near-channel buildings are almost all damaged).
- Normalised HAND (HAND / HAND₅₀(d), with the curve fitted on training reaches only) is unstable. With few bins the fitted curve can even slope the wrong way (near-channel, middle fold).
- Caveat: c was compared on the test reaches here. In the full study, choose the form and c by nested CV.

Attenuation curve (`results/fig_attenuation_curve.png`; all data, descriptive):
- HAND₅₀ per 5-km bin fits a + b·log(d + 10 km) with R² = 0.96 (affected) and 0.91 (destroyed).
- HAND₅₀ falls by about 26 m (affected) and 20 m (destroyed) per doubling of along-river distance: from about 65–95 m near the border to 10–20 m at 65–90 km.
- There are no mapped buildings between about 16 and 42 km, and the upper bins have few undamaged buildings (wide intervals).

## Cleaning decisions (and why)
1. Matching: 3,400 of 3,816 EMS points (89%) matched a footprint within 10 m. The 416 unmatched points are kept as point-only rows. Footprint area is not used as a feature, because unmatched rows are all damaged and would leak the label.
2. Ambiguous negatives dropped (1,282 rows):
   - These are unlabelled footprints *inside* the EMS observed-event polygon.
   - They are small (median ~30 m² vs ~60 m² for graded buildings) and close to graded points, so they are mostly sheds or split footprints.
   - Counting them as "not affected" made Timure's undamaged buildings look *closer* to the river than its destroyed ones, a clear label artefact.
3. The `in_event` flag is post-event information, so it is used only for cleaning and never as a feature.
4. AOI01 and AOI02 are merged as "Upper". Each alone has too few undamaged buildings (Timure has 22 after cleaning).

Final table: 17,626 buildings:

| Region | Not affected | Possibly damaged | Damaged | Destroyed |
|---|---|---|---|---|
| Upper (Syaprubesi + Timure) | 198 | 83 | 39 | 531 |
| Middle (Betrawati–Bidur) | 8,377 | 254 | 188 | 1,699 |
| Lower (Phosretar) | 5,691 | 236 | 55 | 275 |

## Results (flood-path features)

Affected vs not, ROC-AUC, leave-one-region-out (train on two regions, test on the third):

| Model | Upper | Middle | Lower | Spatial-block CV |
|---|---|---|---|---|
| LogReg, HAND + dist only | 0.959 | 0.980 | 0.917 | 0.954 |
| LogReg, 4 features | 0.958 | 0.979 | 0.945 | 0.965 |
| SVM (RBF) | 0.866 | 0.979 | 0.952 | 0.964 |
| Random forest | 0.810 | 0.979 | 0.935 | 0.966 |
| Naive Bayes | 0.964 | 0.962 | 0.892 | 0.949 |
| k-NN (k = 25) | 0.821 | 0.955 | 0.930 | 0.950 |
| Decision tree (depth 3) | 0.728 | 0.924 | 0.853 | 0.941 |
| **LogReg + attenuation** | **0.962** | **0.988** | 0.940 | 0.981 |
| **LogReg + log attenuation** (c = 10 km) | **0.967** | **0.989** | 0.945 | 0.982 |
| **RF + attenuation** | 0.856 | 0.988 | 0.947 | 0.985 |

"+ attenuation" = the 4 features + `along_km` + `hand × along_km`; "log" uses `log(along_km + 10)` instead of `along_km`.

Destroyed vs not, LogReg:

| Features | Upper | Middle | Lower |
|---|---|---|---|
| 4 features | 0.930 | 0.979 | 0.931 |
| + attenuation | 0.950 | 0.989 | 0.965 |

Full table: `results/pilot_scores.csv`.

Harder checks (`results/pilot_hard_checks.txt`):

| Task | LogReg | RF | LogReg + attenuation |
|---|---|---|---|
| Near-channel only (HAND ≤ 30 m, dist ≤ 300 m; 3,622 buildings) | Middle 0.885, Lower 0.826 | Middle 0.846, Lower 0.748 | Middle 0.894, **Lower 0.662** |
| Destroyed vs damaged, among hit buildings (3,360) | Upper 0.868, Middle 0.818, **Lower 0.575** | Upper 0.695, Middle 0.732, Lower 0.589 | Upper 0.901, Middle 0.812, **Lower 0.766** |

## What the pilot shows
1. Geography carries most of the signal. Two numbers per building (height above and distance to the flood path) separate hit from un-hit buildings with AUC > 0.9, even on a region the model never saw.
2. The critical height falls downstream (`results/fig_destroyed_rate_by_hand.png`). Destruction drops off at these heights:

   | Region (along-river km from border) | Destruction drops off at |
   |---|---|
   | Upper (0–16 km) | HAND ≈ 40–60 m |
   | Middle (42–66 km) | HAND ≈ 20–30 m |
   | Lower (65–91 km) | HAND ≈ 10–20 m |

3. Attenuation features: mostly helpful, sometimes harmful (first look at RQ2).
   - They help LogReg for *destroyed* buildings in every reach, and destroyed-vs-damaged in the lower reach (0.58 → 0.77).
   - They slightly lower affected-vs-not in the lower reach (0.945 → 0.940), and sharply lower near-channel lower-reach performance (0.83 → 0.66).
   - The reason is extrapolation: each held-out reach lies outside the training range of `along_km`, and a linear term in km extrapolates poorly.
   - Trees cannot extrapolate at all, so RF + attenuation still fails on the upper reach (0.856).
   - Log(distance) works better than linear; see the section above.
4. Simple beats complex on transfer. Logistic regression generalises across regions better than RF, k-NN or a depth-3 tree. That is a good bias–variance discussion point.
5. Headroom is in the hard tasks. New terrain features (valley width, bend position, curvature, TWI, channel slope, stream power) should be judged mainly there.

The study is terrain-only by design (decision 2026-09-25): no building attributes, material or land cover.

## Pipeline fixes
- 2026-10-26, verification pass (code review of all scripts; full re-run of `run_all.sh`).
  - `12_features.py`: the curvature sign was the opposite of its docstring (positive meant concave /
    converging). The flip is removed, so now > 0 = convex / diverging as documented. Only the signs of
    `curv_profile` / `curv_plan` change; every AUC is unchanged.
  - `27_damaged_share.py`: the recalibrated share is now scored only on buildings not used to fit the
    intercept (as `26_calibration.py` already did).
  - `21_interpret.py`: `LANDCOVER=worldcover` forces the ESA WorldCover stand-in; `run_all.sh` runs it
    after the NLCMS pass and writes `results/log_21_interpret_worldcover.txt`. Earlier the WorldCover
    log had been produced by hand with an older script version that lacked the Mantel-Haenszel test.
  - `15_models.py`: `random_state` added to the saga logistic regression (bit-reproducible runs).
- 2026-09-27, missing DEM tile.
  - The first run lacked DEM tile N28 E084, and the mosaic zero-filled the gap.
  - The tile is now downloaded, and `merge(..., nodata=nan)` never zero-fills.
  - Results were unchanged to three decimals.
- 2026-09-27, flood-path channel. Replaces the all-channels definition; see above.

## Known limitations / to fix in the full study
- The DEM is a 30 m DSM, so HAND is coarse. Distances are quantised to 30 m and some buildings sample HAND = 0 on channel cells. Try ALOS 12.5 m or FABDEM (a bare-earth DEM) and compare.
- The upstream trace follows the largest-accumulation inflow for 15 km. Check on imagery that it stays on the Bhote Koshi / Kerung main stem.
- Labels are photo-interpreted; "possibly damaged" is noisy. Report both binary targets.
- AOI03 (Bidur, 1,699 destroyed) needs a visual spot-check against the EMS PDF map and Sentinel-2.
- The unlabelled-in-event rule removes 1,282 rows. Report sensitivity (e.g., relabel them as affected instead).
