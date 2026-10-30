# Week 1 report: data audit and cleaning (28 Sep 2026)

Plan reference: `RESEARCH_PLAN.md`, Week 1. Code: `pilot/` (scripts 01–10, `clean.py`, `run_all.sh`).

## Summary
| Task | Status | Result |
|---|---|---|
| Re-run pipeline from scratch | Done | `run_all.sh --clean` rebuilds everything from `data/raw/` in about 4.5 min; results reproduce |
| Single cleaning module + cleaning log | Done | `clean.py`, `results/cleaning_log.csv` (rules R1–R3) |
| Footprint audit | Done | `08_audit_footprints.py`; see §2 |
| Label audit, automatic checks | Done | EMS inventory totals and imagery sources; see §3 |
| Label audit, visual (image chips) | Done, first pass | 200 buildings; see §3.4. Your review is needed |
| DEM comparison (FABDEM) | Done | Barely matters; see §4 |
| Channel threshold sensitivity | Done | Makes no difference; see §4 |
| RQ4 labels for July 2025 | None found | No building-level damage product (UNOSAT / EMS / ICIMOD) found for 8 July 2025 |
| Data dictionary | Done | `pilot/data/DATA_DICTIONARY.md` |

## 1. Cleaning log (`results/cleaning_log.csv`)
| Step | Rows after | Removed |
|---|---|---|
| EMS points + pre-event footprints in analysed areas | 18,908 | – |
| R1 drop unlabelled footprints inside the flood trace | 17,626 | 1,282 |
| R2 drop unaffected footprints < 10 m² | 17,520 | 106 |
| R3 drop unaffected twins < 8 m from an affected footprint of another source | 17,503 | 17 |

- All 3,360 affected buildings are kept.
- R2 applies to unaffected buildings only. An EMS grade on a small footprint is still a real graded building. The error R2 guards against (ungraded sheds counted as undamaged) only affects the negative class.
- Effect on results: R2 and R3 change pilot AUCs by at most 0.017 (mean 0.001).

## 2. Footprint audit (`results/log_08_audit_footprints.txt`)
- Duplicates.
  - Only 1 pair of footprints overlaps by ≥ 50%.
  - 1,397 footprints have a different-source footprint within 8 m. Almost all are separate neighbouring houses or offset machine-learning footprints.
  - Of those, 59 are "undamaged twins" of a damaged building. 42 were already removed by R1; the remaining 17 are removed by R3.
- Size. OSM contains 144 footprints < 10 m². Google and Microsoft have none (their minimum is about 10 m²).
- Sources in the lower reach (Phosretar). Negatives there are 2,604 OSM, 1,992 Google and 1,095 Microsoft. EMS's own inventory lists about 2,000 fewer buildings in this area than we have footprints (§3.1). The extra buildings are mostly machine-detected footprints far from the river. The sensitivity run with OSM-only negatives tests whether they matter.

## 3. Label audit

### 3.1 EMS inventory vs our table
The public EMS point layer lists affected buildings only. But the EMS summary tables inside each product also count buildings graded "No visible damage".

| Area | EMS affected | EMS no visible damage | EMS total | Our footprints | Our negatives (clean) |
|---|---|---|---|---|---|
| AOI01 Syaprubesi | 433 | 126 | 559 | 556 | 176 |
| AOI02 Timure | 431 | 10 | 441 | 626 | 22 |
| AOI03 Betrawati–Bidur | 2,343 | 9,064 | 11,407 | 10,818 | 8,315 |
| AOI05 Phosretar | 609 | 3,893 | 4,502 | 6,492 | 5,636 |

- The totals agree closely in Syaprubesi (556 vs 559) and Betrawati–Bidur (10,818 vs 11,407). This supports treating ungraded footprints as undamaged.
- Timure: EMS confirms that almost every building (431 of 441) was affected.
- Phosretar: we have about 2,000 more footprints than EMS inventoried; see §2.
- Open item. The individual "No visible damage" points are not in the public product. The tables mention a "Base Layer Package", which the public API did not list. If obtainable, it would give EMS-confirmed negatives.

### 3.2 Plausibility of damage labels (`08_audit_footprints.py` §4)
| Area | Destroyed | HAND median | HAND 90th pct | Inside mapped flood trace |
|---|---|---|---|---|
| Syaprubesi | 239 | 28.8 m | 46.3 m | 100% |
| Timure | 292 | 41.6 m | 55.2 m | 100% |
| Betrawati–Bidur | 1,699 | 17.9 m | 25.8 m | 99.2% |
| Phosretar | 275 | 14.2 m | 22.3 m | 97.5% |

- Bidur's large "destroyed" count is internally consistent. It lies inside the mapped flood trace and low above the river, and it matches EMS's own table (1,817 residential destroyed).
- But Bidur was graded from coarser imagery. EMS used ALOS at 0.8–1.0 m there, against 0.3–0.5 m elsewhere. So its grades are the least certain, which is why the visual audit prioritises it.
- 26 destroyed buildings sit > 60 m above the flood path, mostly in Timure. They are listed in `results/audit_destroyed_high_hand.csv` for visual checking.

### 3.3 Imagery for the visual audit
- Vantor (formerly Maxar) Open Data released pre- and post-event imagery for this event on OpenAerialMap (CC BY-NC 4.0).
  - Post-event: 27 Aug – 12 Sep 2026, 0.35–0.7 m, covering all four areas.
  - Pre-event: 2021, 2023 and Feb 2026.
- `09_label_audit.py` draws a stratified sample per area (20 destroyed, 10 damaged/possibly, 20 not affected; seed 42). For each building it tries every covering image, keeps the clearest one (highest texture, since clouds and haze are smooth), and renders pre/post chips with the footprint outlined. It also writes contact sheets (`results/audit/sheet_AOI0x.png`) and `results/audit/label_audit.csv`.

### 3.4 First-pass visual audit (`11_audit_first_pass.py`, `results/audit/first_pass_summary.csv`)
These judgements are a first pass, made by eye from 80 m chips. They are provisional: review the contact sheets and fill in the `judgement` column of `label_audit.csv` yourself.

| Area | Destroyed: agree / disagree / unclear | Damaged or possibly: agree+plausible / unclear | Not affected: agree / unclear |
|---|---|---|---|
| Syaprubesi | 16 / 0 / 4 | 7 / 3 | 20 / 0 |
| Timure | 18 / 0 / 2 | 8 / 2 | 19 / 1 |
| Betrawati–Bidur | 18 / 0 / 2 | 5 / 5 | 20 / 0 |
| **Phosretar** | **7 / 7 / 6** | 3 / 7 | 17 / 3 |

- Undamaged labels hold everywhere. All 76 judgeable "not affected" buildings look intact. This supports inferring negatives from ungraded footprints.
- "Destroyed" holds in the upper and middle reaches: 52 of 52 judgeable agree. That includes Bidur, despite its coarser EMS imagery.
- "Destroyed" is doubtful in Phosretar.
  - In 7 of 14 judgeable cases the structure still appears to stand in the debris, so "damaged" might be more accurate.
  - Possible reasons: EMS may have graded from roof loss visible at 0.3 m that the chips don't show, used an earlier image, or over-called.
  - This matters for the *destroyed* target in the lower reach. It does not matter for *affected*: a standing but debris-hit building is still affected.
- "Damaged / possibly damaged" often can't be judged at chip scale (about 40% unclear). This is expected, and it is one more reason to treat "possibly damaged" as the noisiest class.
- Two non-house footprints were found in Bidur (a greenhouse or solar array), graded "possibly damaged" or "damaged".

## 4. Sensitivity analyses (`10_sensitivity.py`, `results/sensitivity.csv`)
Logistic regression (4 terrain features), leave-one-region-out, mean ROC-AUC over the three test reaches:

| Variant | Affected | Destroyed |
|---|---|---|
| **Main** (flood path, GLO-30, rules R1–R3) | **0.963** | **0.947** |
| All rivers ≥ 200 / 500 / 1,000 km² as channel | 0.963 / 0.963 / 0.963 | 0.949 / 0.949 / 0.947 |
| FABDEM (bare earth), flood path | 0.961 | 0.946 |
| Ambiguous rows relabelled "possibly damaged" | 0.964 | **0.842** (upper reach 0.65) |
| Only OpenStreetMap footprints as negatives | 0.964 | 0.936 |

- Channel definition: no effect.
- DEM: FABDEM gives HAND about 1.5 m lower on average (mean absolute difference 2.7 m; 16% of buildings differ by > 5 m). Single-feature AUC is 0.937 vs 0.933, and model AUCs are within 0.002. Decision: keep GLO-30 as the main DEM (CC-BY-style licence, identical results) and report FABDEM as a robustness check.
- Extra machine-learning footprints in Phosretar: no effect on *affected*; small effect on *destroyed*.
- Ambiguous rows: *affected* does not change, but *destroyed* is sensitive to how they are handled.

## 5. RQ4 (July 2025 event)
- No building-level damage product was found for the 8 July 2025 Rasuwagadhi GLOF: no EMS activation, no UNOSAT product, and no ICIMOD building dataset.
- Option: Vantor/NEA imagery from September 2025 exists (NEA drone mosaics of Rasuwagadhi–Timure and Simle–Betrawati at 4–6 cm). Labels would have to be digitised by hand.
- Recommendation: drop RQ4 as planned, unless time remains in Week 4.

## 6. Decisions made this week
1. The cleaning rules R1–R3 are fixed in `clean.py`, and every script uses them.
2. Buildings from all footprint sources stay in the main analysis. OSM-only negatives are a sensitivity run.
3. The Bidur labels are kept: the visual audit supports them despite the coarser EMS imagery.
4. GLO-30 stays the main DEM; FABDEM is a robustness check.

## 7. Recommendations for Week 2 (your call)
1. Make *affected vs not affected* the primary target. It holds up under every data choice tested and under the Phosretar doubts. Keep *destroyed* as a secondary target and state its two sensitivities: the ambiguous-row rule, and Phosretar grades that may be overcalled.
2. Review the first-pass audit. Especially the 7 Phosretar "disagree" cases and the "unclear" ones; open the chips in `results/audit/chips/`. If you confirm the Phosretar doubts, add a sensitivity run with Phosretar "destroyed" recoded as "damaged".
3. Proceed with the Week 2 feature engineering (valley width, bend position, curvature, TWI, stream power).
