# Week 4 report — evaluation, uncertainty and interpretation (30 Sep 2026)

Plan reference: `RESEARCH_PLAN.md`, Week 4. Code: `pilot/20_final_eval.py`, `21_interpret.py`, `22_label_sensitivity.py`.

**Working assumption (unchanged).** *Affected* is the primary target. The label-audit review was resolved on 27 Oct 2026 (see §6.4 below; `pilot/results/audit/REVIEW.md`).

## 1. Final models and held-out performance with uncertainty (`20_final_eval.py`)
- **Models:** class-weighted logistic regression (C = 1, L2) and linear SVM (co-winners in Week 3), with random forest and gradient boosting for comparison.
- **Test:** leave-one-reach-out. 95% intervals come from a **block bootstrap** (1-km blocks of the held-out reach resampled 1,000 times).

| Target (features) | Model | Upper | Middle | Lower |
|---|---|---|---|---|
| Affected (main) | **Logistic regression** | **0.970** (0.895–0.998) | **0.983** (0.966–0.991) | **0.958** (0.943–0.971) |
| | Linear SVM | 0.970 (0.900–0.998) | 0.983 (0.965–0.991) | 0.956 (0.941–0.969) |
| | Random forest | 0.854 (0.685–0.970) | 0.959 (0.932–0.977) | 0.947 (0.925–0.964) |
| | Gradient boosting | 0.850 (0.668–0.971) | 0.968 (0.941–0.984) | 0.944 (0.919–0.964) |
| Affected (attenuation) | Random forest | 0.956 (0.822–0.993) | 0.982 (0.970–0.992) | 0.949 (0.926–0.965) |
| | Gradient boosting | 0.963 (0.871–0.995) | 0.988 (0.978–0.994) | 0.949 (0.929–0.963) |
| Destroyed (main) | Logistic regression | 0.952 (0.882–0.990) | 0.983 (0.971–0.992) | 0.958 (0.934–0.975) |
| Destroyed \| affected (attenuation) | Logistic regression | 0.930 (0.871–0.976) | 0.818 (0.781–0.853) | 0.750 (0.678–0.823) |

**Paired comparison** (the same bootstrap samples; positive = logistic regression better):

| Target | vs | Upper | Middle | Lower |
|---|---|---|---|---|
| Affected (main) | Random forest | **+0.116** (+0.023, +0.218) | **+0.024** (+0.010, +0.041) | +0.010 (−0.004, +0.032) |
| | Gradient boosting | **+0.120** (+0.023, +0.232) | **+0.015** (+0.005, +0.031) | +0.013 (−0.003, +0.037) |
| | Linear SVM | −0.001 | +0.000 | +0.002 |
| Destroyed (main) | Random forest | **+0.031** (+0.001, +0.055) | **+0.022** (+0.012, +0.036) | +0.006 (−0.002, +0.016) |
| Destroyed \| affected | Gradient boosting | **+0.176** (+0.086, +0.250) | +0.016 | +0.057 (−0.003, +0.109) |

- On the same features, logistic regression is **significantly better** than both tree ensembles in the upper and middle reaches (the intervals exclude 0). In the lower reach the difference is not significant.
- Logistic regression and linear SVM are indistinguishable.
- **The upper reach has wide intervals** (e.g. 0.90–1.00): it has few 1-km blocks and only 198 undamaged buildings.

_Sections 2–4 (interpretation, error analysis, land-cover check, label sensitivity) will be added when `21_interpret.py` and `22_label_sensitivity.py` have run._

## 5. Validation add-ons (2 Oct 2026)
Code: `pilot/23_ismote.py`, `24_knndm.py`, `25_aoa.py`, `26_calibration.py`; helpers in `pilot/spatial_tools.py`. All four are in `run_all.sh`.

### 5.1 ISMOTE vs SMOTE (`23_ismote.py`, `results/ismote.csv`)
- **Setup:** ISMOTE (Song & Liu 2024, *Neural Process. Lett.* 56:240) is re-implemented from Sect. 3 of the paper, since no code is published. K = 5, 20 iterations, main feature set, leave-one-reach-out, threshold 0.5, 3–5 seeds.
- **Logistic regression:** ISMOTE is worse than SMOTE and class weights on every metric.
  - Affected AUC 0.939 vs 0.970; balanced accuracy 0.856 vs 0.907; Brier 0.140 vs 0.081.
  - Destroyed AUC 0.950 vs 0.965.
- **Random forest:** ISMOTE is on par with SMOTE for affected (0.918 vs 0.922) and better for destroyed (0.941 vs 0.916), but still below class-weighted logistic regression (0.964).
- **Likely cause:** ISMOTE oversamples only the *misclassified* minority points. Here those are atypical or doubtful labels, such as damage high above the channel or Phosretar "destroyed". The paper's safe/noise rule (Sect. 3.3) reduces to "every minority error is safe", so nothing is filtered.
- **Decision:** class weights stay primary; SMOTE and ISMOTE are reported as comparisons.

### 5.2 kNNDM vs 1-km block CV (`24_knndm.py`, `results/knndm_*.csv`, `fig_knndm_ecdf.png`)
kNNDM (Linnenbrink et al. 2024, *GMD* 17:5897) is re-implemented in Python.

| Held-out reach | Median distance, held-out → training | Median CV distance: random / 1-km block / kNNDM | W: block → kNNDM |
|---|---|---|---|
| Upper | 23.4 km | 13 m / 282 m / 3.0 km | 26.8 → 23.5 km |
| Middle | 6.2 km | 13 m / 327 m / 1.8 km | 6.2 → 4.0 km |
| Lower | 10.1 km | 13 m / 254 m / 1.2 km | 8.3 → 6.9 km |

Optimism gap (inner AUC − held-out AUC), mean over reaches and targets:

| Model | Random | 1-km block | kNNDM |
|---|---|---|---|
| Logistic regression | 0.014 | 0.009 | −0.007 |
| Random forest | 0.061 | 0.044 | 0.033 |
| Gradient boosting | 0.056 | 0.038 | 0.026 |

- **kNNDM is the most honest inner CV** (mean |gap| 0.046 random → 0.034 block → 0.026 kNNDM).
- **It still cannot reach transfer distances.** With only two training reaches, CV folds cannot be 6–23 km apart.
- **No inner design ranks the models correctly.** Under kNNDM, RF (0.965) still beats logistic regression (0.960), while held-out reaches show the reverse (0.932 vs 0.967).
- **Decision:** leave-one-reach-out remains the main test. Use kNNDM for inner tuning, and report it as the inner-CV design.

### 5.3 Area of applicability (`25_aoa.py`, `results/aoa*.csv`, `fig_aoa_di.png`)
- **Attenuation set:** the upper reach is 100% outside the AOA, and the lower reach 65–67%. Along-river distance is extrapolated, which confirms the Week 3 extrapolation concern.
- **Main set:** 7–21% of the middle and lower reaches lie outside the AOA. The upper gorge lies 74–83% outside, because its terrain is unlike anything downstream.
- **Upper reach, outside the AOA:** RF 0.806 and GB 0.805 vs logistic regression 0.957. Inside the AOA: RF 0.956 vs logistic regression 0.996. **The tree models' upper-reach collapse is an extrapolation failure; linear models extrapolate gracefully.**
- In the middle and lower reaches, out-of-AOA buildings are easy cases (mostly high hillside, low prevalence), with AUC ≥ 0.93 for all models.

### 5.4 Calibration (`26_calibration.py`, `results/calibration.csv`, `recalibration.csv`, `fig_calibration.png`)
- **Ranking transfers; probabilities do not.** The calibration intercept follows flood strength:
  - Upper (77% affected) is *under*-predicted: intercept +2.3 to +4.1.
  - Lower (9% affected) is *over*-predicted: intercept −1.5 to −2.6.
  - Middle is close (−0.5 to +0.8).
  - This is the attenuation signal again: the training reaches set an average flood strength.
- Class weights and SMOTE raise every probability. That helps the upper reach and hurts the lower one. Averaged over reaches, Brier is 0.075 / 0.078 / 0.080 (weights / SMOTE / none).
- Tree models are worse calibrated (Brier 0.12–0.15). RF probabilities are too timid (slope 1.6); GB's are too extreme (0.57).
- **Intercept-only recalibration** (logistic regression, no correction) uses k random labelled buildings from the new reach. With k = 20, ECE drops 0.211 → 0.047 (upper), 0.036 → 0.024 (middle) and 0.074 → 0.028 (lower) for affected. Most of the gain comes by k = 20–50.
- **Implication for other floods:** terrain gives the ranking, and about 20–50 labelled buildings fix the threshold.

## 6. Verification pass (26 Oct 2026)
A full read-through of the plan, the weekly reports, all 27 scripts (three independent code reviews) and the paper, followed by a complete re-run of `pilot/run_all.sh` from the raw downloads (about 50 min).

### 6.1 What was wrong and is now fixed
| Item | Problem | Fix |
|---|---|---|
| `12_features.py` | Curvature sign was the opposite of the docstring: in the data, positive meant **concave / converging**. The paper read the plan-curvature coefficient as "hollows were hit more often". | Sign flip removed; > 0 = convex / diverging as documented. Coefficients change sign only (plan +0.31, profile −0.06); no AUC changes. Paper now reads: convex fan and spur surfaces were hit more often than hollows. |
| `27_damaged_share.py` | The recalibrated share averaged over all buildings, including the 50 whose labels fitted the intercept (`26_calibration.py` already excluded them). | Scored on the unseen buildings only. Reach shares unchanged to 3 decimals (0.778 / 0.201 / 0.092). |
| `21_interpret.py` | The WorldCover log had been produced by hand with an older script version; the Mantel–Haenszel odds ratios quoted in the paper for WorldCover (0.50, 2.1) had never been computed. | `LANDCOVER=worldcover` switch; `run_all.sh` runs it and writes `log_21_interpret_worldcover.txt`. Computed values: FN odds ratio 0.53 (p = 0.003), FP 2.24 (p < 10⁻¹⁷); same direction as NLCMS (0.22, 2.03). |
| `15_models.py` | saga logistic regression had no `random_state`. | Seeded. |
| Paper | Several method statements were looser than the code: R3 uses the *nearest* footprint only; TWI has +30 m / +0.001 offsets; the forest window is 5 × 5 pixels (≈ 75 m), not 60 m; the attenuation `c` is 5 km (affected) / 20 km (destroyed) and the slope is per doubling of d + c; the silhouette and Ward run on a 2,500 subsample; the final intervals use fixed hyperparameters; the optimism gap uses the best grid score (winner's curse). | All stated in the paper. |

### 6.2 Re-run: what changed
- Logistic regression, linear SVM, LDA, naive Bayes, RBF SVM, attenuation curve, label sensitivity, calibration of the linear models, recalibration, damaged share, error counts: **identical**.
- Tree models (random forest, gradient boosting, k-NN, decision tree) moved by up to 0.03 AUC in single cells (e.g. GB affected, upper reach 0.850 → 0.879; RF destroyed, upper 0.921 → 0.904), because the curvature sign change alters tie-breaking in tree splits. All conclusions are unchanged; the paper tables now carry the re-run values. §1 above keeps the pre-re-run figures.

### 6.3 Checked and found correct
- No leakage: scalers, SMOTE/ISMOTE, grid search and Youden thresholds all fit on the training reaches; feature sets contain no reach identifiers; `in_event` and `footprint_m2` are used only in cleaning.
- Every number in the paper traced to a log or CSV (reach descriptors, EDA medians, single-feature AUCs, PCA, feature selection, tuned-threshold recall/precision, attenuation fit, clusters, AOA, calibration, recalibration, damaged share, coefficients, permutation importance, tree rules, error counts, NLCMS odds ratios).
- Bibliography: all 30 keys cited and resolved; no undefined references; no overfull boxes.
- No new peer-reviewed building-damage or ML paper on the 2026 event was found (web search 26 Oct 2026).

### 6.4 Still open (author)
1. **Label-audit review.** `pilot/results/audit/label_audit.csv` has an empty `judgement` column; the paper carries a red `\pending{}` marker in §2.2 until it is filled (contact sheets in `pilot/results/audit/`). My own second look at the Phosretar sheet agrees with the first pass: about half of the "destroyed" buildings still stand in the post-event image.
2. Minor, not changed: `HistGradientBoosting` uses `early_stopping="auto"` (active only when the training set exceeds 10,000 rows); a sensitivity run dropping the 416 EMS-only positive rows was not made.

### 6.4 Label audit completed on HD chips (27 Oct 2026)

`29_hd_chips.py` re-rendered the 200 audit chips at 0.2 m/px (160 m + 80 m windows) and the
`judgement` column of `results/audit/label_audit.csv` was filled from them (`hd_review.csv` adds a
confidence flag; `28_audit_summary.py` prints the paper numbers). Result: not affected 62/62 intact;
destroyed 52/52 agree in Syaprubesi–Betrawati; Phosretar 8/18 judgeable destroyed still standing;
damaged/possibly 27/27 consistent (20 weak, intact roof); 41 unclear, of which 14 are "not affected" footprints
on trees/open ground. The author resolved the flagged rows on 27 Oct (record in `results/audit/REVIEW.md`); no pending markers remain in the paper.

### 6.5 Paper reformatted as an IEEE two-column journal article (27 Oct 2026)

`paper/paper.tex` now uses `IEEEtran` (journal mode, A4) with numeric citations through natbib and the
`IEEEtranN` bibliography style; the class and style were installed from CTAN into `~/texmf`. Times is
provided by the `texlive-times` package (installed 27 Oct). Courier and Helvetica metrics are not installed,
so the preamble substitutes Latin Modern for them (`texlive-courier texlive-helvetic` would give the real
fonts). The author block under the title carries the Softwarica College / Coventry University MSc Data
Science and Computational Intelligence affiliation and e-mail. The paper is 14 pages. Earlier versions are
kept as `paper_before_humanize_20261027.tex.bak` and `paper_before_ieee_20261027.tex.bak`.

### 6.6 Near-channel results promoted to the headline (29 Oct 2026)

Review note: the all-building AUCs (0.96–0.98) are inflated by hillside buildings that no flood could
reach, so the near-channel test is the fairer measure of the terrain signal. New `30_nearchannel_ci.py`
(in `run_all.sh`) scores the stored held-out predictions of the final fixed models on the 3,597 buildings
with HAND ≤ 30 m and distance ≤ 300 m, with 1-km block-bootstrap intervals and paired differences
(`results/nearchannel_ci.csv`, `nearchannel_paired.csv`, `log_30_nearchannel_ci.txt`).

| Target | Model | Middle (n = 2,308) | Lower (n = 1,084) |
|---|---|---|---|
| Affected | Logistic regression | 0.851 (0.765–0.920) | 0.801 (0.689–0.889) |
| | Random forest | 0.860 (0.781–0.921) | 0.756 (0.672–0.834) |
| | Gradient boosting | 0.794 (0.680–0.885) | 0.742 (0.652–0.833) |
| Destroyed | Logistic regression | 0.859 (0.806–0.911) | 0.781 (0.660–0.895) |

- The upper reach cannot be scored: all 205 near-channel buildings were affected.
- Only one paired difference excludes zero (LR vs GB, affected, middle: +0.056). Near the channel the
  linear and tree models are statistically indistinguishable; the large gap in the all-building table
  comes from the upper-gorge hillside buildings outside the trees' AOA.
- Two earlier near-channel figures were not the same quantity: Table III's 0.818 is the tuned-grid model
  scored on the subset (0.851 / 0.784); Table IV's 0.856 is a model *trained within* the subset
  (0.889 / 0.824). The paper now states which is which. The Discussion's "0.82–0.84" range, which matched
  neither, is replaced with 0.80–0.85.
- Paper changes: abstract, Table I row, contributions, near-channel paragraph in Methods, new §IV-C
  "Near the channel" with Table VIII, Discussion and Limitations numbers, Conclusions, pipeline range 01–30.

**Baselines added (29 Oct 2026).** `30_nearchannel_ci.py` now also scores two training-free baselines
on the same subset and bootstrap samples.

| Target | HAND only (ranking) | HAND cut, chosen near-channel (15–23 m) | HAND cut, chosen on all buildings (31–55 m) |
|---|---|---|---|
| Affected, middle / lower | 0.876 / 0.770 | 0.652 / 0.644 | 0.500 / 0.500 |
| Destroyed, middle / lower | 0.854 / 0.827 | 0.647 / 0.683 | 0.500 / 0.500 |

- **HAND alone ranks as well as the ten-feature logistic regression** near the channel (paired
  differences −0.045 to +0.031, every interval includes zero). The other eight features add no held-out
  ranking skill once HAND is known. The within-subset gain from local terrain in Table IV (0.832 → 0.862)
  does not reproduce when the model is trained on all buildings.
- **A single HAND cut is far worse** (balanced accuracy 0.64–0.68; LR better by 0.10–0.21, three of four
  intervals exclude zero). The valley-wide cut is above the 30 m subset ceiling and flags everything.
- Reading: the model's contribution near the channel is the threshold and the probability, not the
  ordering. Paper: Table VIII rows, new paragraph in §IV-C, abstract, Discussion (two places), Conclusions.

### 6.7 Block-level bootstrap on the attenuation slope (29 Oct 2026)

`17_attenuation_curve.py` now bootstraps the slope three ways with the form and c fixed
(`results/attenuation_boot.csv`); the headline interval resamples 1-km blocks.

| Target | Slope (m per doubling) | 1-km blocks | Buildings (old) | Bins |
|---|---|---|---|---|
| Affected (c = 5 km) | −19.5 | −21.9 to −16.8 | −21.1 to −17.7 | −21.3 to −14.9 |
| Destroyed (c = 20 km) | −23.0 | −28.8 to −19.9 | −25.5 to −21.5 | −30.2 to −20.8 |

- Block intervals are 1.5× (affected) and 2.4× (destroyed) wider than the building-level ones the
  paper used before. The destroyed slope widens most because its c = 20 km form leans on the two
  upstream bins.
- The slope stays far from zero under every scheme; conclusions unchanged. Paper: Methods, §IV-F with
  new Table (attboot), Discussion sentence replaced, Limitations. Figure legend now says "block-bootstrap CI".
- Runtime of step 17 rose from about 1.5 to about 4.5 min (three bootstraps of 300).

### 6.8 Supervisor review, batch 1 (30 Oct 2026)

Correctness points checked against `results/*.csv` and fixed in `paper/paper.tex` (backup `paper/paper_before_review1_20261030.tex.bak`):
abstract cut from 390 to 274 words and the threshold contradiction removed; audit arithmetic now reconciles to 41 (18/10/13);
two false model-comparison claims (Table X "never better", "only comparison in which a tree ensemble leads") rewritten;
near-channel wording ("about a fifth" of ranking error, intervals "five to seven times" wider, 3,392 scorable of 3,597);
attenuation: c-sensitivity stated per target, 2.4 -> 2.2, building-level intervals "just separate", fitted vs observed
(19.6/15.3 m) distinguished; reach geometry (median 6-23 km to nearest training building; Copernicus gap; 42-65 / 65-91 km with
0.4-km overlap footnote); Table VII intervals added for destroyed and destroyed|affected logistic regression; Table VIII caption;
Table XII intercept column (per reach, mean over targets); Table IX gap recomputed for affected only (LR 0.006, RF 0.055,
GB 0.087) and the text updated; Table X RF configuration flagged in the text; Currie & Reinke citations removed where
misattributed (85% accuracy, fixed-model design, 7.7%/27.0% example replaced by their macro-F1 range);
Zevenbergen & Thorne (1987), scikit-learn, imbalanced-learn and pysheds added to `refs.bib` (stray brace fixed).
Completeness: `pilot/requirements.txt` (pip freeze), `run_all.sh` runs 09/11/29 before 28 under `--audit` and copies all
paper figures; `09_label_audit.py` now preserves existing judgements; cleaning-log table (Table IV) and confusion-count table
added; licence column in the data table; availability statement names the deposit still to be made.
Style: three threshold terms defined once (damaging height / decision threshold / probability level); "affected share";
hedged "to our knowledge"; shorter title and running head; `hidelinks` and PDF metadata; `\pending` removed.
Not done (defended): public DOI (author action; done 30 Oct 2026 as release v1.0.1, DOI 10.5281/zenodo.23802509), 10% body trim (body is about 6,400 words by a crude count excluding floats),
Table I limitations column kept. Still 16 pages, 0 overfull.


### 6.9 Supervisor review, batch 2 (30 Oct 2026)

Fixed in `paper/paper.tex` (backup `paper/paper_before_review2_20261030.tex.bak`): Currie & Reinke quoted as median macro-F1 0.68 to 0.54 with the 0.90/0.41 extremes named as single events; the attenuation offset c described as a fitted offset (5 km affected, 20 km destroyed), slopes per doubling of d + c, and the fixed 10 km feature offset distinguished; label arithmetic added (3,400 matched points on 2,944 footprints, 456 duplicates collapsed, plus 416 point-only rows = 3,360); near-channel interval floor 0.66; "related conditioning factors" instead of "the same"; ISMOTE exception in the imbalance paragraph and a tuned-threshold row in the imbalance table; "never at risk" and "could actually reach" replaced; calibration ranges marked as affected-only; linear SVM named beside logistic regression in the abstract, conclusions and Table VIII; label-sensitivity sentence restated with two-reach means and the table footnote extended; cleaning-log first row reworded; "Optimism gap" column name; attenuation captions without "from the source"; nine other features; HAND defined by the drain-to cell in the introduction; balanced-accuracy comparison labelled as not like for like; destroyed paired difference exception named; Table I near-channel cell with the HAND criterion; valley floor defined by HAND; 100 repeats of the share analysis stated in Methods; Figure 2 caption direction corrected. Abstract now under 250 words. The PDF is 17 pages (a few reference lines spill onto the last page).
