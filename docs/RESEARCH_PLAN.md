# Research Plan: building-damage prediction for the 2026 Trishuli cascade

*Individual assignment · six weeks, starting Mon 28 Sep 2026 · the pilot is done (`pilot/`).*

**Status 29 Oct 2026:** Weeks 1-4 complete (`WEEK1_REPORT.md`-`WEEK4_REPORT.md`); the paper draft is in `paper/paper.tex`. The label audit is done, and the near-channel test is now the headline result (`WEEK4_REPORT.md` §6.6, `pilot/30_nearchannel_ci.py`).

The literature and research questions are in `TOPIC_B_LITERATURE_REVIEW.md`. Pilot code and results are in `pilot/`.

---

## Where we start (pilot, 24 Sep)
- Data table: 17,626 buildings in three river reaches.
- Labels: Copernicus EMSR927 damage grades.
- Buildings: pre-event Overture footprints.
- Features: HAND, distance to channel, slope and relief from the GLO-30 DEM.
- Result: Logistic Regression on HAND + distance (to the traced flood path) reaches AUC 0.92–0.98 on an unseen reach.
- Where the headroom is:
  - near-channel buildings: AUC 0.76–0.88
  - destroyed vs damaged: AUC 0.58–0.87 (0.77–0.90 with attenuation features)
- Key pattern: the critical HAND falls downstream (~50 m → ~15 m).

---

## Week-by-week

### Week 1 (28 Sep – 4 Oct): Data audit and cleaning
Goal: a clean, documented building table you can defend.

- [x] Re-run the pilot scripts from scratch with a single runner (`pilot/run_all.sh`).
- [x] Label audit (first pass done; your review pending, see `WEEK1_REPORT.md` §3.4):
  - Open the EMS PDF maps for each area and spot-check about 50 buildings per area against Sentinel-2 or the EMS imagery.
  - Priority: AOI03 Bidur (1,699 destroyed).
- [x] Footprint audit:
  - Measure duplicates or offsets between OSM, Google and Microsoft.
  - Decide the minimum footprint size, e.g. drop <10 m² sheds.
  - Record the match rate for each source.
- [x] DEM comparison: download FABDEM (bare earth; ALOS skipped, needs a login) and ALOS 12.5 m, recompute HAND, and compare HAND at graded buildings.
- [x] Channel threshold sensitivity: 200 / 500 / 1,000 km².
- [x] RQ4 label search: (nothing found; RQ4 dropped unless time remains) UNOSAT / EMS / ICIMOD building damage for the 8 Jul 2025 Rasuwagadhi GLOF.
- [x] Write `data/DATA_DICTIONARY.md`: every column, its source, units and cleaning rule.

**Deliverables:** cleaned `buildings.gpkg` and a cleaning log with counts dropped per rule. The cleaning log goes into the Methods section.

### Week 2 (5 – 11 Oct): Feature engineering and exploratory analysis
Goal: turn "where is the building in the valley" into numbers.

Terrain only. Every feature is computed from the DEM and the derived channel network. No building attributes, construction material or land cover are used; building footprints are only label locations.

| Group | Feature | How | Why |
|---|---|---|---|
| Core (pilot) | HAND | Height above the main channel along the flow path | Vertical exposure |
| Core (pilot) | Distance to channel | Euclidean distance to the main-channel cells | Horizontal exposure |
| Local terrain | Slope, local relief (pilot) | DEM gradient; max − min in a ~500 m window | Hillside vs flat terrace |
| Local terrain | Profile / plan curvature, TWI | DEM derivatives | Hollows and fans collect flow |
| Valley / channel geometry | Valley width at the nearest channel cell | Cross-section width at HAND = 20 / 40 m | Confined gorges → deeper, faster flow |
| Valley / channel geometry | Channel bend position | Planform curvature of the channel; inside vs outside bend | Outer banks erode |
| Valley / channel geometry | Local channel slope, stream-power index | DEM along the channel × drainage area | Flow energy |
| Attenuation | Along-river distance from the border | Done in the pilot (`02_terrain.py`): D8 flood path traced from Rasuwagadhi; each building takes its nearest path cell's distance. Next: try log(distance) | Flood weakens downstream (RQ2) |
| Attenuation | HAND × along-river distance | Interaction term | Critical height changes downstream |

- [x] EDA: distributions by grade and region, correlation matrix. (`WEEK2_REPORT.md`)
- [x] PCA on the correlated terrain block.
- [x] Feature selection: filter by mutual information and wrapper RFE; compare. (+ L1 path; + ablation preview)

**Deliverables:** feature table plus 4–6 EDA figures.

### Week 3 (12 – 18 Oct): Modelling (every syllabus module)
- [x] Targets: (ordinal 4-class skipped; see `WEEK3_REPORT.md`)
  - (a) affected vs not
  - (b) destroyed vs not
  - (c) destroyed vs damaged, among affected buildings
  - (d) optional ordinal 4-class
- [x] Models:
  - Logistic Regression (L1 / L2 / elastic net)
  - LDA
  - Decision Tree
  - k-NN
  - Gaussian Naive Bayes
  - SVM (linear, RBF)
  - Random Forest, gradient boosting
- [x] Linear Regression module: estimate the *attenuation curve*.
  1. Fit a logistic model per river segment.
  2. Take each segment's HAND₅₀ (the height where damage probability is 50%).
  3. Regress HAND₅₀ on along-river distance.

  The slope is a headline physical result.
- [x] Class imbalance: class weights vs SMOTE vs threshold tuning; report PR-AUC.
- [x] Tuning: nested CV. The inner loop is spatial blocks; the outer loop is leave-one-reach-out.
- [x] K-Means / hierarchical clustering of affected buildings: damage regimes, e.g. channel-bank vs fan / terrace.

**Deliverables:** `results/scores.csv` (model × target × CV scheme), tuned hyperparameters.

### Week 4 (19 – 25 Oct): Evaluation and interpretation
- [x] Transfer test (RQ2): leave-one-reach-out, with and without the attenuation features.
- [x] Ablation (RQ3): terrain groups only.
  1. Core: HAND + distance
  2. + local terrain
  3. + valley / channel geometry
  4. + attenuation

  Test whether each step's gain is significant (paired bootstrap on AUC).
- [x] Interpretation:
  - permutation importance and partial dependence (HAND, distance)
  - Decision Tree rules as simple setback guidance
  - map of prediction errors
  - Error check with ICIMOD NLCMS land cover (not a feature). Test whether misclassified buildings concentrate in or next to forest, where the Copernicus DSM overstates ground height and inflates HAND. This keeps one ICIMOD dataset in the study (preferred by the course).
- [x] Sensitivity:
  - DEM (GLO-30 / FABDEM / ALOS)
  - channel threshold
  - ambiguous-label rule (drop vs relabel as affected)
  - minimum footprint size
- [x] RQ4 if labels exist: apply the model to July 2025. (No building-level labels exist; dropped. See `WEEK1_REPORT.md` §5.)

**Deliverables:** final figures and tables, frozen results.

### Week 5 (26 Oct – 1 Nov): Writing
| Section | Content | Approx. length |
|---|---|---|
| Abstract | Question, data, best model, attenuation finding | 200 words |
| 1 Introduction | 2026 event, Himalayan cascades, gap (§5 of lit review), RQs | 1 page |
| 2 Literature review | Themes 3.1–3.5 + comparison table | 2–3 pages |
| 3 Study area and data | Map; EMS labels; footprints (label locations only); DEM and derived channel network; ICIMOD NLCMS (error analysis only) | 1.5 pages |
| 4 Methods | Cleaning log, features, models, CV design, metrics | 2 pages |
| 5 Results | EDA, model comparison, transfer, ablation, attenuation curve, tree rules | 3 pages |
| 6 Discussion | Physical meaning, bias–variance on transfer, label noise, limitations, policy use (setbacks, screening other valleys) | 1.5 pages |
| 7 Conclusion | | 0.5 page |
| References | 24–30 entries (already verified) | |

### Week 6 (2 – 8 Nov): Buffer and submission
- [x] Fill in the "n/r" performance cells in the literature table. (Done in `paper/paper.tex` Table 1; "--" marks descriptive studies.)
- [x] Re-search for new 2026-event papers. (26 Oct 2026: no peer-reviewed building-damage or ML paper on the event found; Li 2026 arXiv remains the only event-specific preprint in the table.)
- [x] Clean notebook / README so the whole pipeline re-runs. (`pilot/run_all.sh` runs steps 01-27; README lists the order.)
- [ ] Proofread and submit.

---

## Syllabus coverage

| Module | Where it appears |
|---|---|
| Decision Trees | Interpretable model + setback rules |
| k-NN + evaluation metrics | Model; ROC-AUC, PR-AUC, balanced accuracy, confusion matrices |
| Linear Regression | Attenuation curve (HAND₅₀ vs along-river distance) |
| Logistic Regression, regularisation, LDA | Main transferable model; L1 / L2; LDA baseline |
| K-Means / hierarchical | Damage-regime clustering |
| Feature selection / extraction, PCA | Terrain-block PCA; filter vs wrapper selection |
| SVM | Linear and RBF |
| Naive Bayes | Probabilistic baseline |
| Ensembles, class imbalance | RF, boosting; class weights / SMOTE / threshold tuning |
| Reinforcement learning | Not used. The Discussion states why: a static prediction problem, no sequential decisions. |

---

## Risks and fallbacks
| Risk | Fallback |
|---|---|
| EMS labels in one area turn out unreliable | Drop or down-weight that area; report it |
| FABDEM / ALOS download or processing fails | Keep GLO-30 and discuss DSM bias |
| Attenuation features extrapolate badly to a held-out reach (pilot: near-channel lower reach 0.83 → 0.66) | Compare log(distance), a fitted HAND₅₀ curve and monotonic constraints; report honestly |
| No July 2025 labels | Drop RQ4; it is a stretch goal |
| Time overrun | Cut the ordinal target and the curvature / TWI features; keep RQ1–RQ3 |
