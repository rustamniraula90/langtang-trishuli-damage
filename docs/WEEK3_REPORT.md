# Week 3 report: tuned models, imbalance, attenuation curve, clustering (29 Sep 2026)

Plan reference: `RESEARCH_PLAN.md`, Week 3. Code: `pilot/15_models.py` … `19_optimism_gap.py`; all run by `pilot/run_all.sh`. `15_models.py` takes about 20 min.

Working assumption (unchanged). *Affected* is the primary target. Your Week 1 audit review is still pending.

## 1. Design
- Outer loop: leave-one-region-out (Upper = Syaprubesi + Timure, Middle = Betrawati–Bidur, Lower = Phosretar). This is the transfer test.
- Inner loop: 3-fold GroupKFold over 1-km spatial blocks *within* the training reaches. It is used for (a) grid-search tuning on ROC-AUC and (b) choosing the decision threshold (Youden's J on out-of-fold training predictions).
- Targets: affected; destroyed; destroyed | affected. Affected models are also scored on the near-channel subset of each held-out reach.
- Feature sets (from Week 2):
  - `main` = core + local + geometry (10 features)
  - `attenuation` = core + local + log along-river distance (+ HAND interaction)
  - `all` = both
- Models:
  - Logistic regression, with elastic-net `l1_ratio` ∈ {0, 0.5, 1} (L2 / elastic net / L1)
  - LDA (with shrinkage)
  - Decision tree
  - k-NN
  - Gaussian naive Bayes
  - Linear SVM
  - RBF SVM (Nystroem kernel approximation, 500 components, for speed)
  - Random forest
  - Histogram gradient boosting
- Class weights are "balanced" where supported (§3).

## 2. Model comparison (`results/scores.csv`, `results/best_params.csv`)
Mean ROC-AUC over the three held-out reaches:

| Model | Affected: main | Affected: attenuation | Affected, near-channel: best set | Destroyed: all | Destroyed \| affected: attenuation |
|---|---|---|---|---|---|
| **Logistic regression** | **0.969** | 0.968 | 0.831 (all) | 0.968 | 0.827 |
| **Linear SVM** | **0.969** | 0.967 | **0.837** (all) | **0.972** | **0.828** |
| LDA | 0.945 | 0.949 | 0.777 (att.) | 0.937 | 0.817 |
| Naive Bayes | 0.934 | 0.961 | 0.763 (att.) | 0.947 | 0.777 |
| RBF SVM (Nystroem) | 0.954 | 0.879 | 0.820 (att.) | 0.951 | 0.724 |
| k-NN | 0.918 | 0.959 | 0.776 (att.) | 0.919 | 0.808 |
| Random forest | 0.920 | 0.963 | 0.828 (all) | 0.966 | 0.805 |
| Gradient boosting | 0.890 | 0.965 | 0.823 (all) | 0.966 | 0.764 |
| Decision tree | 0.878 | 0.829 | 0.750 (main) | 0.945 | 0.640 |

Main feature set, affected, per held-out reach:

| Model | Upper | Middle | Lower |
|---|---|---|---|
| Logistic regression | 0.971 | 0.983 | 0.952 |
| Random forest | **0.854** | 0.959 | 0.948 |
| Gradient boosting | **0.760** | 0.966 | 0.944 |

Findings:
1. Linear models transfer best. Logistic regression and linear SVM are at or near the top on every target and feature set. For affected on the main set they reach 0.969 mean AUC, with PR-AUC 0.85 and balanced accuracy 0.91 at the tuned threshold.
2. Tree ensembles and k-NN need the attenuation features. On the main set they collapse on the upper reach: RF 0.85, gradient boosting 0.76. With log along-river distance they recover to 0.96–0.97. Linear models do not need it.
3. Destroyed vs damaged, among affected buildings, stays hard (best 0.83). It benefits most from attenuation.
4. Near-channel buildings stay hard (best 0.84). Tuning on overall AUC does not specifically help this subset. The untuned Week 2 core + local logistic regression reached 0.862 there.
5. Tuned hyperparameters are consistent across folds for k-NN (k = 25–51, distance weights), RBF SVM (γ = 0.05) and RF (depth 16 / None, `sqrt`, leaf 5). Logistic regression's best C and l1_ratio vary by fold, which shows a flat optimum: regularisation barely matters with 10 features and about 12,000 rows.

## 3. Validation optimism (`19_optimism_gap.py`, `results/optimism_gap.csv`)
Optimism gap = inner spatial-block tuning AUC minus held-out-reach AUC (main set, mean over targets and reaches):

| Model | Inner CV | Held-out reach | Gap |
|---|---|---|---|
| Linear SVM | 0.926 | 0.905 | **0.022** |
| Logistic regression | 0.930 | 0.908 | **0.022** |
| LDA | 0.903 | 0.874 | 0.029 |
| Naive Bayes | 0.881 | 0.848 | 0.034 |
| RBF SVM | 0.923 | 0.880 | 0.043 |
| k-NN | 0.901 | 0.837 | 0.063 |
| Random forest | 0.931 | 0.859 | 0.071 |
| Gradient boosting | 0.929 | 0.835 | 0.093 |
| Decision tree | 0.892 | 0.772 | **0.120** |

- Inside the training reaches, flexible models look as good as linear ones: RF 0.931 vs logistic regression 0.930.
- On a new reach they lose 3–5 times more. Spatial blocks *within* a reach are not enough to reveal this; only the leave-one-reach-out test does.
- This is the central bias–variance result of the project. It also backs the Week 1 choice of leave-one-reach-out as the main test.

## 4. Class imbalance (`16_imbalance.py`, `results/imbalance.csv`)
Main set, mean over held-out reaches:

| Target | Model | Strategy | ROC-AUC | Balanced acc. | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|
| Affected | Logistic regression | none (threshold 0.5) | 0.971 | 0.879 | 0.791 | **0.813** | 0.776 |
| | | tuned threshold | 0.971 | 0.904 | **0.930** | 0.689 | 0.748 |
| | | class weights | 0.970 | **0.910** | 0.916 | 0.721 | **0.777** |
| | | SMOTE | 0.970 | 0.908 | 0.908 | 0.724 | 0.774 |
| Destroyed | Logistic regression | none | 0.965 | 0.825 | 0.689 | 0.753 | 0.695 |
| | | class weights | 0.964 | **0.908** | 0.934 | 0.634 | **0.714** |
| | | SMOTE | 0.965 | 0.906 | 0.906 | 0.650 | 0.713 |

- Ranking (AUC) is unchanged by any strategy. Only the decision rule changes.
- Without reweighting, a 0.5 threshold misses many damaged buildings (recall 0.79 for affected, 0.69 for destroyed).
- Class weights, SMOTE and threshold tuning give almost the same balanced accuracy (≈ 0.90–0.91). Decision: use class weights (simplest; SMOTE adds nothing). For a warning or screening use, prefer high recall (tuned threshold).
- Random forest is weaker under every strategy (AUC 0.91–0.93); weights and SMOTE help it more than they help logistic regression.

## 5. The attenuation curve (Linear Regression module, `17_attenuation_curve.py`)
- Method:
  - 3-km bins of along-river distance; in each bin with ≥ 15 buildings of each class, a logistic fit gives HAND₅₀.
  - OLS of HAND₅₀ on distance: straight line vs line in log(d + c).
  - c is chosen by leave-one-bin-out CV, not on a test reach.
  - Bootstrap (300 resamples of buildings) for the slope.

| Target | Form | Leave-one-bin-out RMSE | R² | Change per doubling of distance (95% CI) |
|---|---|---|---|---|
| Affected | linear | 10.4 m | 0.79 | – |
| Affected | **log, c = 5 km** | **4.7 m** | **0.94** | **−19.5 m** (−21.4 to −17.5) |
| Destroyed | linear | 5.5 m | 0.90 | – |
| Destroyed | **log, c = 20 km** | **4.0 m** | **0.93** | **−23.0 m** (−24.8 to −21.1) |

- Figure: `results/fig_attenuation_final.png`. HAND₅₀ falls from about 87 m (affected) / 63 m (destroyed) at the border to about 16 m / 11 m at 90 km.
- Caveats:
  1. Only two bins cover the upper reach (0–16 km), and no buildings were mapped between about 16 and 42 km. The steep upstream part of the curve rests on few points.
  2. The bootstrap resamples buildings, not spatial blocks, so the intervals are probably too narrow.
  3. The best c differs by target (5 vs 20 km), and the CV error is flat between 5 and 40 km. c should be read as a smoothing parameter. It is not a physical distance.

## 6. Damage regimes, by clustering (`18_clustering.py`, `results/fig_clusters.png`)
K-means on the 3,360 affected buildings (standardised main features):
- Silhouette is best at k = 2 but weak (0.27). Agreement with Ward hierarchical clustering is ARI = 0.73.

| Cluster | n | Destroyed | Median HAND | Slope | Relief | Channel slope | Valley width (log) | Reach mix |
|---|---|---|---|---|---|---|---|---|
| 0 "open valley" | 2,602 | 74% | 18 m | 7.5° | 86 m | 0.6% | 5.55 (≈ 257 m) | 78% middle, 21% lower |
| 1 "steep gorge" | 758 | 78% | 39 m | 20.9° | 238 m | 2.5% | 4.95 (≈ 141 m) | 82% upper |

- The clusters describe the setting: damage high above a steep, narrow gorge vs lower on an open valley floor.
- They do not separate severity: the share destroyed is similar. This matches the supervised result that geometry and attenuation matter mainly *through* the height threshold.

## 7. Syllabus coverage after Week 3
| Module | Where it is used |
|---|---|
| Decision trees | Tuned trees, and the Week 2 tree rules |
| k-NN and metrics | Tuned k-NN; ROC-AUC, PR-AUC, balanced accuracy, F1 |
| **Linear regression** | **Attenuation curve (§5)** |
| Logistic regression + regularisation, LDA | Elastic-net grid; LDA with shrinkage |
| K-means / hierarchical | §6 |
| Feature selection, PCA | Week 2 |
| SVM | Linear and RBF |
| Naive Bayes | Gaussian NB |
| Ensembles, imbalance | RF, gradient boosting; §4 |

## 8. Recommendations for Week 4
1. Final model: class-weighted logistic regression on the main set for *affected*; add attenuation for *destroyed | affected*. Keep linear SVM as the co-winner.
2. Week 4 interpretation:
   - permutation importance and partial dependence (HAND, distance, channel slope)
   - a shallow decision tree as readable rules
   - an error map
   - the ICIMOD land-cover error check
3. Week 4 uncertainty: block-bootstrap confidence intervals for the held-out AUCs, and paired tests between logistic regression and the ensembles.
4. Still open for you: the audit review (Phosretar "destroyed"), and confirming the primary target.
