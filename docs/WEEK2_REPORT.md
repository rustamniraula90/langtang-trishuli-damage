# Week 2 report: terrain features and exploratory analysis (29 Sep 2026)

Plan reference: `RESEARCH_PLAN.md`, Week 2. Code: `pilot/12_features.py`, `13_eda.py`, `14_ablation_preview.py`; all run by `pilot/run_all.sh`. A clean rebuild takes about 7 min.

Working assumption. *Affected vs not affected* is the primary target; *destroyed* is secondary (Week 1 recommendation). Your label-audit review is still pending.

## 1. New terrain features (`12_features.py`)
| Group | Feature | How | Check |
|---|---|---|---|
| Local terrain | `curv_profile` | Profile curvature of the Gaussian-smoothed DEM (> 0 = convex spur) | – |
| | `curv_plan` | Plan curvature (< 0 = converging hollow); set to 0 below 2° slope, where it is undefined | Outliers fell from 1.69 to 0.29 |
| | `twi` | ln(specific catchment area / tan slope) | Range 3.8–25 |
| Valley / channel geometry | `valley_w20`, `valley_w40` | Width of valley floor with HAND ≤ 20 / 40 m, per 500 m of flood path | 84–843 m (w20) |
| | `chan_slope` | Flood-path gradient over ±500 m | Median 0.5%, max 4.7% |
| | `spi` | Drainage area × channel slope | – |
| | `bend` | Signed curvature of the ~1 km-smoothed path; + = building on the outer bank | Synthetic test passed: outer bank +1/km, inner −1/km for a 1 km-radius bend |

Each building takes the geometry values of its nearest flood-path cell. There are no missing values. `clean.load_clean()` merges `data/processed/features.csv` automatically.

## 2. Exploratory analysis (`13_eda.py`)
Figures: `results/fig_eda_boxplots.png`, `fig_eda_correlation.png`, `fig_eda_pca.png`.

Direction-free single-feature AUC (`results/eda_single_feature_auc.csv`):

| Feature | Affected | Destroyed | Affected, near-channel | Destroyed \| affected |
|---|---|---|---|---|
| HAND | **0.933** | **0.935** | 0.758 | **0.709** |
| Distance | 0.914 | 0.915 | 0.763 | 0.676 |
| log along-river distance | 0.716 | 0.766 | **0.797** | 0.684 |
| Channel slope | 0.654 | 0.701 | 0.708 | 0.661 |
| Local relief | 0.627 | 0.641 | 0.698 | 0.569 |
| log valley width | 0.508 | 0.504 | 0.644 | 0.518 |
| Bend exposure | 0.590 | 0.553 | 0.567 | 0.597 |
| Profile curvature | 0.657 | 0.669 | 0.585 | 0.576 |
| TWI / plan curvature | ≤ 0.56 | ≤ 0.53 | ≤ 0.52 | ≤ 0.56 |

Findings:
- Among near-channel buildings, the best single features are the channel-scale ones: along-river distance, channel slope and relief. HAND is not.
- Redundancy: channel slope and stream-power index correlate at ρ = 0.99, so SPI is dropped from the model sets. Other strong pairs:

  | Pair | ρ |
  |---|---|
  | HAND–distance | 0.67 |
  | along-river distance – channel slope | −0.76 |
  | slope–TWI | −0.69 |

  The last two are expected: the channel steepens upstream, and TWI is built from slope.
- PCA: no dominant component (PC1 25%, PC2 17%; 4 PCs = 67%).
  - PC1 is a "position along the valley / channel energy" axis: along-river distance, channel slope and SPI.
  - PC2 is a "hillside vs valley floor" axis: slope and relief against distance and TWI.
  - PCA is therefore more useful for interpretation than for dimension reduction.
- Feature selection (descriptive, all data; `results/eda_feature_selection.csv`): HAND, distance and along-river distance rank at the top in mutual information, RFE and the L1 entry order. The geometry features have high mutual information but low RFE rank, because they overlap with along-river distance.

## 3. Which feature groups transfer? (`14_ablation_preview.py`)
Mean leave-one-region-out ROC-AUC, untuned models:

| Features | Affected | Destroyed | Affected, near-channel | Destroyed \| affected |
|---|---|---|---|---|
| core (HAND, distance) | 0.954 | 0.942 | 0.832 | 0.748 |
| + local terrain | 0.964 | 0.951 | **0.862** | 0.759 |
| + local + **geometry** | **0.970** | 0.964 | 0.856 | 0.788 |
| + local + **attenuation** (log distance) | 0.968 | 0.971 | 0.813 | **0.833** |
| all | 0.968 | **0.972** | 0.824 | 0.820 |

These are logistic-regression results. The random forest is lower in every row, except that with attenuation it catches up (0.962 affected, 0.963 destroyed).

Interpretation:
1. Local terrain helps every task.
2. Valley/channel geometry transfers well. It gives the best *affected* score (0.970) and does not break the near-channel lower reach (0.824 vs 0.727 with attenuation). Channel slope carries much of the downstream weakening (ρ = −0.76 with along-river distance) without having to extrapolate a distance. This is a physically meaningful result for RQ2: *the flood weakens where the valley flattens and widens, and distance alone does not explain it*.
3. Attenuation (log distance) is best for the destroyed tasks, but is still fragile in the near-channel lower reach.
4. Random forests need the attenuation terms to transfer. Without them they learn reach-specific thresholds (upper reach 0.86–0.88).

## 4. Proposed feature sets for Week 3
- Main model: core + local + geometry. It transfers reliably and has no extrapolation risk.
- Comparison: core + local + attenuation, and all features. This lets RQ2 compare "geometry" against "distance" explanations of attenuation.
- Dropped: `spi` (duplicate of channel slope) and `valley_w40` (redundant with `valley_w20`).

## 5. Open items for you
1. Review the Week 1 visual label audit (`pilot/results/audit/`), especially the Phosretar "destroyed" cases.
2. Confirm *affected* as the primary target.
