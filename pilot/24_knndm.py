"""Week 4 add-on - does the inner CV estimate transfer performance honestly? (kNNDM vs 1-km blocks)

Week 3 (19_optimism_gap.py) found that inner 1-km block CV overstates held-out-reach AUC, most for
flexible models. kNNDM (spatial_tools.knndm) chooses folds whose distances match the distances from the
held-out reach to the training reaches. For each held-out reach, three inner 3-fold designs are compared:
  random     : random 3-fold
  block_1km  : GroupKFold(3) over 1-km blocks (as in Weeks 3-4)
  kNNDM      : k-means folds matched to the prediction distances
Reported per design: W (Wasserstein distance between CV and prediction NN-distance distributions; lower =
closer to the real transfer setting), median CV NN distance, and for each model the inner CV AUC, the
held-out AUC and the gap (inner - held-out). Main feature set, class-weighted models as in 20_final_eval.
Outputs results/knndm_folds.csv, results/knndm_gap.csv, results/fig_knndm_ecdf.png
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clean import load_clean
from spatial_tools import fold_nn_dist, knndm, nn_dist, xy

warnings.filterwarnings("ignore")
SEED = 42
MAIN = ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "log_valley_w20", "chan_slope", "bend"]
MODELS = {
    "Logistic regression": lambda: Pipeline([("s", StandardScaler()), ("m", LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=5000))]),
    "Random forest": lambda: RandomForestClassifier(300, max_depth=16, min_samples_leaf=5, max_features="sqrt",
                                                    class_weight="balanced", n_jobs=-1, random_state=SEED),
    "Gradient boosting": lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=15,
                                                                class_weight="balanced", random_state=SEED),
}

def inner_auc(make, X, y, folds):
    oof = np.empty(len(y))
    for f in np.unique(folds):
        m = folds == f
        oof[m] = make().fit(X[~m], y[~m]).predict_proba(X[m])[:, 1]
    return roc_auc_score(y, oof)

def main():
    df, _ = load_clean(verbose=False)
    df["region"] = df["region"].str.split(" ").str[0]
    P = xy(df)
    fold_rows, gap_rows = [], []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, reg in zip(axes, ["Upper", "Middle", "Lower"]):
        tr_m = (df.region != reg).to_numpy()
        tr, te = df[tr_m], df[~tr_m]
        xy_tr, xy_te = P[tr_m], P[~tr_m]
        g_ij = nn_dist(xy_te, xy_tr)
        designs = {
            "random": np.random.default_rng(SEED).integers(0, 3, len(tr)),
            "block_1km": np.empty(len(tr), int),
        }
        for f, (_, idx) in enumerate(GroupKFold(3).split(tr, groups=tr["block"])):
            designs["block_1km"][idx] = f
        designs["kNNDM"], w_best, q_best, cand = knndm(xy_tr, xy_te, k=3, seed=SEED)
        ax.plot(np.sort(g_ij) / 1000, np.linspace(0, 1, len(g_ij)), "k-", lw=2.2, label="held-out reach → training")
        for name, folds in designs.items():
            d = fold_nn_dist(xy_tr, folds)
            w = wasserstein_distance(d, g_ij)
            fold_rows.append(dict(test=reg, design=name, W_m=w, median_cv_nn_m=np.median(d),
                                  median_pred_nn_m=np.median(g_ij), q=q_best if name == "kNNDM" else None,
                                  fold_sizes="/".join(map(str, np.bincount(folds)))))
            ax.plot(np.sort(d) / 1000, np.linspace(0, 1, len(d)), lw=1.3, label=f"{name} (W = {w / 1000:.2f} km)")
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_title(f"Held-out: {reg}")
        ax.set_xlabel("nearest-neighbour distance (km)")
        ax.legend(fontsize=7, loc="lower right")
        for target in ["affected", "destroyed"]:
            X, y = tr[MAIN].to_numpy(), tr[target].to_numpy()
            for mn, make in MODELS.items():
                out = roc_auc_score(te[target], make().fit(X, y).predict_proba(te[MAIN].to_numpy())[:, 1])
                for name, folds in designs.items():
                    a = inner_auc(make, X, y, folds)
                    gap_rows.append(dict(target=target, model=mn, test=reg, design=name, inner_auc=a,
                                         heldout_auc=out, gap=a - out))
            print(reg, target, "done", flush=True)
    axes[0].set_ylabel("ECDF")
    fig.tight_layout()
    fig.savefig("results/fig_knndm_ecdf.png", dpi=150)
    folds = pd.DataFrame(fold_rows)
    gaps = pd.DataFrame(gap_rows)
    folds.to_csv("results/knndm_folds.csv", index=False)
    gaps.to_csv("results/knndm_gap.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nFold designs (distances in m):")
    print(folds.round(0).to_string(index=False))
    print("\nMean inner AUC, held-out AUC and gap over reaches and targets:")
    print(gaps.groupby(["model", "design"])[["inner_auc", "heldout_auc", "gap"]].mean().round(3).to_string())
    print("\nMean |gap| by design:")
    print(gaps.assign(abs_gap=gaps.gap.abs()).groupby("design").abs_gap.mean().round(3).to_string())

if __name__ == "__main__":
    main()
