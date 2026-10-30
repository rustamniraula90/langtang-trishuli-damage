"""Week 2 - exploratory data analysis of the terrain features.

Outputs (results/):
  eda_feature_medians.csv       median of each feature by region x grade
  eda_single_feature_auc.csv    direction-free AUC of each feature for four tasks
  eda_pca.csv                   explained variance and loadings of PCA on the standardised features
  eda_feature_selection.csv     mutual information, RFE rank and L1 entry order (target: affected)
  fig_eda_boxplots.png, fig_eda_correlation.png, fig_eda_pca.png
Feature selection here is DESCRIPTIVE (all data). Model selection in Week 3 is done inside CV.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from clean import load_clean

OUT = pathlib.Path("results")
FEATS = {
    "hand": "HAND (m)", "dist": "Distance to flood path (m)", "logd": "log along-river distance",
    "slope": "Slope (deg)", "relief": "Local relief (m)", "curv_profile": "Profile curvature",
    "curv_plan": "Plan curvature", "twi": "TWI", "log_valley_w20": "log valley width (HAND<=20 m)",
    "chan_slope": "Channel slope", "log_spi": "log stream-power index", "bend": "Bend exposure (+ = outer)",
}
GRADES = ["Not affected", "Possibly damaged", "Damaged", "Destroyed"]
COLOURS = {"Not affected": "#8d99ae", "Possibly damaged": "#f2c14e", "Damaged": "#f78154", "Destroyed": "#b3001b"}
SEED = 42

def free_auc(y, x):
    a = roc_auc_score(y, x)
    return max(a, 1 - a), ("+" if a >= 0.5 else "-")

def main():
    df, _ = load_clean(verbose=False)
    X = df[list(FEATS)]

    med = df.groupby(["region", "grade"])[list(FEATS)].median().round(3)
    med.to_csv(OUT / "eda_feature_medians.csv")

    tasks = {"affected": (df, "affected"), "destroyed": (df, "destroyed"),
             "affected near-channel": (df[(df.hand <= 30) & (df.dist <= 300)], "affected"),
             "destroyed | affected": (df[df.affected == 1], "destroyed")}
    rows = []
    for f in FEATS:
        r = {"feature": f}
        for t, (d, y) in tasks.items():
            auc, sign = free_auc(d[y], d[f])
            r[t] = round(auc, 3)
            r[f"{t} dir"] = sign
        rows.append(r)
    sf = pd.DataFrame(rows)
    sf.to_csv(OUT / "eda_single_feature_auc.csv", index=False)
    pd.set_option("display.width", 250)
    print("Direction-free single-feature AUC ('+' = higher value -> more damage):")
    print(sf.to_string(index=False))

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for ax, (f, label) in zip(axes.ravel(), FEATS.items()):
        data = [df.loc[df.grade == g, f].to_numpy() for g in GRADES]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        for patch, g in zip(bp["boxes"], GRADES):
            patch.set_facecolor(COLOURS[g])
        ax.set_xticks(range(1, 5), ["None", "Possibly", "Damaged", "Destroyed"], fontsize=8)
        ax.set_title(label, fontsize=9)
    fig.suptitle("Terrain features by EMS damage grade (all reaches; outliers hidden)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_eda_boxplots.png", dpi=130)
    plt.close(fig)

    corr = X.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATS)), list(FEATS), rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(FEATS)), list(FEATS), fontsize=8)
    for i in range(len(FEATS)):
        for j in range(len(FEATS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman correlation")
    ax.set_title("Correlation between terrain features")
    fig.tight_layout()
    fig.savefig(OUT / "fig_eda_correlation.png", dpi=130)
    plt.close(fig)
    strong = [(a, b, round(corr.loc[a, b], 2)) for i, a in enumerate(FEATS) for b in list(FEATS)[i + 1:]
              if abs(corr.loc[a, b]) >= 0.6]
    print("\nStrongly correlated pairs (|rho| >= 0.6):", strong)

    Z = StandardScaler().fit_transform(X)
    pca = PCA(random_state=SEED).fit(Z)
    load = pd.DataFrame(pca.components_[:4].T, index=list(FEATS), columns=[f"PC{i+1}" for i in range(4)])
    ev = pd.Series(pca.explained_variance_ratio_, index=[f"PC{i+1}" for i in range(len(FEATS))])
    pd.concat([load.round(3), ev.to_frame("explained").T.iloc[:, :4].round(3)]).to_csv(OUT / "eda_pca.csv")
    print("\nPCA explained variance:", ev.round(3).head(6).to_dict(), "cumulative 4 PCs:", round(ev.head(4).sum(), 3))
    print("PC1/PC2 loadings:\n", load[["PC1", "PC2"]].round(2).to_string())
    pcs = pca.transform(Z)[:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    for g in GRADES:
        m = (df["grade"].to_numpy()[idx] == g)
        axes[0].scatter(pcs[idx][m, 0], pcs[idx][m, 1], s=3, alpha=0.4, c=COLOURS[g], label=g)
    axes[0].set_xlabel(f"PC1 ({ev.iloc[0]:.0%})"); axes[0].set_ylabel(f"PC2 ({ev.iloc[1]:.0%})")
    axes[0].legend(markerscale=4, fontsize=8); axes[0].set_title("Buildings in PC1-PC2 space")
    for f in FEATS:
        axes[1].arrow(0, 0, load.loc[f, "PC1"], load.loc[f, "PC2"], head_width=0.02, color="#333")
        axes[1].text(load.loc[f, "PC1"] * 1.12, load.loc[f, "PC2"] * 1.12, f, fontsize=8, ha="center")
    axes[1].set_xlim(-0.8, 0.8); axes[1].set_ylim(-0.8, 0.8); axes[1].axhline(0, c="#ccc"); axes[1].axvline(0, c="#ccc")
    axes[1].set_title("Loadings (feature directions)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_eda_pca.png", dpi=130)
    plt.close(fig)

    y = df["affected"].to_numpy()
    mi = mutual_info_classif(Z, y, random_state=SEED)
    rfe = RFE(LogisticRegression(class_weight="balanced", max_iter=5000), n_features_to_select=1).fit(Z, y)

    entry = {}
    for C in np.logspace(-4, 1, 60):
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced").fit(Z, y)
        for f, c in zip(FEATS, m.coef_[0]):
            if abs(c) > 1e-6 and f not in entry:
                entry[f] = len(entry) + 1
    sel = pd.DataFrame({"feature": list(FEATS), "mutual_info": mi.round(4), "rfe_rank": rfe.ranking_,
                        "l1_entry_order": [entry.get(f, np.nan) for f in FEATS]}).sort_values("rfe_rank")
    sel.to_csv(OUT / "eda_feature_selection.csv", index=False)
    print("\nFeature selection (target = affected):\n", sel.to_string(index=False))

if __name__ == "__main__":
    main()
