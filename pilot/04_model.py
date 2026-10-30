"""Step 4 - pilot models: does valley geography predict flood damage?

Cleaning: clean.load_clean() (rules R1-R3, see clean.py); in_event is never a feature.

Features use the FLOOD-PATH channel (02_terrain.py); hand_all / dist_all (all channels
>= 500 km2, pilot v1) are reported only as single-feature comparisons.

Evaluation:
  * Leave-one-AOI-out (spatial transfer).
  * Spatial block CV (1 km blocks) pooled over all AOIs.
  * Two domains: 'all' buildings in the AOI, and the harder 'valley floor' subset
    (within 1 km of the main channel and < 100 m above it).
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

from clean import load_clean

OUT = pathlib.Path("results")
OUT.mkdir(exist_ok=True)
SEED = 42
GEO2 = ["hand", "dist"]
GEO4 = ["hand", "dist", "slope", "relief"]

ATT = GEO4 + ["along_km", "hand_x_along"]

ATT_LOG = GEO4 + ["logd", "hand_x_logd"]

MODELS = {
    "LogReg (HAND+dist)": (GEO2, make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))),
    "LogReg": (GEO4, make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))),
    "Decision tree (d=3)": (GEO4, DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=SEED)),
    "k-NN (k=25)": (GEO4, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25))),
    "Naive Bayes": (GEO4, make_pipeline(StandardScaler(), GaussianNB())),
    "SVM (RBF)": (GEO4, make_pipeline(StandardScaler(), SVC(class_weight="balanced", probability=True, random_state=SEED))),
    "Random forest": (GEO4, RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                   n_jobs=-1, random_state=SEED)),
    "LogReg + attenuation": (ATT, make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced",
                                                                                     max_iter=2000))),
    "LogReg + log attenuation": (ATT_LOG, make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced",
                                                                                         max_iter=2000))),
    "Random forest + attenuation": (ATT, RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                                class_weight="balanced", n_jobs=-1,
                                                                random_state=SEED)),
}

def load():
    df, log = load_clean()
    log.to_csv(OUT / "cleaning_log.csv", index=False)
    return df

def scores(y, p):
    return dict(n=len(y), prevalence=round(y.mean(), 3), roc_auc=round(roc_auc_score(y, p), 3),
                pr_auc=round(average_precision_score(y, p), 3),
                bal_acc=round(balanced_accuracy_score(y, (p >= 0.5).astype(int)), 3))

def evaluate(df, target, domain):
    d = df[df["valley_floor"]] if domain == "valley floor" else df
    rows = []
    for name, (feats, model) in MODELS.items():

        for reg in sorted(d["region"].unique()):
            tr, te = d[d["region"] != reg], d[d["region"] == reg]
            if te[target].nunique() < 2:
                continue
            m = clone(model).fit(tr[feats], tr[target])
            rows.append(dict(target=target, domain=domain, cv="leave-region-out", test=reg, model=name,
                             **scores(te[target].values, m.predict_proba(te[feats])[:, 1])))

        p = np.zeros(len(d))
        for tri, tei in GroupKFold(n_splits=5).split(d, d[target], d["block"]):
            m = clone(model).fit(d.iloc[tri][feats], d.iloc[tri][target])
            p[tei] = m.predict_proba(d.iloc[tei][feats])[:, 1]
        rows.append(dict(target=target, domain=domain, cv="spatial-block 5-fold", test="all regions", model=name,
                         **scores(d[target].values, p)))
    return rows

def single_feature_auc(df):
    out = []
    for domain, d in [("all", df), ("valley floor", df[df["valley_floor"]])]:
        for f in GEO4 + ["hand_all", "dist_all", "along_km"]:
            out.append(dict(domain=domain, feature=f, auc_affected=round(roc_auc_score(d["affected"], -d[f]), 3)))
    return pd.DataFrame(out)

def figures(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    colours = {"Not affected": "#9aa5b1", "Possibly damaged": "#f2c14e", "Damaged": "#f78154", "Destroyed": "#b3001b"}
    for ax, (reg, d) in zip(axes, df.groupby("region")):
        for g in ["Not affected", "Possibly damaged", "Damaged", "Destroyed"]:
            s = d[d["grade"] == g]
            ax.scatter(s["dist"], s["hand"], s=6, alpha=0.5, c=colours[g], label=f"{g} ({len(s)})")
        ax.set_xscale("symlog", linthresh=30)
        ax.set_yscale("symlog", linthresh=5)
        ax.set_title(reg)
        ax.set_xlabel("Distance to main channel (m)")
        ax.legend(fontsize=7, markerscale=2)
    axes[0].set_ylabel("Height above main channel, HAND (m)")
    fig.suptitle("2026 Trishuli cascade: building damage vs valley position (EMSR927 labels, pre-event footprints)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_hand_vs_distance.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = [0, 5, 10, 15, 20, 30, 40, 60, 100, 1e4]
    labels = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-40", "40-60", "60-100", ">100"]
    for reg, d in df.groupby("region"):
        rate = d.groupby(pd.cut(d["hand"], bins, labels=labels), observed=False)["destroyed"].mean()
        ax.plot(labels, rate.values, marker="o", label=reg)
    ax.set_xlabel("HAND bin (m)")
    ax.set_ylabel("Share of buildings destroyed")
    ax.legend(fontsize=8)
    ax.set_title("Destruction rate falls with height above the river")
    fig.tight_layout()
    fig.savefig(OUT / "fig_destroyed_rate_by_hand.png", dpi=150)

def main():
    df = load()
    print(df.groupby(["region", "grade"]).size().unstack(fill_value=0))
    print(df[df["valley_floor"]].groupby(["region", "grade"]).size().unstack(fill_value=0))

    sf = single_feature_auc(df)
    print("\nSingle-feature ROC-AUC (lower value = more damage):\n", sf.to_string(index=False))

    rows = []
    for target in ["affected", "destroyed"]:
        for domain in ["all", "valley floor"]:
            rows += evaluate(df, target, domain)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "pilot_scores.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n", res.to_string(index=False))

    tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=SEED).fit(df[GEO4], df["destroyed"])
    rules = export_text(tree, feature_names=GEO4, decimals=1)
    (OUT / "tree_rules_destroyed.txt").write_text(rules)
    print("\nDecision-tree rules (destroyed vs not, all data):\n", rules)

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
                                random_state=SEED).fit(df[GEO4], df["affected"])
    print("RF impurity importance:", dict(zip(GEO4, rf.feature_importances_.round(3))))
    figures(df)

if __name__ == "__main__":
    main()
