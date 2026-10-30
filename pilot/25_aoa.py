"""Week 4 add-on - area of applicability (AOA): which held-out buildings lie outside the training data?

For each held-out reach and feature set, the dissimilarity index (DI, spatial_tools.dissimilarity) of every
held-out building is computed in the standardised feature space, weighted by the permutation importance
(ROC-AUC drop) of the class-weighted logistic regression on the training reaches. The AOA threshold comes
from the training DI under the same 1-km block 3-fold CV as Weeks 3-4.
Expectation: the attenuation set (log along-river distance) puts whole reaches outside the AOA, because
a held-out reach lies outside the training range of distance; the main set should not.
Reported: share of held-out buildings outside the AOA, and held-out AUC inside vs outside the AOA.
Outputs results/aoa.csv, results/aoa_buildings.csv, results/fig_aoa_di.png
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from clean import load_clean
from spatial_tools import dissimilarity

warnings.filterwarnings("ignore")
SEED = 42
SETS = {
    "main": ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "log_valley_w20", "chan_slope", "bend"],
    "attenuation": ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "logd", "hand_x_logd"],
}

def auc_or_nan(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 and len(y) >= 20 else np.nan

def main():
    df, _ = load_clean(verbose=False)
    df["region"] = df["region"].str.split(" ").str[0]
    rows, bld = [], []
    for fs, feats in SETS.items():
        for target in ["affected", "destroyed"]:
            for reg in ["Upper", "Middle", "Lower"]:
                tr, te = df[df.region != reg], df[df.region == reg]
                sc = StandardScaler().fit(tr[feats])
                Xtr, Xte = sc.transform(tr[feats]), sc.transform(te[feats])
                m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(Xtr, tr[target])
                imp = permutation_importance(m, Xtr, tr[target], scoring="roc_auc", n_repeats=5,
                                             random_state=SEED).importances_mean
                folds = np.empty(len(tr), int)
                for f, (_, idx) in enumerate(GroupKFold(3).split(tr, groups=tr["block"])):
                    folds[idx] = f
                di, di_tr, thr = dissimilarity(Xtr, Xte, imp, folds, seed=SEED)
                inside = di <= thr
                s, y = m.predict_proba(Xte)[:, 1], te[target].to_numpy()
                rows.append(dict(features=fs, target=target, test=reg, threshold=thr, n=len(te),
                                 share_outside=1 - inside.mean(), auc_all=auc_or_nan(y, s),
                                 auc_inside=auc_or_nan(y[inside], s[inside]),
                                 auc_outside=auc_or_nan(y[~inside], s[~inside]),
                                 prev_inside=y[inside].mean() if inside.any() else np.nan,
                                 prev_outside=y[~inside].mean() if (~inside).any() else np.nan,
                                 top_weight=feats[int(np.argmax(imp))]))
                if target == "affected":
                    bld.append(pd.DataFrame(dict(bid=te.bid, features=fs, test=reg, along_km=te.along_km, di=di,
                                                 threshold=thr, inside=inside, affected=y)))
    res = pd.DataFrame(rows)
    res.to_csv("results/aoa.csv", index=False)
    b = pd.concat(bld)
    b.to_csv("results/aoa_buildings.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))

    p = pd.read_csv("results/predictions_loro.csv")
    p = p[(p.target == "affected") & (p.features == "main")]
    a = b[b.features == "main"][["bid", "test", "inside"]].drop_duplicates(["bid", "test"])
    j = p.merge(a, on=["bid", "test"], how="inner")
    cmp = (j.groupby(["test", "model", "inside"])
           .apply(lambda g: pd.Series(dict(n=len(g), auc=auc_or_nan(g.y.to_numpy(), g.score.to_numpy()))))
           .unstack("inside"))
    cmp.columns = [f"{s}_{'inside' if i else 'outside'}" for s, i in cmp.columns]
    cmp.to_csv("results/aoa_models.csv")
    print("\nHeld-out AUC inside vs outside the AOA (affected, main set, Week 4 models):")
    print(cmp.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, fs in zip(axes, SETS):
        d = b[b.features == fs]
        for reg, c in zip(["Upper", "Middle", "Lower"], ["C0", "C1", "C2"]):
            r = d[d.test == reg]
            ax.scatter(r.along_km, r.di / r.threshold, s=2, alpha=0.3, color=c, label=f"{reg} held out")
        ax.axhline(1, color="k", lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"{fs} feature set (affected model)")
        ax.set_xlabel("distance along the flood path from the border (km)")
        ax.legend(fontsize=7, markerscale=4)
    axes[0].set_ylabel("DI / AOA threshold (>1 = outside AOA)")
    fig.tight_layout()
    fig.savefig("results/fig_aoa_di.png", dpi=150)

if __name__ == "__main__":
    main()
