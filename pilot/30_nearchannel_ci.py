"""Near-channel evaluation with uncertainty (promotes the near-channel test to a headline result).

Re-uses the out-of-reach predictions of the final fixed-hyperparameter models (20_final_eval.py,
results/predictions_loro.csv) and scores only buildings with HAND <= 30 m and distance <= 300 m,
i.e. the subset in which the obvious hillside negatives are removed. ROC-AUC per held-out reach
with 95% block-bootstrap intervals (1-km blocks, B = 1000) and paired differences to logistic
regression on the same bootstrap samples. A reach is scored only if the subset holds at least
15 buildings of each class.
Three training-free baselines are added. "HAND only" ranks buildings by -HAND (no parameters).
"HAND cut" is the single HAND threshold that maximises Youden's J on the near-channel buildings of
the two training reaches, applied as a binary rule; "HAND cut (all)" chooses it on all training
buildings instead, which gives 31-55 m, above the 30 m subset ceiling, so inside the subset it flags
every building. A binary rule gives one point on the ROC curve, so its AUC equals its balanced accuracy.
Outputs results/nearchannel_ci.csv, results/nearchannel_paired.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from clean import load_clean

SEED, B = 42, 1000
HAND_MAX, DIST_MAX, MIN_CLASS = 30, 300, 15

def main():
    df, _ = load_clean(verbose=False)
    df["region"] = df["region"].str.split(" ").str[0]
    near = df.loc[(df.hand <= HAND_MAX) & (df.dist <= DIST_MAX),
                  ["bid", "region", "block", "hand", "affected", "destroyed"]]
    pred = pd.read_csv("results/predictions_loro.csv")
    pred = pred[(pred.target.isin(["affected", "destroyed"])) & (pred.features == "main")]
    pred = pred.merge(near[["bid", "block", "hand"]], on="bid", how="inner")

    base = []
    cuts = {}
    for target in ["affected", "destroyed"]:
        for reg in ["Upper", "Middle", "Lower"]:
            tr_all = df[df.region != reg]
            tr_near = near[near.region != reg]
            for name, tr in [("HAND cut", tr_near), ("HAND cut (all)", tr_all)]:
                fpr, tpr, thr = roc_curve(tr[target], -tr["hand"])
                cuts[(target, reg, name)] = -thr[np.argmax(tpr - fpr)]
            te = pred[(pred.target == target) & (pred.test == reg)].drop_duplicates("bid")
            for name, sc in [("HAND only", -te["hand"]),
                             ("HAND cut", (te["hand"] <= cuts[(target, reg, "HAND cut")]).astype(float)),
                             ("HAND cut (all)", (te["hand"] <= cuts[(target, reg, "HAND cut (all)")]).astype(float))]:
                base.append(te.assign(model=name, score=sc.to_numpy()))
    pred = pd.concat([pred] + base, ignore_index=True)
    rng = np.random.default_rng(SEED)
    ci_rows, paired_rows = [], []
    for (target, reg), g in pred.groupby(["target", "test"]):
        wide = g.pivot_table(index="bid", columns="model", values="score", aggfunc="first")
        meta = g.drop_duplicates("bid").set_index("bid").loc[wide.index, ["y", "block"]]
        y, blocks = meta["y"].to_numpy(), meta["block"].to_numpy()
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        if min(n_pos, n_neg) < MIN_CLASS:
            print(f"{target:9s} {reg:6s}: skipped, n={len(y)} pos={n_pos} neg={n_neg}")
            continue
        ub = np.unique(blocks)
        members = {b: np.flatnonzero(blocks == b) for b in ub}
        boot = {mn: [] for mn in wide.columns}
        for _ in range(B):
            idx = np.concatenate([members[b] for b in rng.choice(ub, size=len(ub), replace=True)])
            if len(np.unique(y[idx])) < 2:
                continue
            for mn in wide.columns:
                boot[mn].append(roc_auc_score(y[idx], wide[mn].to_numpy()[idx]))
        for mn in wide.columns:
            s = wide[mn].to_numpy()
            lo, hi = np.percentile(boot[mn], [2.5, 97.5])
            ci_rows.append(dict(target=target, model=mn, test=reg, n=len(y), n_pos=n_pos, n_neg=n_neg,
                                prevalence=n_pos / len(y), blocks=len(ub), auc=roc_auc_score(y, s),
                                ci_low=lo, ci_high=hi, pr_auc=average_precision_score(y, s),
                                hand_cut_m=cuts.get((target, reg, mn), np.nan)))
        ref = np.array(boot["Logistic regression"])
        for mn in wide.columns:
            if mn == "Logistic regression":
                continue
            d = ref - np.array(boot[mn])
            paired_rows.append(dict(target=target, other=mn, test=reg,
                                    diff_logreg_minus_other=roc_auc_score(y, wide["Logistic regression"])
                                    - roc_auc_score(y, wide[mn]),
                                    ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5)))
    ci = pd.DataFrame(ci_rows)
    ci.to_csv("results/nearchannel_ci.csv", index=False)
    pr = pd.DataFrame(paired_rows)
    pr.to_csv("results/nearchannel_paired.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"\nNear-channel subset (HAND <= {HAND_MAX} m, distance <= {DIST_MAX} m): "
          f"{len(near)} of {len(df)} buildings")
    print(near.groupby("region")[["affected", "destroyed"]].agg(["size", "sum"]).to_string())
    ci["auc (95% CI)"] = ci.apply(lambda r: f"{r.auc:.3f} ({r.ci_low:.3f}-{r.ci_high:.3f})", axis=1)
    print("\nHAND cut chosen on the training reaches (m):",
          {f"{t}/{r}/{n}": round(c, 1) for (t, r, n), c in cuts.items()})
    print("\nROC-AUC, near-channel, fixed final models and baselines")
    print(ci.pivot_table(index=["target", "model"], columns="test", values="auc (95% CI)", aggfunc="first").to_string())
    print("\nPR-AUC / prevalence")
    print(ci.pivot_table(index=["target", "model"], columns="test", values="pr_auc", aggfunc="first").round(3).to_string())
    print(ci.drop_duplicates(["target", "test"]).pivot_table(index="target", columns="test", values="prevalence").round(3).to_string())
    pr["diff (95% CI)"] = pr.apply(lambda r: f"{r.diff_logreg_minus_other:+.3f} ({r.ci_low:+.3f} to {r.ci_high:+.3f})", axis=1)
    print("\nPaired: logistic regression minus other model")
    print(pr.pivot_table(index=["target", "other"], columns="test", values="diff (95% CI)", aggfunc="first").to_string())

if __name__ == "__main__":
    main()
