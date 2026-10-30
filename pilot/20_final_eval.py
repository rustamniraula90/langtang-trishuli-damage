"""Week 4 - final models, held-out predictions and uncertainty.

Final specification (from Week 3): class-weighted logistic regression; linear SVM as co-winner;
random forest and gradient boosting for comparison. Hyperparameters = the most frequent Week 3
choices (logistic regression's optimum is flat, so C = 1, L2).
  affected, destroyed        : main feature set (+ RF/GB also with attenuation)
  destroyed | affected       : attenuation feature set
For each held-out reach (leave-one-region-out) the script
  * stores out-of-reach predictions for every building (results/predictions_loro.csv),
  * computes ROC-AUC with 95% block-bootstrap intervals (1-km blocks resampled, B = 1000),
  * compares logistic regression with each other model on the SAME bootstrap samples (paired).
Outputs results/final_auc_ci.csv, results/final_paired.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from clean import load_clean

warnings.filterwarnings("ignore")
SEED = 42
B = 1000
MAIN = ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "log_valley_w20", "chan_slope", "bend"]
ATTN = ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "logd", "hand_x_logd"]

def models():
    return {
        "Logistic regression": lambda: Pipeline([("s", StandardScaler()), ("m", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=5000))]),
        "Linear SVM": lambda: Pipeline([("s", StandardScaler()), ("m", LinearSVC(
            C=0.1, class_weight="balanced", max_iter=20000))]),
        "Random forest": lambda: RandomForestClassifier(300, max_depth=16, min_samples_leaf=5, max_features="sqrt",
                                                        class_weight="balanced", n_jobs=-1, random_state=SEED),
        "Gradient boosting": lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=15,
                                                                    class_weight="balanced", random_state=SEED),
    }

def score(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)

SPECS = [
    ("affected", "all", "main", MAIN, ["Logistic regression", "Linear SVM", "Random forest", "Gradient boosting"]),
    ("affected", "all", "attenuation", ATTN, ["Random forest", "Gradient boosting"]),
    ("destroyed", "all", "main", MAIN, ["Logistic regression", "Linear SVM", "Random forest", "Gradient boosting"]),
    ("destroyed | affected", "affected", "attenuation", ATTN,
     ["Logistic regression", "Linear SVM", "Random forest", "Gradient boosting"]),
]

def main():
    df, _ = load_clean(verbose=False)
    df["region"] = df["region"].str.split(" ").str[0]
    rng = np.random.default_rng(SEED)
    preds, ci_rows, paired_rows = [], [], []
    for target, subset, fs, feats, mnames in SPECS:
        data = df[df.affected == 1] if subset == "affected" else df
        y_name = "destroyed" if target.startswith("destroyed") else "affected"
        for reg in ["Upper", "Middle", "Lower"]:
            tr, te = data[data.region != reg], data[data.region == reg]
            y = te[y_name].to_numpy()
            s = {}
            for mn in mnames:
                m = models()[mn]().fit(tr[feats], tr[y_name])
                s[mn] = score(m, te[feats])
                inner = GroupKFold(3).split(tr, tr[y_name], tr["block"])
                oof = cross_val_predict(models()[mn](), tr[feats], tr[y_name], cv=inner,
                                        method="predict_proba" if hasattr(m, "predict_proba") else "decision_function")
                oof = oof[:, 1] if oof.ndim == 2 else oof
                fpr, tpr, thr = roc_curve(tr[y_name], oof)
                threshold = thr[np.argmax(tpr - fpr)]
                preds.append(pd.DataFrame({"bid": te["bid"], "target": target, "features": fs, "model": mn,
                                           "test": reg, "y": y, "score": s[mn], "threshold": threshold}))

            blocks = te["block"].to_numpy()
            ub = np.unique(blocks)
            members = {b: np.flatnonzero(blocks == b) for b in ub}
            boot = {mn: [] for mn in mnames}
            for _ in range(B):
                idx = np.concatenate([members[b] for b in rng.choice(ub, size=len(ub), replace=True)])
                if len(np.unique(y[idx])) < 2:
                    continue
                for mn in mnames:
                    boot[mn].append(roc_auc_score(y[idx], s[mn][idx]))
            for mn in mnames:
                lo, hi = np.percentile(boot[mn], [2.5, 97.5])
                ci_rows.append(dict(target=target, features=fs, model=mn, test=reg, n=len(y), blocks=len(ub),
                                    auc=roc_auc_score(y, s[mn]), ci_low=lo, ci_high=hi))
            if "Logistic regression" in mnames:
                ref = np.array(boot["Logistic regression"])
                for mn in mnames:
                    if mn == "Logistic regression":
                        continue
                    d = ref - np.array(boot[mn])
                    paired_rows.append(dict(target=target, features=fs, other=mn, test=reg,
                                            diff_logreg_minus_other=roc_auc_score(y, s["Logistic regression"])
                                            - roc_auc_score(y, s[mn]),
                                            ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                                            p_other_better=float(np.mean(d < 0))))
    pd.concat(preds).to_csv("results/predictions_loro.csv", index=False)
    ci = pd.DataFrame(ci_rows)
    ci.to_csv("results/final_auc_ci.csv", index=False)
    pr = pd.DataFrame(paired_rows)
    pr.to_csv("results/final_paired.csv", index=False)
    pd.set_option("display.width", 250)
    ci["auc (95% CI)"] = ci.apply(lambda r: f"{r.auc:.3f} ({r.ci_low:.3f}-{r.ci_high:.3f})", axis=1)
    print(ci.pivot_table(index=["target", "features", "model"], columns="test", values="auc (95% CI)",
                         aggfunc="first")[["Upper", "Middle", "Lower"]].to_string())
    pr["diff (95% CI)"] = pr.apply(lambda r: f"{r.diff_logreg_minus_other:+.3f} ({r.ci_low:+.3f} to {r.ci_high:+.3f})",
                                   axis=1)
    print("\nPaired: logistic regression minus other model (positive = logistic regression better)")
    print(pr.pivot_table(index=["target", "features", "other"], columns="test", values="diff (95% CI)",
                         aggfunc="first")[["Upper", "Middle", "Lower"]].to_string())

if __name__ == "__main__":
    main()
