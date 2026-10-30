"""Week 3 - class-imbalance handling (syllabus: ensembles and class imbalance).

Main feature set, leave-one-region-out. Prevalence of the positive class differs strongly by
reach (affected: upper 77%, middle 20%, lower 9%), so the decision rule matters as much as the
ranking. Strategies compared for logistic regression and random forest:
  none        no reweighting, threshold 0.5
  none+tuned  no reweighting, threshold = Youden's J on out-of-fold training predictions
  weights     class_weight="balanced", threshold 0.5
  SMOTE       synthetic minority oversampling fitted ONLY on the training folds, threshold 0.5
Outputs results/imbalance.csv
"""
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clean import load_clean

warnings.filterwarnings("ignore")
SEED = 42
FEATS = ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "log_valley_w20", "chan_slope", "bend"]

def make(model, strategy):
    w = "balanced" if strategy == "weights" else None
    clf = (LogisticRegression(C=1.0, class_weight=w, max_iter=5000) if model == "Logistic regression" else
           RandomForestClassifier(300, min_samples_leaf=20, max_features="sqrt", class_weight=w, n_jobs=-1,
                                  random_state=SEED))
    steps = [("scale", StandardScaler())]
    if strategy == "SMOTE":
        steps.append(("smote", SMOTE(random_state=SEED)))
        return ImbPipeline(steps + [("clf", clf)])
    return Pipeline(steps + [("clf", clf)])

def main():
    df, _ = load_clean(verbose=False)
    df["region"] = df["region"].str.split(" ").str[0]
    rows = []
    for target in ["affected", "destroyed"]:
        for model in ["Logistic regression", "Random forest"]:
            for strategy in ["none", "none+tuned", "weights", "SMOTE"]:
                for reg in ["Upper", "Middle", "Lower"]:
                    tr, te = df[df.region != reg], df[df.region == reg]
                    est = make(model, "none" if strategy == "none+tuned" else strategy)
                    est.fit(tr[FEATS], tr[target])
                    s = est.predict_proba(te[FEATS])[:, 1]
                    thr = 0.5
                    if strategy == "none+tuned":
                        inner = GroupKFold(3).split(tr, tr[target], tr["block"])
                        oof = cross_val_predict(make(model, "none"), tr[FEATS], tr[target], cv=inner,
                                                method="predict_proba")[:, 1]
                        fpr, tpr, t = roc_curve(tr[target], oof)
                        thr = t[np.argmax(tpr - fpr)]
                    y, pred = te[target].to_numpy(), (s >= thr).astype(int)
                    rows.append(dict(target=target, model=model, strategy=strategy, test=reg, prevalence=y.mean(),
                                     threshold=thr, roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s),
                                     bal_acc=balanced_accuracy_score(y, pred), recall=recall_score(y, pred),
                                     precision=precision_score(y, pred, zero_division=0), f1=f1_score(y, pred)))
    res = pd.DataFrame(rows)
    res.to_csv("results/imbalance.csv", index=False)
    pd.set_option("display.width", 250)
    summ = res.groupby(["target", "model", "strategy"])[["roc_auc", "pr_auc", "bal_acc", "recall", "precision", "f1"]].mean()
    print(summ.round(3).to_string())

if __name__ == "__main__":
    main()
