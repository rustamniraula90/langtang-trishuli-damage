"""Can ISMOTE replace SMOTE? (syllabus: class imbalance)

ISMOTE (Song & Liu 2024, Neural Processing Letters 56:240, doi:10.1007/s11063-024-11695-w) is
re-implemented from Section 3 of the paper, since no code is published:
  1. fit the classifier on the (current) training data and predict the training data;
  2. the error set is the misclassified training points;
  3. an error point is "safe" if its true class is the minority class, "noise" otherwise
     (the four rules of Sect. 3.3 reduce to this: the kNN counts N_M, N_L never change the outcome);
  4. for every safe point x_i, pick a random same-class neighbour x_j among its K nearest and add
     x_i + r (x_j - x_i), r ~ U(0, 1)  (Eq. 4, i.e. the SMOTE rule);
  5. repeat until the classes are balanced or n_iter iterations are reached, then fit the final model.
The paper does not say how many points are made per safe point per iteration; one is used here, capped
so the minority never exceeds the majority. K = 5 and n_iter = 20 follow the paper's recommendations.

Design as in 16_imbalance.py: main feature set, leave-one-reach-out, threshold 0.5, oversampling fitted
on the training reaches only. Stochastic strategies are repeated over seeds. Learners: logistic
regression (main model), a decision tree (stand-in for the paper's C4.5) and random forest.
Outputs results/ismote.csv
"""
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from clean import load_clean

warnings.filterwarnings("ignore")
FEATS = ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "log_valley_w20", "chan_slope", "bend"]
SEEDS = {"Logistic regression": 5, "Decision tree": 5, "Random forest": 3}

def learner(model, weighted, seed):
    w = "balanced" if weighted else None
    if model == "Logistic regression":
        return LogisticRegression(C=1.0, class_weight=w, max_iter=5000)
    if model == "Decision tree":
        return DecisionTreeClassifier(criterion="entropy", min_samples_leaf=20, class_weight=w, random_state=seed)
    return RandomForestClassifier(200, min_samples_leaf=20, max_features="sqrt", class_weight=w, n_jobs=-1,
                                  random_state=seed)

def ismote(X, y, clf, k=5, n_iter=20, seed=0):
    rng = np.random.default_rng(seed)
    minority = int(np.argmin(np.bincount(y)))
    X, y = X.copy(), y.copy()
    added = []
    for _ in range(n_iter):
        gap = (y != minority).sum() - (y == minority).sum()
        if gap <= 0:
            break
        pred = clone(clf).fit(X, y).predict(X)
        safe = np.flatnonzero((pred != y) & (y == minority))
        if len(safe) == 0:
            break
        if len(safe) > gap:
            safe = rng.choice(safe, gap, replace=False)
        pool = np.flatnonzero(y == minority)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X[pool])
        nbr = nn.kneighbors(X[safe], return_distance=False)[:, 1:]
        j = pool[nbr[np.arange(len(safe)), rng.integers(0, k, len(safe))]]
        new = X[safe] + rng.random((len(safe), 1)) * (X[j] - X[safe])
        X, y = np.vstack([X, new]), np.concatenate([y, np.full(len(safe), minority)])
        added.append(len(safe))
    return X, y, len(added), sum(added)

def main():
    df, _ = load_clean(verbose=False)
    df["region"] = df["region"].str.split(" ").str[0]
    rows = []
    for target in ["affected", "destroyed"]:
        for model, n_seeds in SEEDS.items():
            for strategy in ["none", "weights", "SMOTE", "ISMOTE"]:
                for reg in ["Upper", "Middle", "Lower"]:
                    tr, te = df[df.region != reg], df[df.region == reg]
                    sc = StandardScaler().fit(tr[FEATS])
                    Xtr, Xte = sc.transform(tr[FEATS]), sc.transform(te[FEATS])
                    ytr, yte = tr[target].to_numpy().astype(int), te[target].to_numpy().astype(int)
                    for seed in range(n_seeds if strategy in ("SMOTE", "ISMOTE") or model != "Logistic regression" else 1):
                        clf = learner(model, strategy == "weights", seed)
                        X, y, iters, n_new = Xtr, ytr, 0, 0
                        if strategy == "SMOTE":
                            X, y = SMOTE(k_neighbors=5, random_state=seed).fit_resample(Xtr, ytr)
                            n_new = len(y) - len(ytr)
                        elif strategy == "ISMOTE":
                            X, y, iters, n_new = ismote(Xtr, ytr, clf, seed=seed)
                        s = clf.fit(X, y).predict_proba(Xte)[:, 1]
                        pred = (s >= 0.5).astype(int)
                        rec = recall_score(yte, pred)
                        spec = recall_score(yte, pred, pos_label=0)
                        rows.append(dict(target=target, model=model, strategy=strategy, test=reg, seed=seed,
                                         train_prev=ytr.mean(), test_prev=yte.mean(), iterations=iters,
                                         synthetic=n_new, roc_auc=roc_auc_score(yte, s),
                                         pr_auc=average_precision_score(yte, s),
                                         bal_acc=balanced_accuracy_score(yte, pred), recall=rec,
                                         precision=precision_score(yte, pred, zero_division=0),
                                         f1=f1_score(yte, pred), g_mean=np.sqrt(rec * spec),
                                         brier=brier_score_loss(yte, s)))
                print(target, model, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("results/ismote.csv", index=False)
    pd.set_option("display.width", 250)
    per_reach = res.groupby(["target", "model", "strategy", "test"]).mean(numeric_only=True).reset_index()
    summ = per_reach.groupby(["target", "model", "strategy"])[
        ["roc_auc", "pr_auc", "bal_acc", "recall", "precision", "f1", "g_mean", "brier", "synthetic", "iterations"]].mean()
    print(summ.round(3).to_string())
    print("\nTraining prevalence by held-out reach:")
    print(res.groupby(["target", "test"]).train_prev.first().round(3).to_string())
    sd = res[res.strategy.isin(["SMOTE", "ISMOTE"])].groupby(["target", "model", "strategy", "test"]).roc_auc.std()
    print("\nMax seed-to-seed SD of ROC-AUC:", round(sd.max(), 4))

if __name__ == "__main__":
    main()
