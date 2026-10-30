"""Week 3 - tuned classical models with nested spatial cross-validation.

Outer loop : leave-one-region-out (Upper / Middle / Lower) = the transfer test.
Inner loop : 3-fold GroupKFold over 1-km spatial blocks WITHIN the training regions,
             used for hyperparameter tuning (GridSearchCV, ROC-AUC) and for choosing a
             decision threshold (Youden's J on out-of-fold training predictions).
Targets    : affected (all buildings), destroyed (all), destroyed | affected.
             Affected models are also scored on the near-channel subset of each test reach.
Feature sets (Week 2):
  main        core + local terrain + valley/channel geometry
  attenuation core + local terrain + log along-river distance (+ interaction)
  all         both
Models: logistic regression (L2/L1/elastic net), LDA, decision tree, k-NN, Gaussian naive Bayes,
linear SVM, RBF SVM (Nystroem kernel approximation), random forest, histogram gradient boosting.

Outputs: results/scores.csv, results/best_params.csv
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from clean import load_clean

warnings.filterwarnings("ignore")
SEED = 42
CORE = ["hand", "dist"]
LOCAL = ["slope", "relief", "curv_profile", "curv_plan", "twi"]
GEOM = ["log_valley_w20", "chan_slope", "bend"]
ATT = ["logd", "hand_x_logd"]
FEATURE_SETS = {"main": CORE + LOCAL + GEOM, "attenuation": CORE + LOCAL + ATT, "all": CORE + LOCAL + GEOM + ATT}

def scaled(est):
    return Pipeline([("scale", StandardScaler()), ("clf", est)])

MODELS = {
    "Logistic regression": (scaled(LogisticRegression(solver="saga", class_weight="balanced", max_iter=5000, random_state=SEED)),
                            {"clf__C": [0.001, 0.01, 0.1, 1, 10], "clf__l1_ratio": [0.0, 0.5, 1.0]}),
    "LDA": (scaled(LinearDiscriminantAnalysis(solver="lsqr")), {"clf__shrinkage": [None, "auto"]}),
    "Decision tree": (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                      {"max_depth": [2, 3, 4, 6, 8], "min_samples_leaf": [10, 50, 200]}),
    "k-NN": (scaled(KNeighborsClassifier()), {"clf__n_neighbors": [5, 15, 25, 51, 101],
                                             "clf__weights": ["uniform", "distance"]}),
    "Naive Bayes": (scaled(GaussianNB()), {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]}),
    "Linear SVM": (scaled(LinearSVC(class_weight="balanced", max_iter=20000)), {"clf__C": [0.01, 0.1, 1]}),
    "RBF SVM (Nystroem)": (Pipeline([("scale", StandardScaler()),
                                     ("kernel", Nystroem(n_components=500, random_state=SEED)),
                                     ("clf", LinearSVC(class_weight="balanced", max_iter=20000))]),
                           {"kernel__gamma": [0.05, 0.2, 1.0], "clf__C": [0.1, 1]}),
    "Random forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=1, random_state=SEED),
                      {"max_depth": [None, 8, 16], "min_samples_leaf": [5, 20], "max_features": ["sqrt", 0.5]}),
    "Gradient boosting": (HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
                          {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31], "l2_regularization": [0.0, 1.0]}),
}

def scores_of(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)

def youden_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr - fpr)]

def metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    return dict(n=len(y), prevalence=y.mean(), roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s),
                bal_acc=balanced_accuracy_score(y, pred), f1=f1_score(y, pred))

def main():
    df, _ = load_clean(verbose=False)
    df["region"] = df["region"].str.split(" ").str[0]
    df["near"] = (df.hand <= 30) & (df.dist <= 300)
    targets = {"affected": (df, "affected"), "destroyed": (df, "destroyed"),
               "destroyed | affected": (df[df.affected == 1].copy(), "destroyed")}
    rows, params = [], []
    t0 = time.time()
    for tname, (data, y) in targets.items():
        for fs_name, feats in FEATURE_SETS.items():
            for reg in ["Upper", "Middle", "Lower"]:
                tr, te = data[data.region != reg], data[data.region == reg]
                inner = list(GroupKFold(n_splits=3).split(tr, tr[y], tr["block"]))
                for mname, (est, grid) in MODELS.items():
                    gs = GridSearchCV(clone(est), grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
                    gs.fit(tr[feats], tr[y])
                    oof = cross_val_predict(clone(gs.best_estimator_), tr[feats], tr[y], cv=inner, n_jobs=-1,
                                            method="predict_proba" if hasattr(gs.best_estimator_, "predict_proba")
                                            else "decision_function")
                    oof = oof[:, 1] if oof.ndim == 2 else oof
                    thr = youden_threshold(tr[y], oof)
                    s = scores_of(gs.best_estimator_, te[feats])
                    base = dict(target=tname, features=fs_name, model=mname, test=reg,
                                inner_auc=gs.best_score_, threshold=thr)
                    rows.append({**base, "domain": "all", **metrics(te[y].to_numpy(), s, thr)})
                    if tname == "affected":
                        m = te["near"].to_numpy()
                        if te.loc[m, y].nunique() == 2:
                            rows.append({**base, "domain": "near-channel", **metrics(te[y].to_numpy()[m], s[m], thr)})
                    params.append({**base, "best_params": str(gs.best_params_)})
                print(f"[{time.time() - t0:6.0f}s] {tname} / {fs_name} / test={reg} done", flush=True)
                pd.DataFrame(rows).to_csv("results/scores.csv", index=False)
                pd.DataFrame(params).to_csv("results/best_params.csv", index=False)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    for (tname, dom), g in res.groupby(["target", "domain"]):
        t = g.pivot_table(index="model", columns="features", values="roc_auc", aggfunc="mean").round(3)
        print(f"\n== {tname} ({dom}): mean ROC-AUC over held-out reaches ==\n{t.to_string()}")

if __name__ == "__main__":
    main()
