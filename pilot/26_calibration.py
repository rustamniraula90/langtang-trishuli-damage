"""Week 4 add-on - calibration: do the predicted probabilities transfer as well as the ranking?

AUC only measures ranking. For screening, the probability itself matters ("40% chance of damage").
Class weights and oversampling shift probabilities upward by design (van den Goorbergh et al. 2022,
JAMIA 29:1525; Carriero et al. 2025, Stat. Med. 44:e10320), so calibration is reported for each strategy.

Part 1 - held-out calibration per reach (main feature set, affected and destroyed):
  Brier score; calibration intercept (calibration-in-the-large: logit P(y) = a + logit(p), a = 0 is ideal);
  calibration slope (logit P(y) = a + b logit(p), b = 1 is ideal; b < 1 = too extreme);
  expected calibration error (ECE, 10 equal-width bins). Reliability diagrams in fig_calibration.png.
Part 2 - intercept-only recalibration from a few labelled buildings in the new reach (the cheap fix if a
  new area or event has a different base rate / flood size): keep all slopes, refit only the intercept on
  k randomly chosen held-out buildings, score the rest. k = 10..200, 200 repeats. A weak N(0, 2^2) prior on
  the shift keeps it finite when the sample holds one class only.
Outputs results/calibration.csv, results/recalibration.csv, results/fig_calibration.png
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from clean import load_clean

warnings.filterwarnings("ignore")
SEED = 42
MAIN = ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "log_valley_w20", "chan_slope", "bend"]
EPS = 1e-6

def fit_predict(name, Xtr, ytr, Xte):
    if name == "LogReg, no correction":
        m = LogisticRegression(C=1.0, max_iter=5000)
    elif name == "LogReg, class weights":
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
    elif name == "LogReg, SMOTE":
        Xtr, ytr = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
        m = LogisticRegression(C=1.0, max_iter=5000)
    elif name == "Random forest, class weights":
        m = RandomForestClassifier(300, max_depth=16, min_samples_leaf=5, max_features="sqrt",
                                   class_weight="balanced", n_jobs=-1, random_state=SEED)
    else:
        m = HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=15, class_weight="balanced",
                                           random_state=SEED)
    return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]

MODELS = ["LogReg, no correction", "LogReg, class weights", "LogReg, SMOTE", "Random forest, class weights",
          "Gradient boosting, class weights"]

def intercept_shift(y, lp, prior_sd=None):
    def nll(a):
        p = np.clip(expit(lp + a), EPS, 1 - EPS)
        pen = 0 if prior_sd is None else 0.5 * (a / prior_sd) ** 2
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + pen
    return minimize_scalar(nll, bounds=(-15, 15), method="bounded").x

def cal_slope(y, lp):
    m = LogisticRegression(penalty=None, max_iter=1000).fit(lp.reshape(-1, 1), y)
    return m.coef_[0, 0]

def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())

def main():
    df, _ = load_clean(verbose=False)
    df["region"] = df["region"].str.split(" ").str[0]
    rng = np.random.default_rng(SEED)
    rows, rec_rows, curves = [], [], {}
    for target in ["affected", "destroyed"]:
        for reg in ["Upper", "Middle", "Lower"]:
            tr, te = df[df.region != reg], df[df.region == reg]
            sc = StandardScaler().fit(tr[MAIN])
            Xtr, Xte = sc.transform(tr[MAIN]), sc.transform(te[MAIN])
            ytr, y = tr[target].to_numpy(), te[target].to_numpy()
            for name in MODELS:
                p = np.clip(fit_predict(name, Xtr, ytr, Xte), EPS, 1 - EPS)
                lp = logit(p)
                rows.append(dict(target=target, test=reg, model=name, prevalence=y.mean(), mean_pred=p.mean(),
                                 roc_auc=roc_auc_score(y, p), brier=brier_score_loss(y, p),
                                 cal_intercept=intercept_shift(y, lp), cal_slope=cal_slope(y, lp), ece=ece(y, p)))
                if target == "affected":
                    curves[(reg, name)] = (y, p)
                if name in ("LogReg, no correction", "LogReg, class weights"):
                    for k in [10, 20, 50, 100, 200]:
                        for r in range(200):
                            pick = rng.choice(len(y), k, replace=False)
                            rest = np.setdiff1d(np.arange(len(y)), pick)
                            a = intercept_shift(y[pick], lp[pick], prior_sd=2.0)
                            q = expit(lp[rest] + a)
                            rec_rows.append(dict(target=target, test=reg, model=name, k=k, rep=r, shift=a,
                                                 brier_before=brier_score_loss(y[rest], p[rest]),
                                                 brier_after=brier_score_loss(y[rest], q),
                                                 ece_before=ece(y[rest], p[rest]), ece_after=ece(y[rest], q)))
            print(target, reg, "done", flush=True)
    res, rec = pd.DataFrame(rows), pd.DataFrame(rec_rows)
    res.to_csv("results/calibration.csv", index=False)
    rec.to_csv("results/recalibration.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nHeld-out calibration (per reach):")
    print(res.round(3).to_string(index=False))
    print("\nMean over reaches and targets:")
    print(res.groupby("model")[["roc_auc", "brier", "cal_intercept", "cal_slope", "ece"]].mean().round(3).to_string())
    print("\nIntercept-only recalibration with k labelled buildings from the held-out reach (median over repeats):")
    s = rec.groupby(["model", "target", "test", "k"])[["brier_before", "brier_after", "ece_before", "ece_after"]].median()
    print(s.round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, reg in zip(axes, ["Upper", "Middle", "Lower"]):
        for name in MODELS:
            y, p = curves[(reg, name)]
            bins = np.quantile(p, np.linspace(0, 1, 11))
            idx = np.clip(np.searchsorted(bins, p, side="right") - 1, 0, 9)
            mp = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
            my = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
            ax.plot(mp, my, "o-", ms=3, lw=1.2, label=name)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"Affected, held-out: {reg}")
        ax.set_xlabel("mean predicted probability (decile bins)")
    axes[0].set_ylabel("observed share affected")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/fig_calibration.png", dpi=150)

if __name__ == "__main__":
    main()
