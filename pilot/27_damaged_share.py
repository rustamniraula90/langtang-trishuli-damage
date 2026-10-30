"""Week 5 - predicted vs observed damaged share, per reach and per 3-km bin (as in Currie & Reinke 2026).

Leave-one-reach-out, logistic regression on the main set, target 'affected'.
  observed       share of buildings affected
  raw            mean predicted probability, logistic regression without imbalance correction
  recalibrated   the same after re-fitting the intercept on k = 50 random labelled buildings of the
                 held-out reach (mean of 100 repeats, scored on all buildings of the reach)
  flagged        share predicted positive by the class-weighted model at its tuned threshold
                 (from 20_final_eval.py), i.e. what a map of 'likely affected' would show
Outputs results/damaged_share_reach.csv, results/damaged_share_bins.csv, results/fig_damaged_share.png
"""
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clean import load_clean

warnings.filterwarnings("ignore")
SEED = 42
K, REPEATS, BIN_KM, MIN_BIN = 50, 100, 3.0, 30
OUT = pathlib.Path("results")
MAIN = ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "log_valley_w20", "chan_slope", "bend"]
REACHES = ["Upper", "Middle", "Lower"]

def logreg():
    return Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=1.0, max_iter=5000))])

def intercept_shift(lp, y):
    """Shift d that makes the mean of expit(lp + d) equal the observed share (intercept-only MLE)."""
    share = np.clip(y.mean(), 0.5 / len(y), 1 - 0.5 / len(y))
    return brentq(lambda d: expit(lp + d).mean() - share, -20, 20)

def main():
    df, _ = load_clean(verbose=False)
    df["region"] = df["region"].str.split(" ").str[0]
    rng = np.random.default_rng(SEED)
    flag = pd.read_csv(OUT / "predictions_loro.csv")
    flag = flag[(flag.target == "affected") & (flag.features == "main") & (flag.model == "Logistic regression")]
    flag = flag.assign(flagged=(flag.score >= flag.threshold).astype(int))[["bid", "flagged"]]

    parts = []
    for reg in REACHES:
        tr, te = df[df.region != reg], df[df.region == reg].copy()
        m = logreg().fit(tr[MAIN], tr["affected"])
        lp = logit(np.clip(m.predict_proba(te[MAIN])[:, 1], 1e-12, 1 - 1e-12))
        te["raw"] = expit(lp)
        y = te["affected"].to_numpy()

        recal, count = np.zeros(len(te)), np.zeros(len(te))
        for _ in range(REPEATS):
            idx = rng.choice(len(te), K, replace=False)
            rest = np.ones(len(te), dtype=bool)
            rest[idx] = False
            recal[rest] += expit(lp[rest] + intercept_shift(lp[idx], y[idx]))
            count[rest] += 1
        te["recalibrated"] = recal / np.maximum(count, 1)
        parts.append(te)
    e = pd.concat(parts).merge(flag, on="bid", how="left")

    cols = ["affected", "raw", "recalibrated", "flagged"]
    reach = e.groupby("region")[cols].mean().reindex(REACHES)
    reach.insert(0, "n", e.groupby("region").size().reindex(REACHES))
    reach.to_csv(OUT / "damaged_share_reach.csv")
    e["bin"] = (e["along_km"] // BIN_KM) * BIN_KM + BIN_KM / 2
    bins = e.groupby(["region", "bin"]).agg(n=("affected", "size"), **{c: (c, "mean") for c in cols}).reset_index()
    bins = bins[bins.n >= MIN_BIN]
    bins.to_csv(OUT / "damaged_share_bins.csv", index=False)
    err = bins.assign(**{f"abs_{c}": (bins[c] - bins.affected).abs() for c in cols[1:]})
    pd.set_option("display.width", 200)
    print("Share affected per held-out reach (observed vs predicted):\n", reach.round(3).to_string())
    print(f"\nPer {BIN_KM:.0f}-km bin (n >= {MIN_BIN}): mean absolute error of the share, weighted by buildings")
    for c in cols[1:]:
        print(f"  {c:13s} {np.average(err[f'abs_{c}'], weights=err.n):.3f}")
    print("\nBins:\n", bins.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    style = {"raw": ("#1f77b4", "--", "Predicted, no recalibration"),
             "recalibrated": ("#2ca02c", "-", f"Predicted, intercept re-fitted on {K} labels"),
             "flagged": ("#ff7f0e", ":", "Share flagged (class weights, tuned threshold)")}
    for reg in REACHES:
        b = bins[bins.region == reg]
        ax.scatter(b.bin, b.affected, s=np.clip(b.n / 15, 12, 120), c="k", zorder=3,
                   label="Observed" if reg == "Upper" else None)
        for c, (col, ls, lab) in style.items():
            ax.plot(b.bin, b[c], color=col, ls=ls, marker="o", ms=3, label=lab if reg == "Upper" else None)
    for x0, x1 in [(18, 39)]:
        ax.axvspan(x0, x1, color="0.93", zorder=0)
    ax.text(28.5, 0.9, "no mapped buildings", ha="center", fontsize=8, color="0.4")
    ax.set_xlabel("Distance along the flood path (km)")
    ax.set_ylabel("Share of buildings affected")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.88))
    ax.set_title("Held-out reaches: observed vs predicted share affected (3-km bins; marker size = buildings)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_damaged_share.png", dpi=160)

if __name__ == "__main__":
    main()
