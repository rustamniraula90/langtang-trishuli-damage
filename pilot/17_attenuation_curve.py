"""Week 3 - the attenuation curve (syllabus: linear regression).

1. Split buildings into BIN_KM bins of along-river distance. In each bin with at least MIN_CLASS
   positives and negatives, fit P(damage) = sigmoid(b0 + b1 * HAND) and take HAND50 = -b0 / b1.
2. Regress HAND50 on distance with ordinary least squares, comparing a straight line in d with a
   straight line in log(d + c). The offset c (unknown collapse-to-border distance) is chosen by
   leave-one-bin-out cross-validation - NOT on any test reach.
3. Uncertainty: three bootstraps of the slope, with the form and c held fixed and everything else
   (bins, HAND50, OLS) refitted in each resample:
     "block"    - resample the 1-km spatial blocks with replacement (headline; neighbouring buildings
                  share terrain and flood exposure, so buildings are not independent draws);
     "building" - resample buildings within each bin (the earlier, too-narrow interval);
     "bin"      - resample the bins themselves (between-bin scatter, which is what the OLS residuals see).
Outputs results/attenuation_bins.csv, results/attenuation_fit.csv, results/attenuation_boot.csv,
results/fig_attenuation_final.png
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from clean import load_clean

warnings.filterwarnings("ignore")
BIN_KM = 3.0
MIN_CLASS = 15
C_GRID = [1, 2, 5, 10, 20, 40]
N_BOOT = 300
SEED = 42

def h50_per_bin(df, target):
    out = []
    for b, g in df.groupby(df["along_km"] // BIN_KM):
        pos, neg = int(g[target].sum()), int((1 - g[target]).sum())
        if pos < MIN_CLASS or neg < MIN_CLASS:
            continue
        m = LogisticRegression(C=1e4, max_iter=5000).fit(g[["hand"]], g[target])
        b0, b1 = m.intercept_[0], m.coef_[0, 0]
        if b1 < 0:
            out.append(dict(bin=int(b), km=g["along_km"].median(), h50=-b0 / b1, n=len(g), pos=pos, neg=neg))
    return pd.DataFrame(out)

def transform(km, form, c):
    return np.log(km + c) if form == "log" else km

def lobo_rmse(bins, form, c):
    """Leave-one-bin-out prediction error of the OLS line."""
    err = []
    for i in range(len(bins)):
        tr = bins.drop(bins.index[i])
        m = LinearRegression().fit(transform(tr[["km"]], form, c), tr["h50"])
        err.append(m.predict(transform(bins.iloc[[i]][["km"]], form, c))[0] - bins.iloc[i]["h50"])
    return float(np.sqrt(np.mean(np.square(err))))

def boot_slopes(df, bins, target, form, c, scheme, rng):
    """Bootstrap distribution of the OLS slope under one resampling scheme."""
    slopes = []
    blocks = df["block"].to_numpy()
    ub = np.unique(blocks)
    members = {b: np.flatnonzero(blocks == b) for b in ub}
    for _ in range(N_BOOT):
        if scheme == "building":
            bb = h50_per_bin(df.groupby(df["along_km"] // BIN_KM, group_keys=False).apply(
                lambda g: g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))), target)
        elif scheme == "block":
            idx = np.concatenate([members[b] for b in rng.choice(ub, size=len(ub), replace=True)])
            bb = h50_per_bin(df.iloc[idx], target)
        else:
            bb = bins.iloc[rng.integers(len(bins), size=len(bins))]
        if len(bb) >= 3 and bb["km"].nunique() >= 3:
            slopes.append(LinearRegression().fit(transform(bb[["km"]], form, c), bb["h50"]).coef_[0])
    return np.array(slopes)

SCHEMES = ["block", "building", "bin"]

def main():
    df, _ = load_clean(verbose=False)
    rng = np.random.default_rng(SEED)
    fits, bin_tables, boot_rows = [], [], []
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for target, colour in [("affected", "#1f77b4"), ("destroyed", "#b3001b")]:
        bins = h50_per_bin(df, target)
        cands = [("linear", 0)] + [("log", c) for c in C_GRID]
        cv = {(f, c): lobo_rmse(bins, f, c) for f, c in cands}
        best_c = min(C_GRID, key=lambda c: cv[("log", c)])
        for form, c in [("linear", 0), ("log", best_c)]:
            X = transform(bins[["km"]], form, c)
            m = LinearRegression().fit(X, bins["h50"])
            per_doubling = m.coef_[0] * np.log(2) if form == "log" else np.nan
            ci = {}
            for scheme in SCHEMES:
                sl = boot_slopes(df, bins, target, form, c, scheme, rng)
                lo, hi = np.percentile(sl, [2.5, 97.5])
                ci[scheme] = (lo, hi)
                boot_rows.append(dict(target=target, form=form, c_km=c, scheme=scheme, n_boot=len(sl),
                                      slope=m.coef_[0], slope_ci_low=lo, slope_ci_high=hi,
                                      slope_sd=sl.std(ddof=1),
                                      per_doubling_m=per_doubling,
                                      per_doubling_ci_low=lo * np.log(2) if form == "log" else np.nan,
                                      per_doubling_ci_high=hi * np.log(2) if form == "log" else np.nan))
            lo, hi = ci["block"]
            fits.append(dict(target=target, form=form, c_km=c, bins=len(bins), intercept=m.intercept_,
                             slope=m.coef_[0], slope_ci_low=lo, slope_ci_high=hi, ci_scheme="block",
                             change_per_doubling_m=per_doubling,
                             per_doubling_ci=(lo * np.log(2), hi * np.log(2)) if form == "log" else None,
                             r2=m.score(X, bins["h50"]), lobo_rmse=cv[(form, c)]))
            if form == "log":
                xs = np.linspace(bins["km"].min(), bins["km"].max(), 200)
                ax.plot(xs, m.predict(transform(pd.DataFrame({"km": xs}), form, c)), color=colour,
                        label=f"{target}: {per_doubling:+.0f} m per doubling "
                              f"(95% block-bootstrap CI {lo*np.log(2):+.0f} to {hi*np.log(2):+.0f}); c = {c} km")
        ax.scatter(bins["km"], bins["h50"], color=colour, s=18)
        bins.insert(0, "target", target)
        bin_tables.append(bins)
        print(f"{target}: leave-one-bin-out RMSE", {f"{f}{'' if f=='linear' else f'(c={c})'}": round(v, 2)
                                                     for (f, c), v in cv.items()})
    ax.set_xlabel("Distance along the river from the border (km)")
    ax.set_ylabel("HAND$_{50}$ (m)")
    ax.set_title(f"Attenuation curve: HAND$_{{50}}$ per {BIN_KM:.0f}-km bin, OLS on log(distance + c)", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/fig_attenuation_final.png", dpi=150)
    pd.concat(bin_tables).to_csv("results/attenuation_bins.csv", index=False)
    fit = pd.DataFrame(fits)
    fit.to_csv("results/attenuation_fit.csv", index=False)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv("results/attenuation_boot.csv", index=False)
    pd.set_option("display.width", 250)
    print(fit.round(3).to_string(index=False))
    print("\nSlope intervals by bootstrap scheme (log form: metres per doubling of distance from the source)")
    print(boot.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
