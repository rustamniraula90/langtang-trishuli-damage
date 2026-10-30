"""Step 7 - how should attenuation (flood weakening downstream) enter the model?

Compares, for logistic regression under leave-one-region-out transfer:
  base        hand, dist, slope, relief
  linear      base + along_km + hand * along_km                      (04_model.py variant)
  log(c)      base + log(along_km + c) + hand * log(along_km + c)
  norm(c)     hand_rel, dist, slope, relief, where hand_rel = hand / H50(d) and
              H50(d) = a + b * log(d + c) is an attenuation curve fitted ONLY on the
              training regions (per 5-km bin: HAND at which P(affected) = 0.5)

c is the (unknown) distance in km from the collapse to the border; along_km is measured
from the border, so several values are tested. Random forests are not tested here:
a monotonic transform such as log cannot change a tree's splits.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clean import load_clean

BASE = ["hand", "dist", "slope", "relief"]
OFFSETS_KM = [5, 10, 20, 40]
BIN_KM = 5.0
MIN_PER_CLASS = 10

df, _ = load_clean(verbose=False)
df["region"] = df["region"].str.split(" ").str[0]
for c in OFFSETS_KM:
    df[f"logd_{c}"] = np.log(df["along_km"] + c)
    df[f"hand_x_logd_{c}"] = df["hand"] * df[f"logd_{c}"]

TASKS = {
    "affected (all)": (df, "affected"),
    "destroyed (all)": (df, "destroyed"),
    "affected (near-channel)": (df[(df.hand <= 30) & (df.dist <= 300)], "affected"),
    "destroyed | affected": (df[df.affected == 1], "destroyed"),
}

def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))

def fit_h50_curve(train, target, c):
    """Per 5-km bin: logistic fit of target on HAND -> HAND50; then HAND50 ~ log(d + c)."""
    pts = []
    for _, g in train.groupby((train["along_km"] // BIN_KM)):
        if g[target].sum() < MIN_PER_CLASS or (1 - g[target]).sum() < MIN_PER_CLASS:
            continue
        m = LogisticRegression(C=1e4, max_iter=5000).fit(g[["hand"]], g[target])
        b0, b1 = m.intercept_[0], m.coef_[0, 0]
        if b1 < 0:
            pts.append((np.log(g["along_km"].median() + c), -b0 / b1))
    if len(pts) < 2:
        return None, pts
    x, y = np.array(pts).T
    return LinearRegression().fit(x.reshape(-1, 1), y), pts

def h50(curve, d, c):
    return np.clip(curve.predict(np.log(d + c).to_numpy().reshape(-1, 1)), 2.0, None)

rows, curves = [], []
for task, (data, target) in TASKS.items():
    for reg in ["Upper", "Middle", "Lower"]:
        tr, te = data[data.region != reg], data[data.region == reg]
        if te[target].nunique() < 2 or len(te) < 30:
            continue
        variants = {"base": BASE, "linear": BASE + ["along_km", "hand_x_along"]}
        variants.update({f"log(c={c})": BASE + [f"logd_{c}", f"hand_x_logd_{c}"] for c in OFFSETS_KM})
        for name, feats in variants.items():
            p = logreg().fit(tr[feats], tr[target]).predict_proba(te[feats])[:, 1]
            rows.append(dict(task=task, test=reg, variant=name, auc=roc_auc_score(te[target], p)))
        for c in OFFSETS_KM:
            curve, pts = fit_h50_curve(tr, target, c)
            if curve is None:
                rows.append(dict(task=task, test=reg, variant=f"norm(c={c})", auc=np.nan))
                continue
            curves.append(dict(task=task, test=reg, c=c, n_bins=len(pts), slope_m_per_log_km=curve.coef_[0],
                               h50_at_5km=float(h50(curve, pd.Series([5.0]), c)[0]),
                               h50_at_80km=float(h50(curve, pd.Series([80.0]), c)[0])))
            feats = ["hand_rel", "dist", "slope", "relief"]
            trn, tst = tr.assign(hand_rel=tr["hand"] / h50(curve, tr["along_km"], c)), \
                te.assign(hand_rel=te["hand"] / h50(curve, te["along_km"], c))
            p = logreg().fit(trn[feats], trn[target]).predict_proba(tst[feats])[:, 1]
            rows.append(dict(task=task, test=reg, variant=f"norm(c={c})", auc=roc_auc_score(tst[target], p)))

res = pd.DataFrame(rows)
res.to_csv("results/log_attenuation_scores.csv", index=False)
pd.DataFrame(curves).to_csv("results/log_attenuation_curves.csv", index=False)
order = ["base", "linear"] + [f"log(c={c})" for c in OFFSETS_KM] + [f"norm(c={c})" for c in OFFSETS_KM]
pd.set_option("display.width", 250)
for task in TASKS:
    t = res[res.task == task].pivot(index="variant", columns="test", values="auc").reindex(order)
    t = t[[c for c in ["Upper", "Middle", "Lower"] if c in t.columns]]
    t["mean"] = t.mean(axis=1)
    print(f"\n== {task} (ROC-AUC, leave-one-region-out, logistic regression)\n{t.round(3).to_string()}")
print("\nFitted attenuation curves (training regions only):")
print(pd.DataFrame(curves).round(2).to_string(index=False))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_FIG = 10
fig, ax = plt.subplots(figsize=(7, 4.2))
summary = []
for target, colour, label in [("affected", "#1f77b4", "Affected"), ("destroyed", "#b3001b", "Destroyed")]:
    pts = []
    for key, g in df.groupby(df["along_km"] // BIN_KM):
        if g[target].sum() < MIN_PER_CLASS or (1 - g[target]).sum() < MIN_PER_CLASS:
            continue
        m = LogisticRegression(C=1e4, max_iter=5000).fit(g[["hand"]], g[target])
        b0, b1 = m.intercept_[0], m.coef_[0, 0]
        if b1 < 0:

            rng = np.random.default_rng(0)
            boots = []
            for _ in range(200):
                s = g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
                if s[target].nunique() < 2:
                    continue
                mb = LogisticRegression(C=1e4, max_iter=5000).fit(s[["hand"]], s[target])
                if mb.coef_[0, 0] < 0:
                    boots.append(-mb.intercept_[0] / mb.coef_[0, 0])
            lo, hi = np.percentile(boots, [5, 95])
            pts.append((g["along_km"].median(), -b0 / b1, lo, hi, len(g)))
    pts = pd.DataFrame(pts, columns=["km", "h50", "lo", "hi", "n"])
    fit = LinearRegression().fit(np.log(pts[["km"]] + C_FIG), pts["h50"])
    r2 = fit.score(np.log(pts[["km"]] + C_FIG), pts["h50"])
    per_doubling = fit.coef_[0] * np.log(2)
    summary.append(dict(target=target, bins=len(pts), slope_per_log_km=fit.coef_[0],
                        change_per_doubling_m=per_doubling, r2=r2))
    ax.errorbar(pts["km"], pts["h50"], yerr=[pts["h50"] - pts["lo"], pts["hi"] - pts["h50"]],
                fmt="o", color=colour, capsize=2, ms=4, label=f"{label}: HAND$_{{50}}$ per 5-km bin")
    xs = np.linspace(pts["km"].min(), pts["km"].max(), 100)
    ax.plot(xs, fit.predict(np.log(xs.reshape(-1, 1) + C_FIG)), color=colour, lw=1.5,
            label=f"{label}: fit, {per_doubling:+.0f} m per doubling of distance ($R^2$={r2:.2f})")
ax.set_xlabel(f"Distance along the river from the border (km)")
ax.set_ylabel("HAND$_{50}$ (m): height where P(damage) = 0.5")
ax.set_title(f"Attenuation curve: HAND$_{{50}}$ = a + b log(distance + {C_FIG} km)", fontsize=10)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("results/fig_attenuation_curve.png", dpi=150)
print("\nDescriptive attenuation curve (all data, c = %d km):" % C_FIG)
print(pd.DataFrame(summary).round(3).to_string(index=False))
