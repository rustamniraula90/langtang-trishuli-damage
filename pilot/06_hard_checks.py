"""Harder pilot checks: near-channel buildings only, and destroyed vs damaged among affected buildings."""
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clean import load_clean

F = ["hand", "dist", "slope", "relief"]
ATT = F + ["along_km", "hand_x_along"]
df, _ = load_clean(verbose=False)
df["region"] = df["region"].str.split(" ").str[0]
MODELS = [("LogReg", F, make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))),
          ("RF", F, RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=42)),
          ("LogReg + attenuation", ATT,
           make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)))]

def leave_region_out(d, target, label):
    for name, feats, model in MODELS:
        out = []
        for reg in ["Upper", "Middle", "Lower"]:
            tr, te = d[d.region != reg], d[d.region == reg]
            if te[target].nunique() < 2 or len(te) < 30:
                out.append(f"{reg}: n/a")
                continue
            p = clone(model).fit(tr[feats], tr[target]).predict_proba(te[feats])[:, 1]
            out.append(f"{reg}: n={len(te)} prev={te[target].mean():.2f} "
                       f"AUC={roc_auc_score(te[target], p):.3f} PR={average_precision_score(te[target], p):.3f}")
        print(f"[{label}] {name}: " + " | ".join(out))

near = df[(df.hand <= 30) & (df.dist <= 300)]
print("Near-channel subset (HAND<=30 m, dist<=300 m):", len(near), "buildings; affected share", round(near.affected.mean(), 2))
leave_region_out(near, "affected", "near-channel, affected vs not")
aff = df[df.affected == 1]
print("\nAmong affected buildings only:", len(aff), "; destroyed share", round(aff.destroyed.mean(), 2))
leave_region_out(aff, "destroyed", "affected only, destroyed vs damaged/possibly")
print("\nSingle-feature AUC within near-channel subset:", {f: round(roc_auc_score(near.affected, -near[f]), 3) for f in F})
print("Single-feature AUC destroyed|affected:", {f: round(roc_auc_score(aff.destroyed, -aff[f]), 3) for f in F})
