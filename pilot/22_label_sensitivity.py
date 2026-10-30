"""Week 4 - sensitivity of the final model to label decisions.

Final logistic regression (class-weighted, C = 1), leave-one-region-out ROC-AUC.
Variants:
  main                              rules R1-R3 as in clean.py
  R1': ambiguous rows -> possibly damaged
  R2: no minimum footprint / 20 m2 minimum
  'possibly damaged' removed        drop the noisiest class entirely
  Phosretar 'destroyed' -> damaged  tests the Week 1 audit doubt (7 of 14 judgeable looked standing)
Outputs results/label_sensitivity.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clean import load_clean

warnings.filterwarnings("ignore")
MAIN = ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "log_valley_w20", "chan_slope", "bend"]
ATTN = ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "logd", "hand_x_logd"]

def logreg():
    return Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))])

def evaluate(df, name):
    df = df.copy()
    df["region"] = df["region"].str.split(" ").str[0]
    rows = []
    for target, sub, feats, y in [("affected", df, MAIN, "affected"), ("destroyed", df, MAIN, "destroyed"),
                                  ("destroyed | affected", df[df.affected == 1], ATTN, "destroyed")]:
        r = dict(variant=name, target=target, n=len(sub))
        for reg in ["Upper", "Middle", "Lower"]:
            tr, te = sub[sub.region != reg], sub[sub.region == reg]
            if te[y].nunique() < 2:
                continue
            r[reg] = roc_auc_score(te[y], logreg().fit(tr[feats], tr[y]).predict_proba(te[feats])[:, 1])
        r["mean"] = np.mean([r[k] for k in ["Upper", "Middle", "Lower"] if k in r])
        rows.append(r)
    return rows

def main():
    rows = []
    base, _ = load_clean(verbose=False)
    rows += evaluate(base, "main (R1-R3)")
    rows += evaluate(load_clean(ambiguous="affected", verbose=False)[0], "R1': ambiguous -> possibly damaged")
    rows += evaluate(load_clean(min_m2=0, verbose=False)[0], "R2: no minimum footprint")
    rows += evaluate(load_clean(min_m2=20, verbose=False)[0], "R2: minimum footprint 20 m2")
    rows += evaluate(base[base.grade != "Possibly damaged"], "'possibly damaged' removed")
    recode = base.copy()
    m = (recode.aoi == "AOI05") & (recode.grade == "Destroyed")
    recode.loc[m, ["grade", "rank", "destroyed"]] = ["Damaged", 2, 0]
    rows += evaluate(recode, "Phosretar 'destroyed' -> damaged")
    res = pd.DataFrame(rows)
    res.to_csv("results/label_sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
