"""Week 2 - which terrain feature groups transfer to an unseen reach? (preview of the Week 4 ablation)

Groups are added cumulatively; logistic regression (standardised, balanced) and random forest,
leave-one-region-out ROC-AUC on four tasks. Untuned models: this ranks feature groups, it is not the
final model comparison.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clean import load_clean

GROUPS = {
    "core": ["hand", "dist"],
    "local": ["slope", "relief", "curv_profile", "curv_plan", "twi"],
    "geometry": ["log_valley_w20", "chan_slope", "bend"],
    "attenuation": ["logd", "hand_x_logd"],
}
SETS = {
    "core": GROUPS["core"],
    "core+local": GROUPS["core"] + GROUPS["local"],
    "core+local+geometry": GROUPS["core"] + GROUPS["local"] + GROUPS["geometry"],
    "core+local+attenuation": GROUPS["core"] + GROUPS["local"] + GROUPS["attenuation"],
    "all": GROUPS["core"] + GROUPS["local"] + GROUPS["geometry"] + GROUPS["attenuation"],
}
MODELS = {
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
    "RF": lambda: RandomForestClassifier(400, min_samples_leaf=10, class_weight="balanced", n_jobs=-1, random_state=42),
}

def main():
    df, _ = load_clean(verbose=False)
    df["region"] = df["region"].str.split(" ").str[0]
    tasks = {"affected": (df, "affected"), "destroyed": (df, "destroyed"),
             "affected near-channel": (df[(df.hand <= 30) & (df.dist <= 300)], "affected"),
             "destroyed | affected": (df[df.affected == 1], "destroyed")}
    rows = []
    for task, (d, y) in tasks.items():
        for mname, make in MODELS.items():
            for sname, feats in SETS.items():
                r = dict(task=task, model=mname, features=sname)
                for reg in ["Upper", "Middle", "Lower"]:
                    tr, te = d[d.region != reg], d[d.region == reg]
                    if te[y].nunique() < 2 or len(te) < 30:
                        continue
                    p = make().fit(tr[feats], tr[y]).predict_proba(te[feats])[:, 1]
                    r[reg] = roc_auc_score(te[y], p)
                r["mean"] = np.nanmean([r[k] for k in ["Upper", "Middle", "Lower"] if k in r])
                rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv("results/ablation_preview.csv", index=False)
    pd.set_option("display.width", 250)
    for task in tasks:
        t = res[res.task == task].pivot_table(index="features", columns="model", values="mean").reindex(SETS)
        print(f"\n== {task}: mean leave-one-region-out AUC ==\n{t.round(3).to_string()}")
    print("\nPer-region detail:\n", res.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
