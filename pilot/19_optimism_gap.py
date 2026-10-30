"""Week 3 - optimism gap: inner spatial-block CV AUC (tuning score) minus held-out-reach AUC.

A large gap means spatial blocks inside the training reaches overstate how well a model transfers
to a new reach (flexible models learn reach-specific thresholds). Reads results/scores.csv.
"""
import pandas as pd

r = pd.read_csv("results/scores.csv")
r = r[r.domain == "all"].assign(gap=lambda d: d.inner_auc - d.roc_auc)
pd.set_option("display.width", 250)
print("Optimism gap (mean over targets and reaches):")
print(r.groupby(["model", "features"])["gap"].mean().unstack().round(3).to_string())
u = r[r.features == "main"].groupby("model")[["inner_auc", "roc_auc", "gap"]].mean().round(3).sort_values("gap")
print("\nMain feature set:\n", u.to_string())
u.to_csv("results/optimism_gap.csv")
