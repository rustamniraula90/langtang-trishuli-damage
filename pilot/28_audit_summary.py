"""Summarise the researcher's label-audit judgements (results/audit/label_audit.csv, column 'judgement').

Fill 'judgement' with one of: agree | plausible | disagree | unclear (same codes as the first pass,
see 11_audit_first_pass.py). 'note' is free text. Rows left blank fall back to first_pass and are
counted as 'not yet reviewed'. Prints the sentence-level numbers used in the paper's Label audit paragraph.

Output: results/audit/audit_summary.csv, results/log_28_audit_summary.txt (via run_all.sh)
"""
import sys
import pandas as pd

CODES = {"agree", "plausible", "disagree", "unclear"}
s = pd.read_csv("results/audit/label_audit.csv", dtype=str, keep_default_na=False)
j = s["judgement"].str.strip().str.lower()
bad = sorted(set(j) - CODES - {""})
if bad:
    sys.exit(f"invalid judgement codes {bad}; use one of {sorted(CODES)}")
n_blank = int((j == "").sum())
print(f"reviewed: {len(s) - n_blank}/{len(s)} rows  (blank rows use first_pass)")
s["final"] = j.where(j != "", s["first_pass"])
changed = s[(j != "") & (j != s["first_pass"])]
print(f"researcher overrode the first pass in {len(changed)} rows")
if len(changed):
    print(changed[["sample_id", "ems_grade", "first_pass", "judgement", "note"]].to_string(index=False))

tab = pd.crosstab([s["aoi"], s["stratum"]], s["final"]).reindex(columns=sorted(CODES), fill_value=0)
print("\n", tab)
tab.to_csv("results/audit/audit_summary.csv")

judged = s[s["final"] != "unclear"]
ok = judged["final"].isin(["agree", "plausible"])
print("\n--- numbers for the paper ---")
na = judged[judged["stratum"] == "Not affected"]
print(f"not affected: {int(ok[na.index].sum())} of {len(na)} judgeable appeared intact")
up_mid = judged[(judged["stratum"] == "Destroyed") & (judged["aoi"] != "AOI05")]
print(f"destroyed, upper+middle (AOI01-03): agreed in {int(ok[up_mid.index].sum())} of {len(up_mid)} judgeable")
low = judged[(judged["stratum"] == "Destroyed") & (judged["aoi"] == "AOI05")]
print(f"destroyed, Phosretar (AOI05): {int((low['final'] == 'disagree').sum())} of {len(low)} judgeable still appeared to stand")
dp = judged[judged["stratum"] == "Damaged or possibly"]
print(f"damaged/possibly: {int(ok[dp.index].sum())} of {len(dp)} judgeable consistent ({int((dp['final']=='plausible').sum())} plausible only)")
print(f"unclear overall: {int((s['final'] == 'unclear').sum())} of {len(s)}")
