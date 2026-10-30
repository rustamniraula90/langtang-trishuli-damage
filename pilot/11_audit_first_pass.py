"""Record the FIRST-PASS visual judgements of the audit chips (results/audit/sheet_*.png).

Judgements were made by eye from 80 m x 80 m pre/post chips (Vantor Open Data, 0.35-0.7 m) and are
PROVISIONAL: they must be reviewed by the researcher, who fills the 'judgement' column.
Codes: agree = image is consistent with the EMS grade; plausible = consistent but weak evidence
(damaged/possibly classes); disagree = image contradicts the grade (e.g. 'Destroyed' but the
structure is still visibly standing); unclear = cloud, haze, no image, or cannot tell at this scale.
"""
import pandas as pd

UNCLEAR = {"AOI01": [4, 5, 6, 15, 18, 19, 27], "AOI02": [52, 56, 60, 79, 80], "AOI03": [102, 105, 106, 107, 109, 118, 119],
           "AOI05": [150, 152, 153, 154, 156, 157, 159, 161, 163, 169, 172, 177, 178, 180, 183, 193]}
PLAUSIBLE = {"AOI01": [0, 1, 2, 3, 7, 8, 9], "AOI02": [50, 51, 55], "AOI03": [101, 104], "AOI05": [155]}
DISAGREE = {"AOI05": [162, 164, 165, 167, 170, 176, 179]}
NOTES = {162: "structure still visible in debris; damaged rather than destroyed?",
         164: "building visibly standing, debris around", 165: "building visibly standing, debris around",
         167: "structure visible", 170: "town block looks intact in post image",
         176: "building visibly standing, debris around", 179: "building visibly standing, debris around",
         105: "point inside a solar/greenhouse array, not a house", 107: "footprint is a large greenhouse/solar array",
         119: "no post-event image", 161: "uniform sediment-coloured chip; possible image artefact",
         169: "uniform sediment-coloured chip; possible image artefact"}

s = pd.read_csv("results/audit/label_audit.csv")
num = s["sample_id"].str.split("_").str[1].astype(int)

def code(aoi, n):
    for table, c in [(DISAGREE, "disagree"), (UNCLEAR, "unclear"), (PLAUSIBLE, "plausible")]:
        if n in table.get(aoi, []):
            return c
    return "agree"

s["first_pass"] = [code(a, n) for a, n in zip(s["aoi"], num)]
s["first_pass_note"] = [NOTES.get(n, "") for n in num]
s.to_csv("results/audit/label_audit.csv", index=False)
tab = pd.crosstab([s["aoi"], s["stratum"]], s["first_pass"])
print(tab.to_string())
judged = s[s["first_pass"] != "unclear"]
agree_rate = judged.groupby(["aoi", "stratum"])["first_pass"].apply(lambda x: x.isin(["agree", "plausible"]).mean())
print("\nAgreement among judgeable chips:\n", agree_rate.round(2).unstack().to_string())
tab.to_csv("results/audit/first_pass_summary.csv")
