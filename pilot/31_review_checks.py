"""Checks requested by the November 2026 review.

1. Near-channel cutoffs: held-out ROC-AUC of the final logistic regression and of HAND alone on
   subsets HAND <= {10, 20, 30} m x distance <= {100, 200, 300} m (reaches with >= 15 per class).
2. Phosretar 'destroyed' bound: the audit found 8 of 18 judgeable lower-reach 'destroyed' buildings
   standing (Wilson 95% interval 25-65%). A share q of the 275 is recoded to damaged (still affected),
   at random (200 repeats) and in the two extreme orders (highest / lowest model score first), and the
   lower-reach destroyed and destroyed | affected scores are recomputed. Models are trained on the
   upper and middle reaches, so only the held-out labels change.
3. OSM-only negatives with the final model: retrained and scored on the OSM-only table, and the main
   model scored on that same smaller table (separates the sample effect from the training effect).
4. Clustered recalibration: the intercept re-fitted on k buildings from ONE settlement (the k nearest
   buildings to a random seed building) instead of k random buildings; ECE and affected-share error.
5. Denominator: prevalence of the lower and other reaches under three denominators (our table; our
   table less the audit's share of 'no building visible' negatives; the Copernicus inventory), and
   the thresholded map's flags as a multiple of the recorded affected buildings.
6. Attenuation support: the curve's bins sit at 2.6 and 14.7 km, then 42-91 km (no buildings mapped in
   between). The log fit is repeated on the downstream bins only, with the same offset c.
Outputs results/review_*.csv and results/log_31_review_checks.txt (via run_all.sh)
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clean import load_clean

warnings.filterwarnings("ignore")
SEED = 42
MAIN = ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "log_valley_w20", "chan_slope", "bend"]
REACHES = ["Upper", "Middle", "Lower"]
EPS = 1e-6
pd.set_option("display.width", 250)

def logreg(weights=True):
    return Pipeline([("s", StandardScaler()),
                     ("m", LogisticRegression(C=1.0, class_weight="balanced" if weights else None, max_iter=5000))])

def load():
    df, _ = load_clean(verbose=False)
    df["region"] = df["region"].str.split(" ").str[0]
    return df

def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h

def auc_or_nan(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan

def cutoffs(df):
    p = pd.read_csv("results/predictions_loro.csv")
    p = p[(p.features == "main") & (p.model == "Logistic regression") & p.target.isin(["affected", "destroyed"])]
    p = p.merge(df[["bid", "hand", "dist"]], on="bid")
    rows = []
    for h in [10, 20, 30]:
        for d in [100, 200, 300]:
            for (t, reg), g in p[(p.hand <= h) & (p.dist <= d)].groupby(["target", "test"]):
                npos, nneg = int(g.y.sum()), int((1 - g.y).sum())
                if min(npos, nneg) < 15:
                    continue
                rows.append(dict(hand_max=h, dist_max=d, target=t, test=reg, n=len(g), prevalence=npos / len(g),
                                 logreg=roc_auc_score(g.y, g.score), hand_only=roc_auc_score(g.y, -g.hand)))
    r = pd.DataFrame(rows)
    r.to_csv("results/review_nearchannel_cutoffs.csv", index=False)
    print("1. Near-channel cutoffs (held-out ROC-AUC; LR = final logistic regression, HAND = HAND alone)")
    w = r.pivot_table(index=["target", "hand_max", "dist_max"], columns="test", values=["n", "logreg", "hand_only"])
    print(w.round(3).to_string())
    print("   range of LR AUC over all cutoffs:", r.groupby(["target", "test"]).logreg.agg(["min", "max"]).round(3).to_string())
    print("   share of cells where HAND alone >= LR:", f"{(r.hand_only >= r.logreg).mean():.2f}", f"(n cells = {len(r)})")

def phosretar(df):
    p = pd.read_csv("results/predictions_loro.csv")
    low = df[df.region == "Lower"][["bid", "affected", "destroyed", "hand", "dist"]]
    sd = p[(p.target == "destroyed") & (p.features == "main") & (p.model == "Logistic regression") & (p.test == "Lower")]
    sda = p[(p.target == "destroyed | affected") & (p.features == "attenuation") & (p.model == "Logistic regression")
            & (p.test == "Lower")]
    d = low.merge(sd[["bid", "score"]], on="bid").merge(sda[["bid", "score"]].rename(columns={"score": "score_da"}),
                                                         on="bid", how="left")
    d["near"] = (d.hand <= 30) & (d.dist <= 300)
    dest = np.flatnonzero(d.destroyed.to_numpy() == 1)
    lo, hi = wilson(8, 18)
    print(f"\n2. Phosretar 'destroyed': audit 8/18 standing = {8 / 18:.2f} (Wilson 95% {lo:.2f}-{hi:.2f}); "
          f"{len(dest)} destroyed in the lower reach")
    rng = np.random.default_rng(SEED)

    def score(y):
        a = d.affected.to_numpy() == 1
        nr = d.near.to_numpy()
        return dict(destroyed=auc_or_nan(y, d.score.to_numpy()),
                    destroyed_pr=average_precision_score(y, d.score.to_numpy()),
                    prevalence=y.mean(),
                    destroyed_near=auc_or_nan(y[nr], d.score.to_numpy()[nr]),
                    d_given_a=auc_or_nan(y[a], d.score_da.to_numpy()[a]))

    rows = [dict(q=0.0, mode="as labelled", **score(d.destroyed.to_numpy()))]
    for q in [round(lo, 2), round(8 / 18, 2), round(hi, 2)]:
        m = int(round(q * len(dest)))
        rand = []
        for _ in range(200):
            y = d.destroyed.to_numpy().copy()
            y[rng.choice(dest, m, replace=False)] = 0
            rand.append(score(y))
        rows.append(dict(q=q, mode="random (mean of 200)", **pd.DataFrame(rand).mean().to_dict()))
        order = dest[np.argsort(d.score.to_numpy()[dest])]
        for mode, pick in [("lowest-scored recoded", order[:m]), ("highest-scored recoded", order[-m:])]:
            y = d.destroyed.to_numpy().copy()
            y[pick] = 0
            rows.append(dict(q=q, mode=mode, **score(y)))
    r = pd.DataFrame(rows)
    r.to_csv("results/review_phosretar_bound.csv", index=False)
    print(r.round(3).to_string(index=False))

def osm_only(df):
    osm, _ = load_clean(negatives="osm", verbose=False)
    osm["region"] = osm["region"].str.split(" ").str[0]
    rows = []
    for t in ["affected", "destroyed"]:
        for reg in REACHES:
            te_o = osm[osm.region == reg]
            m_main = logreg().fit(df[df.region != reg][MAIN], df[df.region != reg][t])
            m_osm = logreg().fit(osm[osm.region != reg][MAIN], osm[osm.region != reg][t])
            te_m = df[df.region == reg]
            rows.append(dict(target=t, test=reg, n_main=len(te_m), prev_main=te_m[t].mean(),
                             auc_main=roc_auc_score(te_m[t], m_main.predict_proba(te_m[MAIN])[:, 1]),
                             n_osm=len(te_o), prev_osm=te_o[t].mean(),
                             auc_main_model_on_osm_sample=roc_auc_score(te_o[t], m_main.predict_proba(te_o[MAIN])[:, 1]),
                             auc_osm_model_on_osm_sample=roc_auc_score(te_o[t], m_osm.predict_proba(te_o[MAIN])[:, 1])))
    r = pd.DataFrame(rows)
    r.to_csv("results/review_osm_only.csv", index=False)
    print("\n3. OSM-only negatives, final logistic regression (10 features)")
    print(r.round(3).to_string(index=False))
    print("   means:", r.groupby("target")[["auc_main", "auc_main_model_on_osm_sample", "auc_osm_model_on_osm_sample"]]
          .mean().round(3).to_string())

def intercept_shift(y, lp, prior_sd=2.0):
    def nll(a):
        p = np.clip(expit(lp + a), EPS, 1 - EPS)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * (a / prior_sd) ** 2
    return minimize_scalar(nll, bounds=(-15, 15), method="bounded").x

def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())

def clustered_recal(df):
    rng = np.random.default_rng(SEED)
    rows = []
    for reg in REACHES:
        tr, te = df[df.region != reg], df[df.region == reg]
        lp = logit(np.clip(logreg(False).fit(tr[MAIN], tr.affected).predict_proba(te[MAIN])[:, 1], EPS, 1 - EPS))
        y = te.affected.to_numpy()
        xy = np.c_[te.lon.to_numpy() * np.cos(np.radians(28.0)), te.lat.to_numpy()] * 111_320
        tree = cKDTree(xy)
        for k in [20, 50]:
            for design in ["random", "one settlement"]:
                for _ in range(200):
                    if design == "random":
                        pick = rng.choice(len(y), k, replace=False)
                    else:
                        pick = tree.query(xy[rng.integers(len(y))], k=k)[1]
                    a = intercept_shift(y[pick], lp[pick])
                    rest = np.setdiff1d(np.arange(len(y)), pick)
                    q = expit(lp + a)
                    rows.append(dict(test=reg, k=k, design=design, ece_after=ece(y[rest], q[rest]),
                                     share_error_pp=100 * abs(q.mean() - y.mean()),
                                     spread_m=float(np.ptp(xy[pick], axis=0).max())))
        rows.append(dict(test=reg, k=0, design="none", ece_after=ece(y, expit(lp)),
                         share_error_pp=100 * abs(expit(lp).mean() - y.mean()), spread_m=np.nan))
    r = pd.DataFrame(rows)
    r.to_csv("results/review_clustered_recal.csv", index=False)
    s = r.groupby(["test", "k", "design"]).agg(ece_median=("ece_after", "median"),
                                               ece_p90=("ece_after", lambda v: np.percentile(v, 90)),
                                               share_err_median=("share_error_pp", "median"),
                                               share_err_p90=("share_error_pp", lambda v: np.percentile(v, 90)),
                                               extent_m=("spread_m", "median"))
    print("\n4. Recalibration from random vs one-settlement labels (affected, LR without correction; 200 repeats)")
    print(s.round(3).to_string())

def denominators(df):
    inv = pd.read_csv("results/audit_ems_inventory.csv")
    reach_aoi = {"Upper": ["AOI01", "AOI02"], "Middle": ["AOI03"], "Lower": ["AOI05"]}
    s = pd.read_csv("results/audit/label_audit.csv", dtype=str, keep_default_na=False)
    j = s.judgement.str.strip().str.lower()
    s["final"] = j.where(j != "", s.first_pass)
    neg = s[s.stratum == "Not affected"].copy()
    neg["no_building"] = (neg.final == "unclear") & ~neg.note.str.contains("cloud|haze|artefact|no usable", case=False)
    fl = pd.read_csv("results/predictions_loro.csv")
    fl = fl[(fl.target == "affected") & (fl.features == "main") & (fl.model == "Logistic regression")]
    rows = []
    for reg, aois in reach_aoi.items():
        g = df[df.region == reg]
        n, a = len(g), int(g.affected.sum())
        nb = neg[neg.aoi.isin(aois)]
        k, m = int(nb.no_building.sum()), len(nb)
        lo, hi = wilson(k, m)
        n_neg = n - a
        e = inv[inv.aoi.isin(aois)]
        flags = int((fl[fl.test == reg].score >= fl[fl.test == reg].threshold).sum())
        rows.append(dict(reach=reg, n_table=n, affected=a, prev_table=a / n,
                         audit_no_building=f"{k}/{m}", nb_share=k / m,
                         prev_less_nb=a / (n - k / m * n_neg), prev_less_nb_lo=a / (n - lo * n_neg),
                         prev_less_nb_hi=a / (n - hi * n_neg),
                         ems_total=int(e.ems_total.sum()), ems_affected=int(e.ems_affected.sum()),
                         prev_ems=e.ems_affected.sum() / e.ems_total.sum(),
                         flagged=flags, flags_per_affected=flags / a))
    r = pd.DataFrame(rows)
    r.to_csv("results/review_denominators.csv", index=False)
    print("\n5. Prevalence under three denominators; flags of the thresholded map per recorded affected building")
    print(r.round(3).to_string(index=False))

def attenuation_support():
    b = pd.read_csv("results/attenuation_bins.csv")
    rows = []
    for target, c in [("affected", 5), ("destroyed", 20)]:
        g = b[b.target == target]
        for name, s in [("all bins", g), ("downstream bins only (>= 40 km)", g[g.km >= 40])]:
            x = np.log2(s.km + c)
            slope = np.polyfit(x, s.h50, 1)[0]
            rows.append(dict(target=target, c_km=c, bins=name, n_bins=len(s), km_min=s.km.min(), km_max=s.km.max(),
                             buildings=int(s.n.sum()), fall_per_doubling_m=-slope, r2=np.corrcoef(x, s.h50)[0, 1] ** 2))
    r = pd.DataFrame(rows)
    r.to_csv("results/review_attenuation_support.csv", index=False)
    print("\n6. Attenuation fit with and without the two upstream bins (log form, c fixed)")
    print(r.round(2).to_string(index=False))

def main():
    df = load()
    attenuation_support()
    cutoffs(df)
    phosretar(df)
    osm_only(df)
    clustered_recal(df)
    denominators(df)

if __name__ == "__main__":
    main()
