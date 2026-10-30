"""Week 4 - interpretation and error analysis.

1. Standardised logistic-regression coefficients (all data) with 95% block-bootstrap intervals.
2. Permutation importance on the HELD-OUT reaches (drop in ROC-AUC when a feature is shuffled).
3. Partial dependence of the damage probability on key features (logistic regression vs random forest).
4. A depth-3 decision tree as readable rules.
5. Error map and error rates (from 20_final_eval.py out-of-reach predictions).
6. Land-cover check of the DEM-bias hypothesis: are errors more common next to trees, where the
   GLO-30 surface model reads the canopy? Uses ICIMOD NLCMS if data/raw/nlcms/nlcms_2022.tif exists,
   otherwise ESA WorldCover 2021 (10 m, open) as a stand-in.
"""
import os
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from scipy import ndimage
from scipy.stats import chi2, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from clean import load_clean

warnings.filterwarnings("ignore")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
SEED = 42
OUT = pathlib.Path("results")
LANDCOVER = os.environ.get("LANDCOVER", "nlcms")
SUFFIX = "_worldcover" if LANDCOVER == "worldcover" else ""
MAIN = ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "log_valley_w20", "chan_slope", "bend"]
ATTN = ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "logd", "hand_x_logd"]
NLCMS = pathlib.Path("data/raw/nlcms/Land cover of Nepal 2022/data/lc2022.tif")
NLCMS_FOREST = [4]
WORLDCOVER = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
              "ESA_WorldCover_10m_2021_v200_N27E084_Map.tif")
WORLDCOVER_TREE = [10]
BBOX = (84.90, 27.75, 85.45, 28.32)

def logreg():
    return Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))])

def coefficients(df, rng):
    rows = []
    for target, sub, feats in [("affected", df, MAIN), ("destroyed", df, MAIN),
                               ("destroyed | affected", df[df.affected == 1], ATTN)]:
        y = "destroyed" if target.startswith("destroyed") else "affected"
        coef = logreg().fit(sub[feats], sub[y]).named_steps["m"].coef_[0]
        blocks = sub["block"].to_numpy()
        ub = np.unique(blocks)
        members = {b: np.flatnonzero(blocks == b) for b in ub}
        boots = []
        for _ in range(300):
            idx = np.concatenate([members[b] for b in rng.choice(ub, len(ub), replace=True)])
            s = sub.iloc[idx]
            if s[y].nunique() == 2:
                boots.append(logreg().fit(s[feats], s[y]).named_steps["m"].coef_[0])
        lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
        for f, c, l, h in zip(feats, coef, lo, hi):
            rows.append(dict(target=target, feature=f, coef_std=c, ci_low=l, ci_high=h, significant=(l > 0) or (h < 0)))
    return pd.DataFrame(rows)

def perm_importance(df):
    rows = []
    for target, sub, feats, make in [
        ("affected", df, MAIN, logreg),
        ("affected (RF, attenuation)", df, ATTN, lambda: RandomForestClassifier(
            300, max_depth=16, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=SEED)),
        ("destroyed | affected", df[df.affected == 1], ATTN, logreg)]:
        y = "destroyed" if target.startswith("destroyed") else "affected"
        for reg in ["Upper", "Middle", "Lower"]:
            tr, te = sub[sub.region != reg], sub[sub.region == reg]
            m = make().fit(tr[feats], tr[y])
            pi = permutation_importance(m, te[feats], te[y], scoring="roc_auc", n_repeats=10, random_state=SEED,
                                        n_jobs=-1)
            for f, v in zip(feats, pi.importances_mean):
                rows.append(dict(target=target, test=reg, feature=f, auc_drop=v))
    return pd.DataFrame(rows)

def pdp_figure(df):
    feats_show = ["hand", "dist", "chan_slope", "log_valley_w20", "bend", "slope"]
    lr = logreg().fit(df[MAIN], df["affected"])
    rf = RandomForestClassifier(300, max_depth=16, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
                                random_state=SEED).fit(df[MAIN], df["affected"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, f in zip(axes.ravel(), feats_show):
        for m, name, c in [(lr, "Logistic regression", "#1f77b4"), (rf, "Random forest", "#ff7f0e")]:
            pd_ = partial_dependence(m, df[MAIN], [MAIN.index(f)], grid_resolution=40, percentiles=(0.02, 0.98))
            ax.plot(pd_["grid_values"][0], pd_["average"][0], label=name, color=c)
        ax.set_title(f, fontsize=10)
        ax.set_ylabel("P(affected), averaged")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Partial dependence (all data, main feature set, class-weighted models)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_partial_dependence.png", dpi=140)
    plt.close(fig)

def tree_rules(df):
    t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, class_weight="balanced", random_state=SEED)
    t.fit(df[MAIN], df["affected"])
    txt = export_text(t, feature_names=MAIN, decimals=4, show_weights=False)
    (OUT / "tree_rules_affected_main.txt").write_text(txt)
    return txt

def land_cover_share(df):
    """Share of tree/forest cover within ~60 m of each building."""
    if NLCMS.exists() and LANDCOVER != "worldcover":
        src_path, codes, name = NLCMS, NLCMS_FOREST, "ICIMOD NLCMS 2022"
    else:
        src_path, codes, name = WORLDCOVER, WORLDCOVER_TREE, "ESA WorldCover 2021 (stand-in for NLCMS)"
    with rasterio.open(src_path) as r:
        if r.crs.to_epsg() == 4326:
            w = from_bounds(*BBOX, r.transform)
            arr = r.read(1, window=w)
            tr = r.window_transform(w)
            px_m = abs(r.transform.a) * 111_320 * np.cos(np.radians((BBOX[1] + BBOX[3]) / 2))
        else:
            x0, y0 = Transformer.from_crs("EPSG:4326", r.crs, always_xy=True).transform(BBOX[0], BBOX[1])
            x1, y1 = Transformer.from_crs("EPSG:4326", r.crs, always_xy=True).transform(BBOX[2], BBOX[3])
            w = from_bounds(x0, y0, x1, y1, r.transform)
            arr = r.read(1, window=w)
            tr = r.window_transform(w)
            px_m = abs(r.transform.a)
        crs = r.crs
    tree = np.isin(arr, codes).astype("float32")
    size = max(3, int(round(120 / px_m)) | 1)
    share = ndimage.uniform_filter(tree, size=size)
    xs, ys = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(df["lon"].values, df["lat"].values)
    col, row = ~tr * (np.asarray(xs), np.asarray(ys))
    row = np.clip(row.astype(int), 0, share.shape[0] - 1)
    col = np.clip(col.astype(int), 0, share.shape[1] - 1)
    return share[row, col], name

def mantel_haenszel(g, exposure, outcome, strata):
    """Mantel-Haenszel common odds ratio and continuity-corrected chi-square p over strata."""
    num = den = a_obs = a_exp = var = 0.0
    for _, s in g.groupby(strata, observed=True):
        n = len(s)
        if n < 2 or s[exposure].nunique() < 2:
            continue
        a = ((s[exposure] == 1) & (s[outcome] == 1)).sum(); b = ((s[exposure] == 1) & (s[outcome] == 0)).sum()
        c = ((s[exposure] == 0) & (s[outcome] == 1)).sum(); d = ((s[exposure] == 0) & (s[outcome] == 0)).sum()
        num += a * d / n; den += b * c / n
        n1, m1 = a + b, a + c
        a_obs += a; a_exp += n1 * m1 / n; var += n1 * (n - n1) * m1 * (n - m1) / (n * n * (n - 1))
    stat = (abs(a_obs - a_exp) - 0.5) ** 2 / var
    return num / den, chi2.sf(stat, 1)

def main():
    df, _ = load_clean(verbose=False)
    df["region"] = df["region"].str.split(" ").str[0]
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 250)

    co = coefficients(df, rng)
    co.to_csv(OUT / "interp_coefficients.csv", index=False)
    print("Standardised logistic-regression coefficients (95% block-bootstrap CI):")
    co["txt"] = co.apply(lambda r: f"{r.coef_std:+.2f} ({r.ci_low:+.2f},{r.ci_high:+.2f}){'*' if r.significant else ''}", axis=1)
    print(co.pivot(index="feature", columns="target", values="txt").to_string())

    pi = perm_importance(df)
    pi.to_csv(OUT / "interp_permutation.csv", index=False)
    print("\nPermutation importance on held-out reaches (mean drop in ROC-AUC):")
    print(pi.pivot_table(index="feature", columns="target", values="auc_drop", aggfunc="mean").round(3)
          .sort_values("affected", ascending=False).to_string())

    pdp_figure(df)
    print("\nDepth-3 decision tree (affected, main set):\n" + tree_rules(df))

    p = pd.read_csv(OUT / "predictions_loro.csv")
    p = p[(p.target == "affected") & (p.features == "main") & (p.model == "Logistic regression")]
    e = df.merge(p[["bid", "score", "threshold"]], on="bid")
    e["pred"] = (e["score"] >= e["threshold"]).astype(int)
    e["error"] = np.select([(e.pred == 1) & (e.affected == 1), (e.pred == 1) & (e.affected == 0),
                            (e.pred == 0) & (e.affected == 1)], ["TP", "FP", "FN"], "TN")
    e["tree_share"], lc_name = land_cover_share(e)
    e[["bid", "region", "affected", "score", "threshold", "pred", "error", "tree_share", "lon", "lat"]].to_csv(
        OUT / f"errors_logreg_affected{SUFFIX}.csv", index=False)
    print("\nError counts by reach:\n", pd.crosstab(e["region"], e["error"]).to_string())
    for name, g in [("false negatives (missed damage)", e[e.affected == 1]), ("false positives", e[e.affected == 0])]:
        err = g["error"].isin(["FN", "FP"])
        near_tree = g["tree_share"] >= 0.3
        tab = pd.crosstab(near_tree, err)
        p_val = chi2_contingency(tab)[1] if tab.shape == (2, 2) else np.nan
        print(f"{name}: error rate next to trees (>=30% tree cover within ~60 m) "
              f"{err[near_tree].mean():.3f} (n={near_tree.sum()}) vs elsewhere {err[~near_tree].mean():.3f} "
              f"(n={(~near_tree).sum()}); chi-square p = {p_val:.3g}  [{lc_name}]")

    s = e.copy()
    s["near"] = (s["tree_share"] >= 0.3).astype(int)
    s["hb"] = pd.cut(s["hand"], [-1, 10, 20, 30, 45, 60, 90, 1e4])
    s["db"] = pd.cut(s["dist"], [-1, 100, 200, 300, 500, 1e5])
    for name, g, err in [("false positive (unaffected)", s[s.affected == 0], "FP"),
                         ("false negative (affected)", s[s.affected == 1], "FN")]:
        oratio, p_mh = mantel_haenszel(g.assign(y=(g.error == err).astype(int)), "near", "y", ["region", "hb", "db"])
        print(f"Mantel-Haenszel, near trees vs not, {name}: odds ratio {oratio:.2f}, p = {p_mh:.3g}")
    print("Median slope / relief near trees vs not:\n",
          s.groupby(["region", "near"])[["slope", "relief"]].median().round(1).to_string())
    dem = pathlib.Path("results/dem_comparison_buildings.csv")
    if dem.exists():
        d = pd.read_csv(dem).merge(e[["bid", "tree_share"]], on="bid")
        d["bin"] = pd.cut(d["tree_share"], [-0.01, 0.1, 0.3, 0.6, 1.0], labels=["<10%", "10-30%", "30-60%", ">60%"])
        print("\nHAND(FABDEM) - HAND(GLO-30) by tree cover near the building (negative = GLO-30 reads higher):")
        print(d.groupby("bin", observed=True)["diff"].agg(["median", "mean", "count"]).round(2).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    colours = {"TP": "#b3001b", "TN": "#c9ced6", "FP": "#1f77b4", "FN": "#ff9f1c"}
    for ax, reg in zip(axes, ["Upper", "Middle", "Lower"]):
        g = e[e.region == reg]
        for k in ["TN", "TP", "FP", "FN"]:
            s = g[g.error == k]
            ax.scatter(s.lon, s.lat, s=3 if k == "TN" else 6, c=colours[k], label=f"{k} ({len(s)})")
        ax.set_title(f"{reg} reach (model trained on the other two)")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.legend(fontsize=7, markerscale=3)
    fig.suptitle("Out-of-reach errors, logistic regression (affected, main features, tuned threshold)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_error_map.png", dpi=140)

if __name__ == "__main__":
    main()
