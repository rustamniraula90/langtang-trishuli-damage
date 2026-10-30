"""Shared helpers for the Week 4 validation add-ons (24_knndm, 25_aoa, 26_calibration).

kNNDM (Linnenbrink et al. 2024, GMD 17:5897, doi:10.5194/gmd-17-5897-2024) builds k CV folds whose
fold-to-training nearest-neighbour distances match the distances from the prediction locations to the
training data. It is a Python re-implementation of the idea in R CAST::knndm (no Python package exists):
  * G_ij = NN distance from each prediction location to the training points;
  * for each candidate q, k-means on the training coordinates gives q clusters; clusters are merged into
    k folds greedily (largest cluster to smallest fold), with no fold above max_frac of the data;
  * G_j* = NN distance from each training point to the training points in other folds;
  * keep the q whose G_j* is closest to G_ij in Wasserstein-1 distance (W).
Random k-fold (q = n) is always a candidate, as in CAST.

Area of applicability (Meyer & Pebesma 2021, Methods Ecol. Evol. 12:1620; discussed in Meyer & Pebesma
2022, Nat. Commun. 13:2208): dissimilarity index DI = distance to the nearest training point in the
importance-weighted, standardised feature space, divided by the mean pairwise training distance. The AOA
threshold is the upper whisker (Q75 + 1.5 IQR, capped at the max) of the training DI, where each training
point is compared only with training points in other CV folds.
"""
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans

_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)

def xy(df):
    x, y = _UTM.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    return np.column_stack([x, y])

def nn_dist(a, b):
    return cKDTree(b).query(a, k=1)[0]

def fold_nn_dist(xy_tr, folds):
    d = np.empty(len(xy_tr))
    for f in np.unique(folds):
        m = folds == f
        d[m] = nn_dist(xy_tr[m], xy_tr[~m])
    return d

def merge_clusters(labels, k, max_frac=0.5):
    sizes = np.bincount(labels)
    fold_of, fold_n = np.empty(len(sizes), int), np.zeros(k)
    cap = max_frac * len(labels)
    for c in np.argsort(-sizes):
        order = np.argsort(fold_n)
        f = next((f for f in order if fold_n[f] + sizes[c] <= cap), order[0])
        fold_of[c], fold_n[f] = f, fold_n[f] + sizes[c]
    return fold_of[labels]

def knndm(xy_tr, xy_pred, k=3, qs=None, max_frac=0.5, seed=0):
    """Return (folds, W, q, table of candidates)."""
    rng = np.random.default_rng(seed)
    g_ij = nn_dist(xy_pred, xy_tr)
    qs = qs if qs is not None else sorted(set(np.unique(np.geomspace(k, 300, 15).astype(int))))
    cands = [("random", None, rng.integers(0, k, len(xy_tr)))]
    for q in qs:
        lab = KMeans(q, n_init=3, random_state=seed).fit_predict(xy_tr)
        cands.append(("kmeans", q, merge_clusters(lab, k, max_frac)))
    table = []
    for kind, q, folds in cands:
        if np.bincount(folds, minlength=k).max() > max_frac * len(folds) + 1 and kind != "random":
            continue
        w = wasserstein_distance(fold_nn_dist(xy_tr, folds), g_ij)
        table.append((kind, q, w, folds))
    best = min(table, key=lambda t: t[2])
    return best[3], best[2], best[1], [(t[0], t[1], t[2]) for t in table]

def dissimilarity(Xtr, Xnew, weights, folds, n_pairs=3000, seed=0):
    """Return (DI_new, DI_train, threshold). Xtr, Xnew are already standardised with the training stats."""
    w = np.clip(np.asarray(weights, float), 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    A, B = Xtr * w, Xnew * w
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(A), (2, n_pairs))
    dbar = np.linalg.norm(A[i] - A[j], axis=1)[i != j].mean()
    di_new = cKDTree(A).query(B, k=1)[0] / dbar
    di_tr = np.empty(len(A))
    for f in np.unique(folds):
        m = folds == f
        di_tr[m] = cKDTree(A[~m]).query(A[m], k=1)[0] / dbar
    q1, q3 = np.percentile(di_tr, [25, 75])
    thr = min(q3 + 1.5 * (q3 - q1), di_tr.max())
    return di_new, di_tr, thr
