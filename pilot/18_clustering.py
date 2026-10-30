"""Week 3 - damage regimes among affected buildings (syllabus: K-means and hierarchical clustering).

Unsupervised: cluster AFFECTED buildings on standardised terrain features (main set without labels),
then describe each cluster (median features, share destroyed, reach composition).
k for K-means is chosen by silhouette score; Ward hierarchical clustering on a sample is compared
with K-means using the adjusted Rand index.
Outputs results/clusters_profile.csv, results/fig_clusters.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from clean import load_clean

SEED = 42
FEATS = ["hand", "dist", "slope", "relief", "curv_profile", "curv_plan", "twi", "log_valley_w20", "chan_slope", "bend"]

def main():
    df, _ = load_clean(verbose=False)
    a = df[df.affected == 1].copy()
    Z = StandardScaler().fit_transform(a[FEATS])
    rng = np.random.default_rng(SEED)
    samp = rng.choice(len(a), size=min(2500, len(a)), replace=False)

    sil = {}
    for k in range(2, 7):
        lab = KMeans(k, n_init=20, random_state=SEED).fit_predict(Z)
        sil[k] = silhouette_score(Z[samp], lab[samp])
    k_best = max(sil, key=sil.get)
    km = KMeans(k_best, n_init=20, random_state=SEED).fit(Z)
    a["cluster"] = km.labels_
    print("silhouette by k:", {k: round(v, 3) for k, v in sil.items()}, "-> k =", k_best)

    link = linkage(Z[samp], method="ward")
    ward = fcluster(link, k_best, criterion="maxclust")
    ari = adjusted_rand_score(km.labels_[samp], ward)
    print(f"agreement K-means vs Ward (adjusted Rand index, sample of {len(samp)}): {ari:.3f}")

    prof = a.groupby("cluster").agg(n=("bid", "size"), share_destroyed=("destroyed", "mean"),
                                    **{f: (f, "median") for f in FEATS})
    comp = pd.crosstab(a["cluster"], a["region"].str.split(" ").str[0], normalize="index").round(2)
    prof = prof.join(comp)
    prof.to_csv("results/clusters_profile.csv")
    pd.set_option("display.width", 250)
    print(prof.round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(list(sil), list(sil.values()), marker="o")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("silhouette"); axes[0].set_title("Choosing k (K-means)")
    centres = pd.DataFrame(km.cluster_centers_, columns=FEATS)
    im = axes[1].imshow(centres.T, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    axes[1].set_yticks(range(len(FEATS)), FEATS, fontsize=8)
    axes[1].set_xticks(range(k_best), [f"C{i}\n{prof.loc[i, 'share_destroyed']:.0%} destr." for i in range(k_best)],
                       fontsize=8)
    axes[1].set_title("Cluster centres (standardised)")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    dendrogram(link, truncate_mode="lastp", p=30, ax=axes[2], no_labels=True, color_threshold=None)
    axes[2].set_title(f"Ward dendrogram (sample of {len(samp)})")
    fig.tight_layout()
    fig.savefig("results/fig_clusters.png", dpi=140)

if __name__ == "__main__":
    main()
