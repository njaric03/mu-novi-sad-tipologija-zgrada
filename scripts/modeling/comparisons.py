import os

import numpy as np
import geopandas as gpd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scripts import config

g = gpd.read_parquet(os.path.join(config.DATA, "features_building.parquet"))
FEAT = [c for c in g.columns
        if c not in ("bid", "class", "subtype", "geometry") + config.REDUNDANT_FEATURES
        and g[c].dtype.kind in "fi"]


def scale(df, cols):
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return StandardScaler().fit_transform(X)


Xs = scale(g, FEAT)
Z = PCA(n_components=0.95).fit_transform(Xs)
d = Z.shape[1]

# isti 30k/10k podskupovi (i redosled izvlačenja) kao u k_selection.py, da su brojevi uporedivi
rng = np.random.default_rng(0)
idx30 = rng.choice(len(Z), min(config.DENDROGRAM_SAMPLE, len(Z)), replace=False)
sidx = rng.choice(len(Z), min(10000, len(Z)), replace=False)


def km(Z_, k):
    return KMeans(n_clusters=k, random_state=0, n_init=5).fit(Z_)


def sil(Z_, labels):
    return silhouette_score(Z_[sidx], labels[sidx])


print(f"atributa {len(FEAT)} | PCA komponenti {d}\n")

print("=== 1. Ward vs k-means (uzorak 30k, ARI po k) ===")
ward = linkage(Z[idx30], method="ward")
for kk in (2, 3, 4, 5):
    wl = fcluster(ward, t=kk, criterion="maxclust")
    kl = km(Z[idx30], kk).labels_
    print(f"  k={kk}: ARI = {adjusted_rand_score(wl, kl):.3f}")
print()

print(f"=== 2. redukcija dimenzionalnosti i podskup atributa (k=2 i k={config.K_OP}) ===")
rp = GaussianRandomProjection(n_components=d, random_state=0).fit_transform(Xs)
full = [c for c in g.columns if c not in ("bid", "class", "subtype", "geometry") and g[c].dtype.kind in "fi"]
Zf = PCA(n_components=0.95).fit_transform(scale(g, full))
print(f"  pun skup ({len(full)} atributa): {Zf.shape[1]} PCA komponenti")
for kk in (2, config.K_OP):
    lp, lr, lf = km(Z, kk).labels_, km(rp, kk).labels_, km(Zf, kk).labels_
    print(f"  k={kk}: silueta PCA/uzak({len(FEAT)}) {sil(Z, lp):.3f} | RandProj {sil(rp, lr):.3f} | "
          f"pun-{len(full)} {sil(Zf, lf):.3f} | ARI PCA vs RandProj {adjusted_rand_score(lp, lr):.3f} | "
          f"ARI uzak vs pun {adjusted_rand_score(lp, lf):.3f}")
print("  (siluete su merene u tri različita prostora, pa nisu direktno uporedive; merodavan je ARI)\n")

print(f"=== 3. konfiguracije k-means (PCA, k={config.K_OP}) ===")
for init in ["k-means++", "random"]:
    for n_init in [1, 10]:
        m = KMeans(n_clusters=config.K_OP, init=init, n_init=n_init, random_state=0).fit(Z)
        print(f"  init={init:10s} n_init={n_init:2d}  silueta={sil(Z, m.labels_):.3f}")
