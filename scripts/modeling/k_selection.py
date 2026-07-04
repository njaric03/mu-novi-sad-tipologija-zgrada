import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts import config

# --full: ista analiza na punom skupu atributa (bez izbacivanja redundantnih),
# za proveru osetljivosti izbora k na sastav skupa; izlazi dobijaju sufiks _full
FULL = "--full" in sys.argv
SUF = "_full" if FULL else ""
DROP = () if FULL else config.REDUNDANT_FEATURES

g = gpd.read_parquet(os.path.join(config.DATA, "features_building.parquet"))
FEAT = [c for c in g.columns
        if c not in ("bid", "class", "subtype", "geometry") + DROP
        and g[c].dtype.kind in "fi"]

X = g[FEAT].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
Z = PCA(n_components=0.95).fit_transform(StandardScaler().fit_transform(X))
print(f"atributa {len(FEAT)} | PCA komponenti {Z.shape[1]}\n")

# 30k za dendrogram, 10k za siluetu; jedan generator, fiksan redosled izvlačenja
rng = np.random.default_rng(0)
idx30 = rng.choice(len(Z), min(config.DENDROGRAM_SAMPLE, len(Z)), replace=False)
sidx = rng.choice(len(Z), min(10000, len(Z)), replace=False)


def ward_gaps(link):
    # iz poslednjih 10 spajanja: k iz najvećeg i drugog najvećeg skoka;
    # gleda se samo vrh hijerarhije, pa je kandidat ograničen na k <= 10
    top = link[:, 2][-10:]
    order = np.argsort(np.diff(top))[::-1]
    return len(top) - int(order[0]), len(top) - int(order[1]), top


print("=== 1. broj klastera: Ward dendrogram ===")
ward = linkage(Z[idx30], method="ward")
figd, axd = plt.subplots(figsize=(10, 4))
dendrogram(ward, truncate_mode="lastp", p=30, no_labels=True, ax=axd)
axd.set_title(f"Ward dendrogram (uzorak n={len(idx30)})")
axd.set_xlabel("grana (truncate: lastp=30)")
axd.set_ylabel("rastojanje spajanja")
figd.savefig(os.path.join(config.FIGURES, f"dendrogram{SUF}.png"), dpi=config.DPI, bbox_inches="tight")
plt.close(figd)
k_top, k_2nd, top = ward_gaps(ward)
print("poslednjih 10 rastojanja spajanja:", "  ".join(f"{h:.1f}" for h in top))
print(f"najveći skok -> k={k_top}; drugi po veličini -> k={k_2nd}")
print(f"sačuvano figures/dendrogram{SUF}.png\n")

N_STAB = 20
print(f"=== 1b. stabilnost izbora k ({N_STAB} nezavisnih uzoraka) ===")
rows, k1s, k2s = [], [], []
for s in range(N_STAB):
    seed = s + 1
    ix = np.random.default_rng(seed).choice(len(Z), min(config.DENDROGRAM_SAMPLE, len(Z)), replace=False)
    k1, k2, _ = ward_gaps(linkage(Z[ix], method="ward"))
    rows.append((s, len(ix), k1, k2))
    k1s.append(k1)
    k2s.append(k2)
    print(f"  uzorak {s} (n={len(ix)}): najveći skok -> k={k1}, drugi -> k={k2}")
d1, d2 = dict(sorted(Counter(k1s).items())), dict(sorted(Counter(k2s).items()))
print(f"  raspodela k (najveći skok): {d1}")
print(f"  raspodela k (drugi skok):   {d2}")
pd.DataFrame(rows, columns=["sample", "n", "k_largest_gap", "k_second_gap"]).to_csv(
    os.path.join(config.RESULTS, f"k_selection_stability{SUF}.csv"), index=False)
with open(os.path.join(config.RESULTS, f"k_selection_stability{SUF}.txt"), "w", encoding="utf-8") as f:
    f.write(f"Ward stabilnost izbora k ({len(FEAT)} atributa): "
            f"{N_STAB} nezavisnih uzoraka, n={config.DENDROGRAM_SAMPLE}\n")
    f.write(f"raspodela k (najveci skok):  {d1}\n")
    f.write(f"raspodela k (drugi skok):    {d2}\n")
print(f"sačuvano results/k_selection_stability{SUF}.csv + .txt\n")

print("=== 2. (opisno) elbow + silueta po k ===")
ks = list(range(2, 11))
inertia, sils = [], []
for k in ks:
    m = KMeans(n_clusters=k, random_state=0, n_init=5).fit(Z)
    inertia.append(m.inertia_)
    sils.append(silhouette_score(Z[sidx], m.labels_[sidx]))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ks, inertia, "o-")
ax[0].set_title("Elbow (inercija)")
ax[0].set_xlabel("k")
ax[0].set_ylabel("inercija")
ax[1].plot(ks, sils, "o-")
ax[1].axvline(config.K_OP, color="r", ls="--", lw=1, label=f"usvojeno k={config.K_OP}")
ax[1].legend(fontsize=8)
ax[1].set_title("Silueta po k (opisno)")
ax[1].set_xlabel("k")
ax[1].set_ylabel("silueta")
fig.savefig(os.path.join(config.FIGURES, f"k_selection{SUF}.png"), dpi=config.DPI, bbox_inches="tight")
plt.close(fig)
print("k        :", "  ".join(f"{k:5d}" for k in ks))
print("silueta  :", "  ".join(f"{s:5.3f}" for s in sils))
print(f"sačuvano figures/k_selection{SUF}.png")
