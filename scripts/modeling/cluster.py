import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scripts import config

g = gpd.read_parquet(os.path.join(config.DATA, "features_building.parquet"))
# numerički atributi za grupisanje (bez id/labela i redundantnih)
EXCLUDE = ("bid", "class", "subtype", "geometry")
FEAT = [c for c in g.columns if c not in EXCLUDE + config.REDUNDANT_FEATURES and g[c].dtype.kind in "fi"]
print(f"redova {len(g)} | atributa {len(FEAT)}")

# standardizacija, pa PCA (95% varijanse), pa k-means
X = g[FEAT].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
Xs = StandardScaler().fit_transform(X)

pca = PCA(n_components=0.95).fit(Xs)
Z = pca.transform(Xs)
pd.DataFrame({"pc": range(1, Z.shape[1] + 1),
             "explained": pca.explained_variance_ratio_}).to_csv(
    os.path.join(config.RESULTS, "pca_explained.csv"), index=False)
print(f"PCA: {Z.shape[1]} komponenti, {pca.explained_variance_ratio_.sum():.2f} varijanse")

k = int(sys.argv[1]) if len(sys.argv) > 1 else config.K_OP
km = KMeans(n_clusters=k, random_state=0, n_init=5).fit(Z)
g["cluster"] = km.labels_
print(f"KMeans k={k}")

for suf in (f"_k{k}", ""):  # sa sufiksom (poređenje k) + kanonski (bez sufiksa)
    g.groupby("cluster")[FEAT].mean().round(3).to_csv(os.path.join(config.RESULTS, f"cluster_profiles{suf}.csv"))
    g.groupby("cluster").size().rename("n").to_csv(os.path.join(config.RESULTS, f"cluster_sizes{suf}.csv"))
if "subtype" in g.columns:
    print("\nklaster x Overture subtype:")
    print(pd.crosstab(g.cluster, g.subtype.fillna("None")).to_string())

# upis sa sufiksom (poređenje k) i kanonski (za notebook/mape)
out_geo = g[["bid", "cluster", "geometry"]]
out_geo.to_parquet(os.path.join(config.DATA, f"features_building_clustered_k{k}.parquet"), index=False)
out_geo.to_parquet(os.path.join(config.DATA, "features_building_clustered.parquet"), index=False)
print("\nveličine klastera:")
print(g.groupby("cluster").size().to_string())
