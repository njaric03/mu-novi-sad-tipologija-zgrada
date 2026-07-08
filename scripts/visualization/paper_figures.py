import os

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts import config
from scripts.visualization.map_style import add_scale_north, add_basemap

NAMES_K4 = config.LABELS[4]["names"]

g = gpd.read_parquet(os.path.join(config.DATA, "features_building.parquet"))
k2 = gpd.read_parquet(os.path.join(config.DATA, "features_building_clustered_k2.parquet"))[["bid", "cluster"]]
k3 = gpd.read_parquet(os.path.join(config.DATA, "features_building_clustered_k3.parquet"))[["bid", "cluster"]]
k4 = gpd.read_parquet(os.path.join(config.DATA, "features_building_clustered_k4.parquet"))[["bid", "cluster"]]
df = g.drop(columns="geometry").merge(k2.rename(columns={"cluster": "k2"}), on="bid").merge(
    k3.rename(columns={"cluster": "k3"}), on="bid").merge(
    k4.rename(columns={"cluster": "k4"}), on="bid")

LEAN = [c for c in g.columns if c not in ("bid", "class", "subtype", "geometry")
        + config.REDUNDANT_FEATURES and g[c].dtype.kind in "fi"]
X = df[LEAN].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
Z = pd.DataFrame(StandardScaler().fit_transform(X), columns=LEAN, index=df.index)

print("\n######## 1. UDELI PO KLASTERU ########")
print("-- k=2 --"); print((df["k2"].value_counts(normalize=True).sort_index() * 100).round(1).astype(str) + " %")
print("-- k=4 --"); print((df["k4"].value_counts(normalize=True).sort_index() * 100).round(1).astype(str) + " %")

print("\n######## 2. Z-PROFIL k=4 (srednji z-skor po klasteru) ########")
zprof = Z.groupby(df["k4"]).mean().round(2)
print(zprof.T.to_string())

print("\n######## 3. CROSSTAB k=4 x Overture subtype ########")
if "subtype" in df.columns:
    st = df["subtype"].fillna("None")
    print("-- broj --"); print(pd.crosstab(df["k4"], st).to_string())
    print("-- normalizovano po redu --"); print(pd.crosstab(df["k4"], st, normalize="index").round(3).to_string())

print("\n######## 4. k=3: udeli + preslikavanje na k=4 ########")
print("-- k=3 udeli --")
print((df["k3"].value_counts(normalize=True).sort_index() * 100).round(1).astype(str) + " %")
print("-- crosstab k=3 (red) x k=4 (kolona), broj --")
print(pd.crosstab(df["k3"], df["k4"]).to_string())
print("-- normalizovano po k=4 koloni (koji k=4 tip ide u koji k=3 klaster) --")
print(pd.crosstab(df["k3"], df["k4"], normalize="columns").round(3).to_string())

# slika A: profil tipova (heatmap)
order = [0, 1, 2, 3]  # kolone po indeksu klastera
H = zprof.loc[order].T
fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(order) + 4), 0.42 * len(LEAN) + 1.5))
im = ax.imshow(H.values, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"{c}: {NAMES_K4[c]}" for c in order], rotation=20, ha="right", fontsize=10)
ax.set_yticks(range(len(LEAN)))
ax.set_yticklabels(LEAN, fontsize=9)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f"{H.values[i, j]:.1f}", ha="center", va="center", fontsize=7,
                color="black" if abs(H.values[i, j]) < 1.3 else "white")
ax.set_title("Profil tipova (k=4): srednji z-skor atributa po klasteru", fontsize=13)
fig.colorbar(im, ax=ax, label="z-skor (crveno iznad proseka, plavo ispod)", fraction=0.046, pad=0.04)
fig.savefig(os.path.join(config.FIGURES, "profile_heatmap.png"), dpi=config.DPI, bbox_inches="tight")
plt.close(fig)
print("\nsačuvano figures/profile_heatmap.png")

# slika B: korelaciona matrica (svih 21 atribut)
ALL = [c for c in g.columns if c not in ("bid", "class", "subtype", "geometry") and g[c].dtype.kind in "fi"]
Xa = df[ALL].replace([np.inf, -np.inf], np.nan)
Xa = Xa.fillna(Xa.median(numeric_only=True))
corr = Xa.corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(ALL)))
ax.set_xticklabels(ALL, rotation=90, fontsize=8)
ax.set_yticks(range(len(ALL)))
ax.set_yticklabels(ALL, fontsize=8)
ax.set_title("Korelaciona matrica atributa (Pearson)", fontsize=13)
fig.colorbar(im, ax=ax, label="Pirsonov r", fraction=0.046, pad=0.04)
fig.savefig(os.path.join(config.FIGURES, "korelacije_atributa.png"), dpi=config.DPI, bbox_inches="tight")
plt.close(fig)
print("sačuvano figures/korelacije_atributa.png")

# slika C: zelenilo okruženja (mean NDVI po hex ćeliji) na basemap podlozi
bnd = gpd.read_file(config.NS_BOUNDARY_GPKG).to_crs(config.CRS_M)
cen = g.geometry.centroid
xx, yy, nd = cen.x.values, cen.y.values, g["ndvi"].values
vmin, vmax = float(np.quantile(nd, 0.02)), float(np.quantile(nd, 0.98))
fig, ax = plt.subplots(figsize=(14, 12))
xmin, ymin, xmax, ymax = bnd.total_bounds
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")
hb = ax.hexbin(xx, yy, C=nd, reduce_C_function=np.mean, gridsize=90, cmap="RdYlGn",
               vmin=vmin, vmax=vmax, mincnt=1, linewidths=0.15, edgecolors="#ffffff", zorder=2)
add_basemap(ax)
ax.set_axis_off()
ax.set_title("Novi Sad: zelenilo okruženja zgrada (mean NDVI, ~300 m)", fontsize=15)
add_scale_north(ax)
cbar = fig.colorbar(hb, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label("NDVI (zelenilo okoline)")
fig.savefig(os.path.join(config.FIGURES, "zelenilo_okruzenja.png"), dpi=config.DPI, bbox_inches="tight")
plt.close(fig)
print("sačuvano figures/zelenilo_okruzenja.png")
print("gotovo")
