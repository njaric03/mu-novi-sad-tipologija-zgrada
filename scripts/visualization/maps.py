import os
import sys

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from pyproj import Transformer
from scripts import config
from scripts.visualization.map_style import add_scale_north, add_basemap

k = int(sys.argv[1]) if len(sys.argv) > 1 else config.K_OP
NAMES, COLORS = config.LABELS[k]["names"], config.LABELS[k]["colors"]
CMAP = ListedColormap([COLORS[c] for c in range(k)])

g = gpd.read_parquet(os.path.join(config.DATA, f"features_building_clustered_k{k}.parquet"))
boundary = gpd.read_file(config.NS_BOUNDARY_GPKG).to_crs(config.CRS_M)
cen = g.geometry.centroid
g["cx"], g["cy"] = cen.x.values, cen.y.values
rare = int(g["cluster"].value_counts().idxmin())  # najređi tip se crta poslednji, na vrhu
print(f"k={k} | {len(g)} zgrada | najređi tip = {rare} ({NAMES[rare]})")
ccx, ccy = Transformer.from_crs(4326, config.CRS_M, always_xy=True).transform(*config.NS_CENTER_WGS)


def decorate(ax, title):
    add_basemap(ax)
    ax.set_axis_off()
    ax.set_title(title, fontsize=16)
    add_scale_north(ax)
    leg = ax.legend(handles=[Patch(facecolor=COLORS[c], label=f"{c}: {NAMES[c]}") for c in sorted(NAMES)],
                    loc="lower right", fontsize=11, framealpha=0.95, title="Tip zgrade", title_fontsize=11)
    leg.set_zorder(5)


def modal_cluster(values):
    # najčešći (modalni) tip zgrada u jednoj heksagonalnoj ćeliji
    counts = np.bincount(np.asarray(values, dtype=int), minlength=k)
    return counts.argmax()


def plot_hex(path, title, gridsize=90):
    # cela teritorija: heksagonalni grid, ćelija obojena po dominantnom tipu
    fig, ax = plt.subplots(figsize=(14, 12))
    xmin, ymin, xmax, ymax = boundary.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.hexbin(g["cx"].values, g["cy"].values, C=g["cluster"].values.astype(int),
              reduce_C_function=modal_cluster,
              gridsize=gridsize, cmap=CMAP,
              vmin=-0.5, vmax=k - 0.5,  # celobrojni id-evi 0..k-1 legnu tačno na k diskretnih boja
              mincnt=1, linewidths=0.15, edgecolors="#ffffff", zorder=2)
    decorate(ax, title)
    fig.savefig(path, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    print("sačuvano", path)


def plot_core(path, title, R=2500):
    # zoom na centar: pojedinačni otisci zgrada, obojeni po tipu
    core = g.cx[ccx - R:ccx + R, ccy - R:ccy + R]
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(ccx - R, ccx + R)
    ax.set_ylim(ccy - R, ccy + R)
    ax.set_aspect("equal")
    # prvo svi tipovi, pa najređi još jednom preko njih (sa ivicom) da se ne izgubi
    for c in sorted(COLORS):
        sub = core[core["cluster"] == c]
        if len(sub):
            sub.plot(ax=ax, color=COLORS[c], linewidth=0, zorder=2)
    rare_sub = core[core["cluster"] == rare]
    if len(rare_sub):
        rare_sub.plot(ax=ax, color=COLORS[rare], linewidth=0.25, edgecolor="#7f0000", zorder=3)
    decorate(ax, title)
    fig.savefig(path, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    print("sačuvano", path)


suf = f"_k{k}"
plot_hex(os.path.join(config.FIGURES, f"typology_full{suf}.png"), f"Novi Sad: tipologija zgrada, cela teritorija (k={k})")
plot_core(os.path.join(config.FIGURES, f"typology_core{suf}.png"), f"Novi Sad, centar: tipologija zgrada (k={k})")
if k == config.K_OP:  # kanonske verzije (bez sufiksa) za notebook/README/rad
    plot_hex(os.path.join(config.FIGURES, "typology_full.png"), f"Novi Sad: tipologija zgrada, cela teritorija (k={k})")
    plot_core(os.path.join(config.FIGURES, "typology_core.png"), f"Novi Sad, centar: tipologija zgrada (k={k})")
