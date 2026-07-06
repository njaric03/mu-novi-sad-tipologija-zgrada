import os

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts import config
from scripts.visualization.map_style import add_scale_north, add_basemap

NAMES = config.LABELS[4]["names"]
COLORS = config.LABELS[4]["colors"]

g = gpd.read_parquet(os.path.join(config.DATA, "features_building_clustered_k4.parquet"))
pts = g[["cluster"]].copy()
pts["geometry"] = g.geometry.centroid
pts = gpd.GeoDataFrame(pts, crs=g.crs)
boundary = gpd.read_file(config.NS_BOUNDARY_GPKG).to_crs(config.CRS_M)
xmin, ymin, xmax, ymax = boundary.total_bounds

# svaki tip na zasebnom panelu, u istom opsegu, radi poređenja prostornog otiska
fig, axes = plt.subplots(2, 2, figsize=(18, 15))
for c, ax in zip(sorted(NAMES), axes.ravel()):
    sub = pts[pts["cluster"] == c]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    sub.plot(ax=ax, color=COLORS[c], markersize=1.5, linewidth=0, zorder=2)
    add_basemap(ax, attribution=False)
    ax.set_axis_off()
    ax.set_title(f"{c}: {NAMES[c]}  (n={len(sub)})", fontsize=14, color=COLORS[c])
    add_scale_north(ax)
fig.suptitle("Prostorni otisak svakog tipa zasebno (k=4)", fontsize=17)
fig.savefig(os.path.join(config.FIGURES, "small_multiples_k4.png"), dpi=config.DPI, bbox_inches="tight")
plt.close(fig)
print("sačuvano figures/small_multiples_k4.png")
