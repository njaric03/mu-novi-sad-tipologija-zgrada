import os

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyproj import Transformer
from scripts import config
from scripts.visualization.map_style import add_scale_north, add_basemap

# kategorije sadržaja: boja + čitljivo ime (redosled = redosled u legendi)
CATS = {
    "grocery":   ("#e41a1c", "prodavnice prehrane"),
    "food":      ("#ff7f00", "ugostiteljstvo"),
    "education": ("#377eb8", "obrazovanje"),
    "health":    ("#4daf4a", "zdravstvo"),
    "civic":     ("#984ea3", "javne službe"),
    "office":    ("#a65628", "poslovni prostori"),
    "leisure":   ("#f781bf", "rekreacija"),
    "retail":    ("#999999", "ostala trgovina"),
}

pois = gpd.read_parquet(os.path.join(config.DATA, "ns_pois.parquet"))
edges = gpd.read_parquet(os.path.join(config.DATA, "ns_road_edges.parquet"))
ccx, ccy = Transformer.from_crs(4326, config.CRS_M, always_xy=True).transform(*config.NS_CENTER_WGS)
print(f"POI: {len(pois)} | ivica mreže: {len(edges)}")

fig, ax = plt.subplots(figsize=(11, 11))

# kategorije sadržaja u užem jezgru preko ulične mreže (ilustruje poi_mix i retail_ratio)
R = 2500
ax.set_xlim(ccx - R, ccx + R)
ax.set_ylim(ccy - R, ccy + R)
ax.set_aspect("equal")
edges.plot(ax=ax, color="#9a9a9a", linewidth=0.5, zorder=1)
for cat, (color, _) in CATS.items():
    sub = pois[pois.cat == cat]
    if len(sub):
        sub.plot(ax=ax, color=color, markersize=12, linewidth=0, zorder=2)
add_basemap(ax)
ax.set_axis_off()
add_scale_north(ax)

handles = [Line2D([], [], marker="o", linestyle="", color=color,
                  markersize=6, label=f"{name} ({(pois.cat == cat).sum()})")
           for cat, (color, name) in CATS.items()]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11, frameon=True,
           title="Kategorija sadržaja (broj)", title_fontsize=11)
fig.suptitle("Sadržaji (POI) iz OpenStreetMap-a, uže gradsko jezgro", fontsize=16)
fig.savefig(os.path.join(config.FIGURES, "poi_karta.png"), dpi=config.DPI, bbox_inches="tight")
plt.close(fig)
print("sačuvano figures/poi_karta.png")
