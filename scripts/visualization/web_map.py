import os
from collections import defaultdict

import numpy as np
import geopandas as gpd
from shapely.geometry import box
import folium
from scripts import config

NAMES = config.LABELS[4]["names"]
COLORS = config.LABELS[4]["colors"]
CELL = 400.0  # m; veličina ćelije mreže za agregaciju (lakši interaktivni HTML)

g = gpd.read_parquet(os.path.join(config.DATA, "features_building_clustered_k4.parquet"))
cen = g.geometry.centroid
x, y, cl = cen.x.values, cen.y.values, g["cluster"].values.astype(int)
x0, y0 = x.min(), y.min()
ix = ((x - x0) // CELL).astype(int)
iy = ((y - y0) // CELL).astype(int)

# zgrade se grupišu po ćeliji mreže, ćelija se boji po dominantnom tipu
cells = defaultdict(list)
for cell_x, cell_y, clu in zip(ix, iy, cl):
    cells[(cell_x, cell_y)].append(clu)

recs = []
for (cell_x, cell_y), vals in cells.items():
    modal = int(np.bincount(vals, minlength=4).argmax())
    gx, gy = x0 + cell_x * CELL, y0 + cell_y * CELL
    recs.append({"cluster": modal, "tip": NAMES[modal], "broj_zgrada": len(vals),
                 "geometry": box(gx, gy, gx + CELL, gy + CELL)})
grid = gpd.GeoDataFrame(recs, crs=config.CRS_M).to_crs(config.CRS_WGS)
print(f"mreža {CELL:.0f} m: {len(grid)} popunjenih ćelija")

m = folium.Map(location=[config.NS_CENTER_WGS[1], config.NS_CENTER_WGS[0]], zoom_start=12,
               tiles="CartoDB positron", control_scale=True)


def style(feat):
    # tanka ivica iste boje pokriva antialiasing šavove između susednih ćelija
    c = feat["properties"]["cluster"]
    return {"fillColor": COLORS[c], "color": COLORS[c], "weight": 0.6, "fillOpacity": 0.72}


# svaki tip kao zaseban sloj, pa se u LayerControl mogu paliti/gasiti pojedinačno;
# smooth_factor=0 sprečava Leaflet da uprošćava male poligone pri zumiranju
for c in sorted(NAMES):
    sub = grid[grid["cluster"] == c]
    folium.GeoJson(
        sub.__geo_interface__, name=f"{c}: {NAMES[c]}", style_function=style, smooth_factor=0,
        tooltip=folium.GeoJsonTooltip(fields=["tip", "broj_zgrada"], aliases=["Tip:", "Broj zgrada:"]),
    ).add_to(m)

legend = ('<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;'
          'padding:10px 14px;border:1px solid #999;border-radius:6px;font-size:13px;'
          'font-family:sans-serif"><b>Tip zgrade (k=4)</b><br>'
          + "".join(f'<span style="color:{COLORS[c]};font-size:16px">&#9632;</span> {c}: {NAMES[c]}<br>'
                    for c in sorted(NAMES)) + '</div>')
m.get_root().html.add_child(folium.Element(legend))
folium.LayerControl().add_to(m)

out = os.path.join(config.FIGURES, "tipologija_interaktivna.html")
m.save(out)
print("sačuvano", out)
