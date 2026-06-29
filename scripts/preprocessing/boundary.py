import json

import geopandas as gpd
import osmnx as ox
from scripts import config

# granica Grada NS iz OSM/Nominatim, u metarsku projekciju (UTM 34N)
print(f"geocoding: {config.NS_PLACE!r} (OSM / Nominatim) ...")
gdf = ox.geocode_to_gdf(config.NS_PLACE).to_crs(config.CRS_M)

# popravi nevalidne delove pa ih spoji u jednu geometriju grada
ns_geom = gdf.geometry.make_valid().union_all()

boundary = gpd.GeoDataFrame(geometry=[ns_geom], crs=config.CRS_M)
boundary.to_file(config.NS_BOUNDARY_GPKG, driver="GPKG")

# bbox u dve projekcije: WGS za upite (buildings, satellite), UTM za analizu
bbox_wgs = [float(x) for x in boundary.to_crs(config.CRS_WGS).total_bounds]
bbox_utm = [float(x) for x in boundary.total_bounds]
display_name = str(gdf.iloc[0].get("display_name", ""))
with open(config.NS_BBOX_JSON, "w", encoding="utf-8") as f:
    json.dump({"bbox_wgs": bbox_wgs, "bbox_utm": bbox_utm, "place": config.NS_PLACE, "display_name": display_name}, f, indent=2)

# provera: pravo mesto i realna površina (~700 km2 za ceo NS)
print(f"NS: {display_name} | {ns_geom.area / 1e6:.1f} km^2")
