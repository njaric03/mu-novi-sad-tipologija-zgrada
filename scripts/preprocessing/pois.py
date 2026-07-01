import os

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from scripts import config

RAW_POIS = os.path.join(config.DATA, "ns_pois_raw.parquet")
poly = gpd.read_file(config.NS_BOUNDARY_GPKG).to_crs(config.CRS_WGS).geometry.iloc[0]

# OSM oznake grupisane u kategorije sadržaja koje nas zanimaju
GROCERY = {"supermarket", "convenience", "greengrocer", "bakery", "butcher", "deli"}
FOOD = {"restaurant", "cafe", "fast_food", "bar", "pub", "food_court", "ice_cream"}
EDU = {"school", "university", "college", "kindergarten", "childcare"}
HEALTH = {"hospital", "clinic", "doctors", "pharmacy", "dentist", "veterinary"}
CIVIC = {"townhall", "police", "courthouse", "community_centre", "library", "place_of_worship",
         "fire_station", "post_office"}
TAGCOLS = ["shop", "amenity", "office", "leisure"]

# tačke sadržaja iz OSM se preuzimaju samo prvi put
if os.path.exists(RAW_POIS):
    g = gpd.read_parquet(RAW_POIS)
else:
    print("preuzimanje sadržaja (shop/amenity/office/leisure) ...")
    g = ox.features_from_polygon(poly, {c: True for c in TAGCOLS}).to_crs(config.CRS_M)
    g["geometry"] = g.geometry.representative_point()
    keep = ["geometry"] + [c for c in TAGCOLS if c in g.columns]
    g = gpd.GeoDataFrame(g[keep].reset_index(drop=True), crs=config.CRS_M)
    g.to_parquet(RAW_POIS, index=False)

# tag-kolone koje ne postoje dodaju se kao prazan string
for c in TAGCOLS:
    g[c] = g[c].astype("string").fillna("") if c in g.columns else ""

# svaka tačka se svrstava u jednu kategoriju (prvi uslov koji se poklopi)
shop, amen, off, leis = g["shop"], g["amenity"], g["office"], g["leisure"]
conds = [shop.isin(GROCERY), amen.isin(FOOD), amen.isin(EDU), amen.isin(HEALTH),
         amen.isin(CIVIC), (amen == "bank") | (off != ""), leis != "", shop != ""]
labels = ["grocery", "food", "education", "health", "civic", "office", "leisure", "retail"]
g["cat"] = pd.Series(np.select(conds, labels, default=""), index=g.index, dtype="string").replace("", pd.NA)

pois = g[g["cat"].notna()][["cat", "geometry"]].reset_index(drop=True)
pois.to_parquet(os.path.join(config.DATA, "ns_pois.parquet"), index=False)
print("sadržaja:", len(pois))
print(pois["cat"].value_counts().to_string())
