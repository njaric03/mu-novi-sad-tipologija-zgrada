import os

import geopandas as gpd
import osmnx as ox
from scripts import config

GRAPH = os.path.join(config.DATA, "ns_drive.graphml")
poly = gpd.read_file(config.NS_BOUNDARY_GPKG).to_crs(config.CRS_WGS).geometry.iloc[0]

# ulična mreža (vozni graf) iz OSM se preuzima samo prvi put
if os.path.exists(GRAPH):
    G = ox.load_graphml(GRAPH)
else:
    print("preuzimanje ulične mreže ...")
    G = ox.graph_from_polygon(poly, network_type="drive")
    ox.save_graphml(G, GRAPH)

# graf se pretvara u tabele čvorova (raskrsnice) i ivica (segmenti), u metarskoj projekciji;
# graf je usmeren, pa dvosmerne ulice daju dve ivice i zbir dužina je veći od stvarne mreže,
# ali je road_density zbog toga samo uporediva mera među zgradama, ne apsolutna dužina
G = ox.project_graph(G, to_crs=config.CRS_M)
nodes, edges = ox.graph_to_gdfs(G)
nodes = nodes.reset_index()[["street_count", "geometry"]]
edges = edges.reset_index()
edges = edges[[c for c in ["length", "highway", "geometry"] if c in edges.columns]]
# highway ume da bude lista (segment sa više oznaka), pa se spaja u jedan string
edges["highway"] = edges["highway"].apply(
    lambda v: ";".join(map(str, v)) if isinstance(v, list) else ("" if v is None else str(v)))

nodes.to_parquet(os.path.join(config.DATA, "ns_road_nodes.parquet"), index=False)
edges.to_parquet(os.path.join(config.DATA, "ns_road_edges.parquet"), index=False)
print(f"čvorova {len(nodes)} | ivica {len(edges)} | dužina {edges['length'].sum() / 1000:.0f} km")
print("raskrsnice 4+ krakova:", int((nodes.street_count >= 4).sum()))
