import os
import math

import numpy as np
import geopandas as gpd
import rasterio
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter
from rasterio.transform import rowcol
from pyproj import Transformer
from scripts import config

# atributi okoline za svaku zgradu, u krugu poluprečnika R
R = config.RADIUS_M
CIRCLE_KM2 = math.pi * (R / 1000) ** 2
CIRCLE_M2 = math.pi * R * R

# ulazi: zgrade + sadržaji + čvorovi/ivice mreže
zgrade = gpd.read_parquet(config.FOOTPRINTS_PARQUET)
pois = gpd.read_parquet(os.path.join(config.DATA, "ns_pois.parquet"))
nodes = gpd.read_parquet(os.path.join(config.DATA, "ns_road_nodes.parquet"))
edges = gpd.read_parquet(os.path.join(config.DATA, "ns_road_edges.parquet"))

centroids = zgrade.geometry.centroid
zgrade["cx"], zgrade["cy"] = centroids.x.values, centroids.y.values
pts = np.c_[zgrade["cx"].values, zgrade["cy"].values]
areas = zgrade["area_m2"].values.astype(float)


def tree_of(gdf):
    return cKDTree(np.c_[gdf.geometry.x.values, gdf.geometry.y.values])


print("računanje atributa okoline ...")

# oblik zgrade
zgrade["perimeter"] = zgrade.geometry.length
zgrade["compactness"] = (4 * math.pi * zgrade["area_m2"]) / (zgrade["perimeter"] ** 2).clip(lower=1.0)

# gustina zgrada u okolini
building_tree = cKDTree(pts)
zgrade["nn_dist"] = building_tree.query(pts, k=2)[0][:, 1]
zgrade["n_within_50m"] = building_tree.query_ball_point(pts, 50, return_length=True) - 1
building_neighbors = building_tree.query_ball_point(pts, R)
n_buildings = np.array([len(ix) for ix in building_neighbors], dtype="int64")
roof_area = np.array([areas[ix].sum() for ix in building_neighbors])
zgrade["building_density"] = n_buildings / CIRCLE_KM2
zgrade["built_fraction"] = np.clip(roof_area / CIRCLE_M2, 0, 1)
zgrade["mean_bsize"] = np.array([areas[ix].mean() for ix in building_neighbors])

# rastojanja do sadržaja i do centra
grocery = pois[pois.cat == "grocery"]
education = pois[pois.cat == "education"]
zgrade["dist_grocery"] = tree_of(grocery).query(pts, k=1)[0]
zgrade["dist_school"] = tree_of(education).query(pts, k=1)[0] if len(education) else np.nan
intersections = nodes[nodes.street_count >= 3]
node_tree = tree_of(intersections)
zgrade["dist_intersection"] = node_tree.query(pts, k=1)[0]
ccx, ccy = Transformer.from_crs(4326, config.CRS_M, always_xy=True).transform(*config.NS_CENTER_WGS)
zgrade["dist_center"] = np.hypot(zgrade["cx"] - ccx, zgrade["cy"] - ccy)

# gustina i raznovrsnost sadržaja u okolini
cats = sorted(pois.cat.unique())
catidx = {c: i for i, c in enumerate(cats)}
catcode = np.array([catidx[c] for c in pois.cat.values])
poi_tree = tree_of(pois)
poi_neighbors = poi_tree.query_ball_point(pts, R)
n_pois = np.array([len(ix) for ix in poi_neighbors], dtype="int64")
rcodes = [catidx[c] for c in ("retail", "office", "food") if c in catidx]
poi_mix = np.zeros(len(zgrade))
retail = np.zeros(len(zgrade))
for i, ix in enumerate(poi_neighbors):
    if ix:
        cnt = np.bincount(catcode[ix], minlength=len(cats))
        poi_mix[i] = float((cnt > 0).sum())
        retail[i] = float(cnt[rcodes].sum())
zgrade["poi_density"] = n_pois / CIRCLE_KM2
zgrade["poi_mix"] = poi_mix
zgrade["retail_ratio"] = retail / np.maximum(n_buildings, 1)

# gustina ulične mreže i raskrsnica u okolini
edge_mid = edges.geometry.representative_point()
edge_tree = cKDTree(np.c_[edge_mid.x.values, edge_mid.y.values])
edge_len = edges["length"].values.astype(float)
edge_neighbors = edge_tree.query_ball_point(pts, R)
zgrade["road_density"] = np.array([edge_len[ix].sum() if ix else 0.0 for ix in edge_neighbors]) / CIRCLE_KM2
street_counts = intersections.street_count.values
node_neighbors = node_tree.query_ball_point(pts, R)
n_intersections = np.array([len(ix) for ix in node_neighbors], dtype="int64")
zgrade["inter_density"] = n_intersections / CIRCLE_KM2
n_4way = np.array([(street_counts[ix] >= 4).sum() if ix else 0 for ix in node_neighbors])
zgrade["share_4way"] = n_4way / np.maximum(n_intersections, 1)

with rasterio.open(config.INDICES_TIFF) as ds:
    arr = ds.read().astype("float32")
    transform, height, width = ds.transform, ds.height, ds.width
# NDVI/NDBI kao prosek okoline (~R m), a ne tačkasto na krovu: mean filter pa uzorkovanje
# piksel je 10 m, pa je prozor 2R/10 px (kvadratni ~2R m; hvata zelenilo oko zgrade, ne sam otisak)
kernel_px = max(1, int(round(2 * R / 10)))
for i in range(arr.shape[0]):
    arr[i] = uniform_filter(arr[i], size=kernel_px, mode="nearest")
rows, cols = rowcol(transform, zgrade["cx"].values, zgrade["cy"].values)
rows = np.clip(np.asarray(rows), 0, height - 1)
cols = np.clip(np.asarray(cols), 0, width - 1)
for i, name in enumerate(["ndvi", "ndbi"]):
    zgrade[name] = arr[i, rows, cols]

# spajanje svih atributa i upis (parquet za pipeline, csv za pregled)
FEAT = ["area_m2", "perimeter", "compactness", "floors",
        "nn_dist", "n_within_50m", "ndvi", "ndbi",
        "dist_grocery", "dist_school", "dist_intersection", "dist_center",
        "building_density", "built_fraction", "mean_bsize", "poi_density", "poi_mix",
        "road_density", "inter_density", "share_4way", "retail_ratio"]
keep_label = [c for c in ["class", "subtype"] if c in zgrade.columns]
out = zgrade[["bid"] + FEAT + keep_label + ["geometry"]].copy()
out[FEAT] = out[FEAT].astype("float32")
out.to_parquet(os.path.join(config.DATA, "features_building.parquet"), index=False)
out.drop(columns=["geometry"]).to_csv(os.path.join(config.DATA, "features_building.csv"),
                                      index=False, encoding="utf-8-sig")
print(f"sačuvano redova={len(out)} atributa={len(FEAT)}")
print(out[FEAT].describe().T[["mean", "min", "max"]].round(2).to_string())
