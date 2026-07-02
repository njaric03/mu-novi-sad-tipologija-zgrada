import os
import json
import time

import numpy as np
import rasterio
import openeo
from scripts import config

# bbox u UTM metrima + mali padding, za openEO prostorni presek
with open(config.NS_BBOX_JSON, encoding="utf-8") as f:
    w, s, e, n = json.load(f)["bbox_utm"]
PAD = 500
ext = {"west": w - PAD, "south": s - PAD, "east": e + PAD, "north": n + PAD, "crs": "EPSG:32634"}

# Sentinel-2 letnja medijana bez oblaka daje jedan kompozit; preuzima se samo prvi put
if not os.path.exists(config.S2_TIFF):
    con = openeo.connect("openeo.dataspace.copernicus.eu")
    con.authenticate_oidc()
    print("prijava ok; priprema openEO kocke ...")
    cube = (
        con.load_collection("SENTINEL2_L2A", spatial_extent=ext, temporal_extent=config.S2_TEMPORAL,
                            bands=config.S2_BANDS, max_cloud_cover=config.S2_MAX_CLOUD)
        .reduce_dimension(dimension="t", reducer="median")
        .resample_spatial(resolution=10, projection=32634)
    )
    # openEO batch posao ume da padne ili istekne, pa se pokušava do 3 puta
    for attempt in range(3):
        try:
            print(f"openEO batch posao, pokušaj {attempt + 1} ...")
            cube.execute_batch(config.S2_TIFF, out_format="GTiff", title="ns_s2_median")
            break
        except Exception as ex:
            print(f"pokušaj {attempt + 1} pao: {str(ex)[:90]}")
            time.sleep(15)
    else:
        raise RuntimeError("S2 kompozit nije preuzet ni iz 3 pokušaja")

print(f"S2 kompozit: {os.path.getsize(config.S2_TIFF) / 1e6:.0f} MB")

with rasterio.open(config.S2_TIFF) as ds:
    bands = ds.read().astype("float32")
    prof = ds.profile
B04, B08, B11 = bands


def ratio(x, y):
    # normalizovana razlika, npr. (B08-B04)/(B08+B04); +1e-9 protiv deljenja nulom
    return ((x - y) / (x + y + 1e-9)).astype("float32")


# NDVI = zelenilo, NDBI = izgrađenost
ndvi = ratio(B08, B04)
ndbi = ratio(B11, B08)
idx = np.stack([ndvi, ndbi])

prof.update(count=2, dtype="float32", compress="deflate")
with rasterio.open(config.INDICES_TIFF, "w", **prof) as dst:
    dst.write(idx)
    for i, name in enumerate(["NDVI", "NDBI"], 1):
        dst.set_band_description(i, name)

print(f"sačuvano {config.INDICES_TIFF}  NDVI[{ndvi.min():.2f},{ndvi.max():.2f}] mean {ndvi.mean():.3f}")
