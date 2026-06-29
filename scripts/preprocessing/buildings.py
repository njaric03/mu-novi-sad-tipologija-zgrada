import os
import sys
import json
import subprocess

import numpy as np
import pandas as pd
import geopandas as gpd
from scripts import config

# bbox u WGS koordinatama za Overture download
raw = os.path.join(config.DATA, "ns_buildings_raw.parquet")
with open(config.NS_BBOX_JSON, encoding="utf-8") as f:
    w, s, e, n = json.load(f)["bbox_wgs"]  # strane sveta

# preuzimanje otisaka zgrada iz Overture Maps (samo prvi put)
if not os.path.exists(raw):
    print(f"overturemaps preuzimanje, bbox {w:.4f},{s:.4f},{e:.4f},{n:.4f} ...")
    subprocess.run(
        [sys.executable, "-m", "overturemaps", "download", f"--bbox={w},{s},{e},{n}",
         "-f", "geoparquet", "--type", "building", "-o", raw],
        check=True
    )

zgrade = gpd.read_parquet(raw)
if zgrade.crs is None:
    zgrade = zgrade.set_crs(config.CRS_WGS)
zgrade = zgrade.to_crs(config.CRS_M)

# samo zgrade unutar granice grada
boundary = gpd.read_file(config.NS_BOUNDARY_GPKG).to_crs(config.CRS_M).geometry.iloc[0]
zgrade = zgrade[zgrade.geometry.representative_point().within(boundary)].copy()
print(f"broj zgrada u granicama NS: {len(zgrade)}")

# id, geometrija i (ako postoje) atributi visine/namene
keep = ["id", "geometry"]
for col in ["height", "num_floors", "class", "subtype"]:
    if col in zgrade.columns:
        keep.append(col)
zgrade = zgrade[keep].copy()

# ako visina ili spratovi ne postoje u izvozu, dodaju se kao prazne (NaN) kolone
for col in ["height", "num_floors"]:
    if col not in zgrade.columns:
        zgrade[col] = np.nan
zgrade["area_m2"] = zgrade.geometry.area
h = pd.to_numeric(zgrade["height"], errors="coerce")
fl = pd.to_numeric(zgrade["num_floors"], errors="coerce")
# spratovi: iz podatka, pa procena iz visine (~3 m po spratu), pa 1; nikad ispod 1
zgrade["floors"] = fl.fillna((h / 3).round()).fillna(1).clip(lower=1)

# sitni poligoni (šum) se izbacuju, dodeljuje se stabilan id
zgrade = zgrade[zgrade["area_m2"] >= 10].reset_index(drop=True)
zgrade["bid"] = zgrade.index
zgrade.to_parquet(config.FOOTPRINTS_PARQUET, index=False)

print(f"sačuvano {len(zgrade)} zgrada | median {zgrade.area_m2.median():.0f} m2 | "
      f"sa visinom {int(h.notna().sum())}, sa spratovima {int(fl.notna().sum())}")
