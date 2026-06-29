import os
import sys

# srpska slova u ispisu: kad je izlaz preusmeren u fajl, Windows Python pada na
# zastareli kodni raspored bez č/ž; try jer Jupyter stdout nema reconfigure
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# putanje
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(SCRIPTS)
DATA = os.path.join(PROJ, "data")
RESULTS = os.path.join(PROJ, "results")
FIGURES = os.path.join(PROJ, "figures")
for _d in (DATA, RESULTS, FIGURES):
    os.makedirs(_d, exist_ok=True)

# fajlovi podataka
NS_BOUNDARY_GPKG = os.path.join(DATA, "ns_boundary.gpkg")
NS_BBOX_JSON = os.path.join(DATA, "ns_bbox.json")
FOOTPRINTS_PARQUET = os.path.join(DATA, "ns_buildings.parquet")
S2_TIFF = os.path.join(DATA, "ns_s2_median.tiff")
INDICES_TIFF = os.path.join(DATA, "ns_indices.tiff")

# geografija
NS_PLACE = "Grad Novi Sad, Serbia"
NS_CENTER_WGS = (19.8335, 45.2551)
CRS_M = 32634    # UTM 34N (metri, za analizu)
CRS_WGS = 4326   # lat/lon (za upite)

# Sentinel-2 (satellite.py)
S2_BANDS = ["B04", "B08", "B11"]
S2_TEMPORAL = ["2024-05-01", "2024-09-30"]
S2_MAX_CLOUD = 30

# modelovanje
RADIUS_M = 300   # poluprečnik okoline za atribute po zgradi (features.py)
REDUNDANT_FEATURES = ("perimeter", "share_4way", "floors", "compactness")
DENDROGRAM_SAMPLE = 30000
K_OP = 4         # operativno (kanonsko) k

# vizualizacija: nazivi i boje tipova po broju klastera k
LABELS = {
    2: {"names": {0: "Ruralno/raspršeno", 1: "Urbano/izgrađeno"},
        "colors": {0: "#4daf4a", 1: "#377eb8"}},
    3: {"names": {0: "Ruralno/retko", 1: "Širi urbani tip", 2: "Stambeno tkivo"},
        "colors": {0: "#4daf4a", 1: "#e41a1c", 2: "#377eb8"}},
    4: {"names": {0: "Urbano stambeno", 1: "Prigradsko/seosko stambeno",
                  2: "Ruralno/retko", 3: "Centralno mešovito jezgro"},
        "colors": {0: "#377eb8", 1: "#984ea3", 2: "#4daf4a", 3: "#e41a1c"}},
}

# zajedničko
DPI = 160   # rezolucija svih figura za plotovanje
