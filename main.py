import subprocess
import sys


def run(module, *args):
    print(f"\n===== {module} {' '.join(args)} =====")
    subprocess.run([sys.executable, "-u", "-m", module, *args], check=True)


# predobrada
run("scripts.preprocessing.boundary")
run("scripts.preprocessing.buildings")
run("scripts.preprocessing.satellite")
run("scripts.preprocessing.pois")      # OSM sadržaji (prodavnice, škole, ...)
run("scripts.preprocessing.roads")
run("scripts.preprocessing.features")  # atributi po zgradi (radijus 300 m)

# modelovanje
run("scripts.modeling.cluster", "2")   # broj = k (broj tipova)
run("scripts.modeling.cluster", "3")
run("scripts.modeling.cluster", "4")
run("scripts.modeling.k_selection")    # izbor k: dendrogram + stabilnost + elbow/silueta
run("scripts.modeling.comparisons")    # Ward, k-means, PCA vs slučajna projekcija

# vizualizacija
# mape tipologije za isto k=2,3,4
run("scripts.visualization.maps", "2")
run("scripts.visualization.maps", "3")
run("scripts.visualization.maps", "4")
# figure za rad + interaktivna HTML mapa (izlaz u figures/)
run("scripts.visualization.paper_figures")
run("scripts.visualization.type_maps")
run("scripts.visualization.poi_map")
run("scripts.visualization.web_map")

print("\nDONE")
