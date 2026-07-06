import contextily as cx
from matplotlib_scalebar.scalebar import ScaleBar

BASEMAP = cx.providers.CartoDB.PositronNoLabels


def add_basemap(ax, attribution=True):
    # CartoDB podloga; bez interneta stavlja sivu pozadinu umesto greške
    try:
        kw = {"attribution_size": 6} if attribution else {"attribution": False}
        cx.add_basemap(ax, crs=32634, source=BASEMAP, zorder=0, **kw)
    except Exception as ex:
        ax.set_facecolor("#ececec")
        print(f"  (basemap preskočen: {str(ex)[:60]})")


def add_scale_north(ax, north=True):
    # 1 podatkovna jedinica = 1 m (EPSG:32634), pa ScaleBar(1)
    ax.add_artist(ScaleBar(1, location="lower left", box_alpha=0.75, frameon=True, color="black"))
    if north:
        ax.annotate("N", xy=(0.045, 0.13), xytext=(0.045, 0.03), xycoords="axes fraction",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
