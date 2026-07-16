"""Geographic case-study maps for the Lattice Theory.

Real country borders (Natural Earth 50m, via geomap.py) carrying the theory's
overlay:

  map_road_to_1914.png     Case D (§22) — the pre-1914 fatigue cycles and the
                           July-1914 fracture, on the alliance grain boundary.
  map_serbia_kosovo.png    F4 (§27) — the western-Balkans high-misorientation
                           boundary and its compressing crisis cluster.

Alliance/bloc shading is schematic and drawn on MODERN borders (Austria-Hungary
and the Ottoman Empire no longer exist as single features); it marks the two
grains, not 1914 cartography.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import politviz as pv
import geomap

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.environ.get("LT_FIGDIR", os.path.join(HERE, "..", "figures"))
pv.apply_style()

def _mtitle(ax, title, subtitle):
    """Title/subtitle with enough headroom for a full-bleed map axes."""
    if not pv.SHOW_TITLES:
        return
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=32)
    ax.annotate(subtitle, xy=(0, 1), xytext=(0, 7), xycoords="axes fraction",
                textcoords="offset points", fontsize=9, color=pv.INK2, va="bottom")


ENTENTE = "#c3d9f2"
CENTRAL = "#f6d2b8"
HL_A = "#f4c9a3"   # Serbia
HL_B = "#bcd4f2"   # Kosovo


# --------------------------------------------------------------------------
# Case D — the road to 1914
# --------------------------------------------------------------------------
def map_1914():
    fig = plt.figure(figsize=(11, 8.4))
    gs = GridSpec(2, 1, height_ratios=[5.4, 1.25], hspace=0.16)
    ax = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1])

    fills = {}
    for c in ["France", "United Kingdom", "Russia", "Serbia", "Belgium",
              "Montenegro"]:
        fills[c] = ENTENTE
    for c in ["Germany", "Austria", "Hungary", "Bulgaria", "Turk"]:
        fills[c] = CENTRAL

    geomap.draw(ax, bbox=(-12, 31, 28, 60), fills=fills,
                base="#ecebe4", edge="#bdbcb2", lw=0.6, ocean="#f2f4f6")

    # crisis nodes: (lon, lat, label, number, label-offset, ha, va)
    nodes = [
        (-5.8, 35.8, "Tangier 1905", "1", (0, -20), "center", "top"),
        (16.4, 45.4, "Bosnia 1908", "2", (-14, 6), "right", "center"),
        (-9.6, 30.4, "Agadir 1911", "3", (0, -20), "center", "top"),
        (21.4, 41.3, "1st Balkan 1912", "4", (-14, -14), "right", "top"),
        (23.4, 42.1, "2nd Balkan 1913", "5", (14, -8), "left", "top"),
    ]
    for lon, lat, lab, num, off, ha, va in nodes:
        ax.plot([lon], [lat], marker="o", ms=15, color=pv.YELLOW,
                markeredgecolor="white", markeredgewidth=1.4, zorder=6)
        ax.text(lon, lat, num, ha="center", va="center", fontsize=9,
                fontweight="bold", color=pv.INK, zorder=7)
        ax.annotate(lab, (lon, lat), xytext=off, textcoords="offset points",
                    ha=ha, va=va, fontsize=8.6, fontweight="bold",
                    color="#9a6a00", zorder=7)
    # the fracture
    fx, fy = 18.42, 43.85
    ax.plot([fx], [fy], marker="*", ms=30, color=pv.RED,
            markeredgecolor="white", markeredgewidth=1.2, zorder=8)
    ax.annotate("Sarajevo\nJuly 1914 → FRACTURE", (fx, fy), xytext=(14, 14),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=9.5, fontweight="bold", color=pv.RED, zorder=8,
                arrowprops=dict(arrowstyle="->", color=pv.RED, lw=1.6))

    # bloc legend (manual swatches), placed over empty North Atlantic
    ax.add_patch(plt.Rectangle((-11.4, 56.0), 1.4, 1.1, facecolor=ENTENTE,
                               edgecolor="#bdbcb2", lw=0.6, zorder=5))
    ax.text(-9.7, 56.55, "Entente grain", fontsize=9, va="center", color=pv.INK2)
    ax.add_patch(plt.Rectangle((-11.4, 57.6), 1.4, 1.1, facecolor=CENTRAL,
                               edgecolor="#bdbcb2", lw=0.6, zorder=5))
    ax.text(-9.7, 58.15, "Central-Powers grain", fontsize=9, va="center", color=pv.INK2)

    _mtitle(ax, "The road to 1914: fatigue on the alliance boundary",
            "six sub-critical crises grow the crack across the Entente / "
            "Central-Powers grain boundary (schematic — modern borders)")

    # timeline strip: the compressing intervals
    years = [1905.25, 1908.79, 1911.50, 1912.79, 1913.46, 1914.56]
    labs = ["1", "2", "3", "4", "5", "★"]
    cols = [pv.YELLOW] * 5 + [pv.RED]
    axt.axhline(0, color=pv.BASELINE, lw=1.4, zorder=1)
    for y, lab, col in zip(years, labs, cols):
        m = "*" if lab == "★" else "o"
        ms = 20 if lab == "★" else 12
        axt.plot([y], [0], marker=m, ms=ms, color=col, markeredgecolor="white",
                 markeredgewidth=1.0, zorder=3)
        if lab != "★":
            axt.text(y, 0, lab, ha="center", va="center", fontsize=7.5,
                     fontweight="bold", zorder=4)
    for a, b in zip(years[:-1], years[1:]):
        axt.annotate("", (b, -0.45), (a, -0.45),
                     arrowprops=dict(arrowstyle="<->", color=pv.MUTED, lw=1.0))
        axt.text((a + b) / 2, -0.72, f"{b - a:.1f} yr", ha="center", va="top",
                 fontsize=7.5, color=pv.INK2)
    axt.text(1914.56, 0.62, "fracture", ha="center", va="bottom", fontsize=8.5,
             fontweight="bold", color=pv.RED)
    axt.text(1904.7, 0.62, "intervals compress toward the singularity  "
             "(Eq. 4 fit: p = 1.03, R² = 0.98)", ha="left", va="bottom",
             fontsize=8.5, color=pv.INK2)
    axt.set_xlim(1904.4, 1915.4)
    axt.set_ylim(-1.1, 1.1)
    axt.set_yticks([])
    for s in ["top", "left", "right"]:
        axt.spines[s].set_visible(False)
    axt.spines["bottom"].set_visible(False)
    axt.set_xticks(range(1905, 1916, 2))
    axt.tick_params(length=0)

    fig.savefig(os.path.join(FIGS, "map_road_to_1914.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# F4 — the Serbia–Kosovo boundary
# --------------------------------------------------------------------------
def map_serbia_kosovo():
    fig, ax = plt.subplots(figsize=(9.8, 8.0))
    fills = {"Kosovo": HL_B, "Serbia": HL_A}
    geomap.draw(ax, bbox=(18.2, 23.4, 40.6, 46.4), fills=fills,
                base="#ecebe4", edge="#b6b5ab", lw=0.7, ocean="#f2f4f6")

    # country labels
    for lon, lat, name, col in [
        (20.8, 44.4, "SERBIA", "#9a6a00"),
        (20.9, 42.55, "KOSOVO", "#245c9c"),
        (19.3, 42.7, "MONTE-\nNEGRO", pv.MUTED),
        (20.1, 41.2, "ALBANIA", pv.MUTED),
        (21.7, 41.5, "N. MACEDONIA", pv.MUTED),
        (18.9, 44.2, "BOSNIA", pv.MUTED),
        (22.7, 42.7, "BULGARIA", pv.MUTED)]:
        ax.text(lon, lat, name, ha="center", va="center", fontsize=9,
                fontweight="bold", color=col, zorder=6)

    # northern-Kosovo flashpoint cluster (2004 -> 2024)
    cluster = [
        (20.75, 43.23, "Jarinje/Brnjak border posts"),
        (20.87, 42.89, "N. Mitrovica"),
        (20.96, 42.99, "Banjska 2023"),
        (20.62, 42.92, "Ibar-Lepenac canal 2024"),
    ]
    for lon, lat, _ in cluster:
        ax.plot([lon], [lat], marker="o", ms=11, color=pv.RED,
                markeredgecolor="white", markeredgewidth=1.2, zorder=7)
    ax.annotate("northern-Kosovo flashpoint cluster\n"
                "ten coercive crises, 2004 → 2024",
                (20.8, 43.05), xytext=(21.7, 44.6),
                textcoords="data", ha="left", va="center", fontsize=9.2,
                fontweight="bold", color=pv.RED, zorder=8,
                arrowprops=dict(arrowstyle="->", color=pv.RED, lw=1.6))

    ax.annotate("high-misorientation\ngrain boundary", (20.62, 42.75),
                xytext=(18.5, 41.5), textcoords="data", ha="left", va="center",
                fontsize=9.2, fontweight="bold", color=pv.INK, zorder=8,
                arrowprops=dict(arrowstyle="->", color=pv.INK, lw=1.4))

    _mtitle(ax, "Europe's most fracture-ripe boundary (Eq. 4 / F4)",
            "Serbia–Kosovo: interval compression τ = −0.61, p = 0.025; "
            "fitted singularity already in its terminal window")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02)
    fig.savefig(os.path.join(FIGS, "map_serbia_kosovo.png"))
    plt.close(fig)


if __name__ == "__main__":
    map_1914()
    map_serbia_kosovo()
    print("maps written to", os.path.abspath(FIGS))
