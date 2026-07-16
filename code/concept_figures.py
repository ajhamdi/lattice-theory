"""Concept figures for Part I of the Lattice Theory.

Six schematic explainers, one per core mechanism, in the shared politviz
style. These carry no fitted data (Part III owns the fits); they exist to make
the qualitative mechanisms legible before the case studies:

  concept_potential_well.png   §3  bonding / the security dilemma
  concept_free_energy.png      §4  G = H - TS crossover (the unification)
  concept_phase_diagram.png    §5  order-types as a phase diagram
  concept_defects.png          §6  the defect zoo (vacancy/interstitial/...)
  concept_misorientation.png   §7  grain-boundary energy vs misorientation
  concept_quench.png           §9  cooling rate: pearlite vs martensite
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, Polygon
import politviz as pv

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.environ.get("LT_FIGDIR", os.path.join(HERE, "..", "figures"))
pv.apply_style()


# --------------------------------------------------------------------------
# §3 — the interatomic potential well and the security dilemma
# --------------------------------------------------------------------------
def morse(r, D, a, r0):
    return D * ((1 - np.exp(-a * (r - r0))) ** 2 - 1)


def fig_potential_well():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.9))

    # -- left: one anharmonic well, thermal expansion --
    r = np.linspace(0.72, 2.6, 500)
    U = morse(r, D=1.0, a=1.9, r0=1.0)
    ax1.plot(r, U, color=pv.INK, lw=2.4, zorder=3)
    ax1.axhline(0, color=pv.BASELINE, lw=1)

    for E, col, lab, yo in [(-0.72, pv.BLUE, "low volatility", 0.05),
                            (-0.30, pv.RED, "high volatility", 0.05)]:
        # turning points where U = E
        sel = r[U <= E]
        rl, rr = sel.min(), sel.max()
        mid = (rl + rr) / 2
        ax1.plot([rl, rr], [E, E], color=col, lw=2.0, zorder=4)
        ax1.plot([mid], [E], marker="o", color=col, ms=7, zorder=5)
        ax1.text(rr + 0.03, E, f"  {lab}", color=col, fontsize=9,
                 fontweight="bold", va="center")
    # drift arrow between the two midpoints
    m1 = (r[U <= -0.72].min() + r[U <= -0.72].max()) / 2
    m2 = (r[U <= -0.30].min() + r[U <= -0.30].max()) / 2
    ax1.annotate("", xy=(m2, -0.30), xytext=(m1, -0.72),
                 arrowprops=dict(arrowstyle="->", color=pv.INK2, lw=1.6))
    ax1.text((m1 + m2) / 2 - 0.02, -0.51, "mean distance\ndrifts outward\n(allies hedge)",
             color=pv.INK2, fontsize=8.5, ha="right", va="center")
    ax1.plot([1.0], [-1.0], marker="o", color=pv.INK, ms=6, zorder=6)
    ax1.text(1.0, -1.06, "equilibrium", color=pv.INK2, fontsize=8.5,
             ha="center", va="top")
    ax1.annotate("steep repulsive wall\n(the security dilemma)", xy=(0.83, 0.45),
                 xytext=(1.35, 0.62), color=pv.INK2, fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color=pv.MUTED, lw=1.2))
    pv.titles(ax1, "The bond is an anharmonic well",
              "closer = steep repulsion; hotter = greater average distance")
    ax1.set_xlim(0.72, 2.6)
    ax1.set_ylim(-1.15, 0.8)
    ax1.set_xlabel("separation between two states  (bilateral distance)")
    ax1.set_ylabel("potential energy")

    # -- right: stiff deep well vs shallow well --
    r2 = np.linspace(0.72, 2.8, 500)
    ax2.plot(r2, morse(r2, 1.0, 2.4, 1.0), color=pv.BLUE, lw=2.4)
    ax2.plot(r2, morse(r2, 0.32, 1.3, 1.25), color=pv.YELLOW, lw=2.4)
    ax2.axhline(0, color=pv.BASELINE, lw=1)
    ax2.text(1.02, -1.02, "deep, sharply curved\nUS–Canada", color=pv.BLUE,
             fontsize=9, fontweight="bold", ha="left", va="top")
    ax2.text(1.9, -0.30, "shallow, soft\nRussia–Türkiye", color=pv.YELLOW,
             fontsize=9, fontweight="bold", ha="left", va="center")
    pv.titles(ax2, "Stiffness = curvature at the minimum",
              "the same perturbation, two restoring forces")
    ax2.set_xlim(0.72, 2.8)
    ax2.set_ylim(-1.15, 0.8)
    ax2.set_xlabel("separation between two states")
    ax2.set_ylabel("potential energy")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "concept_potential_well.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# §4 — G = H - T S : the free-energy crossover / the unification
# --------------------------------------------------------------------------
def fig_free_energy():
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    Th = np.linspace(0, 100, 200)
    Hr, Sr = 10.0, 0.08     # rigid order: low enthalpy (serves interests), low entropy
    Hf, Sf = 16.0, 0.20     # flexible order: worse enthalpy, high entropy
    Gr = Hr - Th * Sr
    Gf = Hf - Th * Sf
    Tstar = (Hf - Hr) / (Sf - Sr)

    ax.axvspan(0, Tstar, color=pv.BLUE, alpha=0.05)
    ax.axvspan(Tstar, 100, color=pv.AQUA, alpha=0.06)
    ax.plot(Th, Gr, color=pv.BLUE, lw=2.6)
    ax.plot(Th, Gf, color=pv.AQUA, lw=2.6)
    ax.text(3, 8.9, "rigid order   G = H − ΘS", color=pv.BLUE,
            fontsize=9, fontweight="bold", ha="left")
    ax.text(21, 13.4, "flexible, pluralist order", color=pv.AQUA,
            fontsize=9, fontweight="bold", ha="left")
    ax.axvline(Tstar, color=pv.INK2, lw=1.2, ls=":")
    ax.plot([Tstar], [Hr - Tstar * Sr], marker="o", color=pv.INK, ms=7, zorder=6)
    ax.text(Tstar, 1.0, f" Θ*  ≈ {Tstar:.0f}", color=pv.INK, fontsize=9,
            fontweight="bold", va="bottom")

    ax.text(34, 15.2, "low Θ: enthalpy wins\n→ realism's world", color=pv.BLUE,
            fontsize=9.5, ha="center", va="center", fontweight="bold")
    ax.text(74, 15.2, "high Θ: entropy wins\n→ liberal & constructivist world",
            color="#0d7a54", fontsize=9.5, ha="center", va="center",
            fontweight="bold")

    pv.titles(ax, "One free-energy landscape, three schools of IR",
              "viability G of an order vs systemic temperature Θ — lower G wins")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 17)
    ax.set_xlabel("systemic temperature  Θ   (volatility, 0–100)")
    ax.set_ylabel("free energy  G  (lower = more viable)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "concept_free_energy.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# §5 — the political phase diagram
# --------------------------------------------------------------------------
def fig_phase_diagram():
    fig, ax = plt.subplots(figsize=(10, 5.8))

    # melt / anarchy band across the top (liquidus)
    x = np.linspace(0, 1, 200)
    liquidus = 62 + 14 * np.sin(np.pi * x) - 6 * x
    ax.fill_between(x, liquidus, 100, color=pv.RED, alpha=0.08, zorder=0)
    ax.plot(x, liquidus, color=pv.RED, lw=1.4, ls="--", alpha=0.7)
    ax.text(0.5, 90, "ANARCHY  /  THE MELT",
            color=pv.RED, fontsize=11, ha="center", fontweight="bold", alpha=0.8)
    ax.text(0.5, 83, "no stable order: Θ above the liquidus dissolves any arrangement",
            color=pv.INK2, fontsize=8.5, ha="center")

    # solid-order regions (translucent patches)
    regions = [
        (Polygon([(0, 0), (0.2, 0), (0.2, 62), (0, 68)]), pv.BLUE, 0.10),
        (Polygon([(0.2, 0), (0.42, 0), (0.42, 55), (0.2, 62)]), pv.AQUA, 0.12),
        (Polygon([(0.42, 0), (0.72, 0), (0.72, 30), (0.42, 30)]), pv.YELLOW, 0.16),
        (Polygon([(0.42, 30), (0.72, 30), (0.78, 66), (0.42, 55)]), "#8a5cc4", 0.14),
        (Polygon([(0.72, 0), (1.0, 0), (1.0, 60), (0.78, 66), (0.72, 30)]),
         pv.MUTED, 0.14),
    ]
    for poly, col, al in regions:
        poly.set_facecolor(col)
        poly.set_alpha(al)
        poly.set_edgecolor("none")
        ax.add_patch(poly)

    labels = [
        (0.09, 34, "HEGEMONIC\n(single crystal)", pv.BLUE),
        (0.31, 24, "BIPOLAR\n(eutectic lamellae)", "#0d7a54"),
        (0.60, 20, "SEGREGATED\nMULTIPOLARITY\n(brittle)", "#9a6a00"),
        (0.61, 52, "HIGH-ENTROPY\nMULTIPOLARITY\n(tough)", "#6a3fb0"),
        (0.88, 24, "FRAGMENTED\n(micro-states)", pv.INK2),
    ]
    for xx, yy, tx, col in labels:
        ax.text(xx, yy, tx, color=col, fontsize=8.6, ha="center", va="center",
                fontweight="bold")

    # historical exemplars
    pts = [
        (0.10, 17, "Pax Britannica", pv.BLUE, (10, 7)),
        (0.30, 15, "Cold War", "#0d7a54", (8, -13)),
        (0.55, 41, "Concert of Europe", "#6a3fb0", (9, -15)),
        (0.49, 9, "1930s", "#9a6a00", (-4, -15)),
        (0.72, 78, "2020s", pv.RED, (10, -3)),
    ]
    for xx, yy, tx, col, off in pts:
        ax.plot([xx], [yy], marker="o", color=col, ms=8, zorder=6,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate(tx, (xx, yy), xytext=off, textcoords="offset points",
                    color=col, fontsize=8.6, fontweight="bold")

    pv.titles(ax, "The phase diagram of world order",
              "which order-type is stable at which power distribution and volatility")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("composition  →   power distribution, from pure hegemony (left) "
                  "to fine fragmentation (right)")
    ax.set_ylabel("systemic temperature  Θ   (volatility)")
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "concept_phase_diagram.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# §6 — the defect zoo
# --------------------------------------------------------------------------
def fig_defects():
    fig, ax = plt.subplots(figsize=(12, 6.6))

    def grid(x0, y0, nx, ny, ang, d=1.0):
        pts = []
        c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        for i in range(nx):
            for j in range(ny):
                pts.append((x0 + d * (i * c - j * s),
                            y0 + d * (i * s + j * c), i, j))
        return pts

    left = grid(1.0, 1.0, 7, 8, 0)
    right = grid(9.6, 0.7, 7, 8, 19)

    # boundary band
    ax.add_patch(Rectangle((7.7, -0.5), 1.05, 10.5, facecolor=pv.RED,
                           alpha=0.07, edgecolor="none", zorder=0))

    vac = (2, 6)          # vacancy site (skipped)
    sub = (4, 1)          # substitutional site (recolored)
    for (x, y, i, j) in left:
        if (i, j) == vac:
            ax.add_patch(Circle((x, y), 0.24, fill=False, ls=(0, (2, 2)),
                                edgecolor=pv.RED, lw=1.9, zorder=4))
            continue
        col = pv.YELLOW if (i, j) == sub else pv.BLUE
        ax.plot([x], [y], marker="o", color=col, ms=10, zorder=3,
                markeredgecolor="white", markeredgewidth=0.9)
    # interstitial atom, wedged off-site
    ax.plot([2.6], [4.6], marker="o", color=pv.AQUA, ms=9, zorder=5,
            markeredgecolor="white", markeredgewidth=0.9)

    for (x, y, i, j) in right:
        if x < 8.9:
            continue
        ax.plot([x], [y], marker="o", color=pv.INK2, ms=10, zorder=3,
                markeredgecolor="white", markeredgewidth=0.9)

    # edge dislocation: an extra half-plane inserted between two columns, + ⊥
    ax.plot([5.5, 5.5], [4.8, 8.2], color=pv.INK, lw=2.4, zorder=6)
    ax.text(5.5, 4.45, "⊥", fontsize=18, ha="center", va="center",
            color=pv.INK, fontweight="bold", zorder=7)

    def call(xy, xytext, text, col, ha="center"):
        ax.annotate(text, xy=xy, xytext=xytext, fontsize=9.3, color=col,
                    fontweight="bold", ha=ha, va="center", zorder=8,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))

    call((3.0, 7.0), (2.8, 10.0),
         "vacancy =\nfailed / collapsed state", pv.RED)
    call((2.6, 4.6), (-3.4, 4.6),
         "interstitial =\nnon-state actor\n(militia, cartel, platform)", "#0d7a54", ha="left")
    call((5.0, 2.0), (5.0, -1.1),
         "substitutional =\nforeign-aligned state", "#9a6a00")
    call((5.5, 4.45), (-3.4, 7.6),
         "dislocation =\nmobile line of instability\n(norm cascade, 1989)", pv.INK, ha="left")
    call((8.2, 6.4), (8.6, 10.0),
         "grain boundary =\ngeopolitical fault line", pv.RED)
    ax.text(15.3, 10.05, "a second grain,\nmisoriented", color=pv.INK2, fontsize=9.3,
            fontweight="bold", ha="right", va="center")

    pv.titles(ax, "The defect zoo: where all the politics actually happens",
              "a perfect state-lattice is a fiction — real world order is defined by its imperfections")
    ax.set_xlim(-3.6, 15.4)
    ax.set_ylim(-1.6, 10.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sname in ax.spines.values():
        sname.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "concept_defects.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# §7 — grain-boundary energy vs misorientation (schematic; the founding insight)
# --------------------------------------------------------------------------
def fig_misorientation():
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    th = np.linspace(0, 60, 400)
    # schematic monotone-saturating boundary energy (the law itself is Part I prose)
    g = 1 - np.exp(-th / 11.0)
    ax.plot(th, g, color=pv.INK, lw=2.6, zorder=3)
    ax.axvspan(0, 15, color=pv.BLUE, alpha=0.06)
    ax.axvspan(15, 60, color=pv.RED, alpha=0.05)

    x1, y1 = 3.0, 1 - np.exp(-3.0 / 11.0)
    ax.plot([x1], [y1], marker="o", color=pv.BLUE, ms=10, zorder=5,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.annotate("US–Canada,\nintra-Scandinavia", (x1, y1), xytext=(12, -2),
                textcoords="offset points", color=pv.BLUE, fontsize=9,
                fontweight="bold", va="center")
    x2, y2 = 45.0, 1 - np.exp(-45.0 / 11.0)
    ax.plot([x2], [y2], marker="o", color=pv.RED, ms=10, zorder=5,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.text(58, 0.80, "DMZ · India–Pakistan LoC · NATO–Russia line",
            color=pv.RED, fontsize=9, fontweight="bold", ha="right", va="top")

    ax.text(7.5, 0.90, "low-angle:\nnear-zero energy\n(democratic peace)",
            color=pv.BLUE, fontsize=8.6, ha="center", fontweight="bold")
    ax.text(37, 0.33, "high-angle: full energy regardless of how much trade crosses",
            color="#a12b2a", fontsize=8.6, ha="center", fontweight="bold")

    pv.titles(ax, "Friction scales with misorientation, not distance",
              "the founding insight — interfacial energy of a boundary vs the angular "
              "mismatch of the blocs it separates (schematic)")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("misorientation angle between blocs   (mismatch of ordering principles)")
    ax.set_ylabel("boundary energy   (fuel for conflict)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "concept_misorientation.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# §9 — the quench: cooling rate decides the regime (TTT diagram)
# --------------------------------------------------------------------------
def fig_quench():
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    T = np.linspace(120, 720, 300)
    Tnose = 430.0
    # C-curve: transformation-start time (log) with a nose at Tnose
    logt = 0.7 + 0.9 * ((T - Tnose) / 150.0) ** 2
    ts = 10 ** logt
    ax.plot(ts, T, color=pv.YELLOW, lw=2.6, zorder=3)
    pv.direct_label(ax, ts[np.argmin(ts)], Tnose,
                    "transformation-start\n'nose'", pv.YELLOW, dx=10, va="center")

    Ms = 250.0
    ax.axhline(Ms, color=pv.INK2, lw=1.2, ls=":")
    ax.text(1.2, Ms + 12, "Ms  (martensite start)", color=pv.INK2, fontsize=8.6)

    # slow cool: crosses the nose -> pearlite (tough)
    t_slow = np.array([1.0, 6, 30, 120, 700, 4000])
    T_slow = np.array([700, 620, 500, 400, 300, 200])
    ax.plot(t_slow, T_slow, color=pv.AQUA, lw=2.4, zorder=4)
    ax.plot([120], [400], marker="o", color=pv.AQUA, ms=8, zorder=5)
    ax.text(4200, 200, "slow cool → pearlite\ntough democracy\n"
            "(Spain, S. Korea, Taiwan)", color="#0d7a54", fontsize=9,
            fontweight="bold", va="center")

    # fast quench: misses the nose -> martensite (brittle)
    t_fast = np.array([1.0, 1.6, 2.4, 3.4])
    T_fast = np.array([700, 520, 330, 150])
    ax.plot(t_fast, T_fast, color=pv.RED, lw=2.4, zorder=4)
    ax.text(3.6, 150, "fast quench → martensite\nbrittle regime\n"
            "(1990s Russia, de-Ba'athified Iraq)", color="#a12b2a",
            fontsize=9, fontweight="bold", va="center")

    ax.fill_between([1e3, 1e5], Ms, 720, color=pv.AQUA, alpha=0.05)
    pv.titles(ax, "Rate is destiny: the same endpoint, two regimes",
              "identical composition, different cooling rate through the "
              "transformation nose")
    ax.set_xscale("log")
    ax.set_xlim(1, 3e4)
    ax.set_ylim(120, 720)
    ax.set_xlabel("time  (speed of the political transition; log scale)")
    ax.set_ylabel("temperature  (of the transition process)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "concept_quench.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_potential_well()
    fig_free_energy()
    fig_phase_diagram()
    fig_defects()
    fig_misorientation()
    fig_quench()
    print("concept figures written to", os.path.abspath(FIGS))
