"""Explanatory figures for core materials-science concepts and their governing
equations, rendered for a non-specialist reader. Pure-metallurgy versions with
the equations annotated, embedded in materials_science_related_work.md.

Outputs (figures/ms_*.png):
  ms_griffith.png        Griffith energy balance + critical crack sigma_c ~ a^-1/2
  ms_paris.png           Paris fatigue crack-growth curve da/dN vs dK
  ms_weibull.png         Weibull failure CDF for varying modulus + size effect
  ms_nucleation.png      Classical nucleation dG(r), barrier and r*
  ms_jmak.png            JMAK/Avrami sigmoids for varying exponent n
  ms_grain_boundary.png  Read-Shockley gamma(theta) + Hall-Petch / inverse HP
  ms_arrhenius.png       Arrhenius vacancy fraction (exp and linearised)

Needs only numpy + matplotlib; runs on system python.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import politviz as pv

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.environ.get("LT_FIGDIR", os.path.join(HERE, "..", "figures"))
pv.apply_style()


def _morse(r, D=1.0, a=3.5, r0=1.0):
    return D * ((1 - np.exp(-a * (r - r0)))**2 - 1)


def _well_endpoints(level, D=1.0, a=3.5, r0=1.0):
    rr = np.linspace(0.70, 3.2, 4000)
    vv = _morse(rr, D, a, r0)
    idx = np.where(vv <= level)[0]
    return rr[idx[0]], rr[idx[-1]]


# --------------------------------------------------------------------------
def fig_potential_well():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # left: the anharmonic well and thermal expansion
    r = np.linspace(0.72, 2.6, 500)
    axL.axhline(0, color=pv.BASELINE, lw=1)
    axL.plot(r, _morse(r), color=pv.INK, lw=2.6)
    axL.plot([1.0], [-1.0], "o", color=pv.INK, ms=7, zorder=6)
    axL.annotate("equilibrium spacing", (1.0, -1.0), xytext=(16, -3),
                 textcoords="offset points", ha="left", va="center",
                 fontsize=8.5, color=pv.INK2)
    axL.annotate("steep repulsive\nwall", (0.80, 0.45), xytext=(1.15, 0.55),
                 fontsize=8.6, color=pv.INK2,
                 arrowprops=dict(arrowstyle="->", color=pv.MUTED, lw=1.2))
    for level, col, lab in [(-0.72, pv.BLUE, "lower $T$"),
                            (-0.30, pv.RED, "higher $T$")]:
        rl, rr = _well_endpoints(level)
        mid = 0.5 * (rl + rr)
        axL.plot([rl, rr], [level, level], color=col, lw=2)
        axL.plot([mid], [level], "o", color=col, ms=8, zorder=6)
        pv.direct_label(axL, rr, level, lab, col, dx=5, dy=0, fontsize=8.8)
    m_lo = 0.5 * sum(_well_endpoints(-0.72))
    m_hi = 0.5 * sum(_well_endpoints(-0.30))
    axL.annotate("", (m_hi, -0.30), (m_lo, -0.72),
                 arrowprops=dict(arrowstyle="->", color=pv.INK, lw=1.7))
    axL.text(1.78, -0.56, "mean spacing drifts\noutward as $T$ rises\n(thermal expansion)",
             fontsize=8.3, color=pv.INK, fontweight="bold", va="center")
    axL.set_xlabel("interatomic separation $r$")
    axL.set_ylabel("potential energy")
    axL.set_ylim(-1.15, 0.8)
    pv.titles(axL, "The interatomic bond is an anharmonic well",
              "the well's asymmetry causes thermal expansion")

    # right: stiffness = curvature
    r2 = np.linspace(0.72, 2.8, 500)
    axR.axhline(0, color=pv.BASELINE, lw=1)
    axR.plot(r2, _morse(r2, D=1.0, a=4.6), color=pv.BLUE, lw=2.6)
    axR.plot(r2, _morse(r2, D=0.34, a=2.6), color=pv.YELLOW, lw=2.6)
    pv.direct_label(axR, 1.02, -1.0, "stiff bond\n(high modulus)", pv.BLUE,
                    dx=6, dy=-4, fontsize=9)
    pv.direct_label(axR, 1.85, -0.20, "compliant bond\n(low modulus)", pv.YELLOW,
                    dx=2, dy=6, fontsize=9)
    axR.set_xlabel("interatomic separation $r$")
    axR.set_ylabel("potential energy")
    axR.set_ylim(-1.15, 0.5)
    pv.titles(axR, "Stiffness = curvature at the minimum",
              "a deeper, more sharply curved well is a stiffer bond")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ms_potential_well.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_quench():
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    T = np.linspace(232, 712, 300)
    Tn = 545.0
    c = 3.0 / (180.0**2)
    logt_start = c * (T - Tn)**2
    logt_finish = logt_start + 0.62
    ax.plot(logt_start, T, color=pv.INK, lw=2.4, label="transformation start")
    ax.plot(logt_finish, T, color=pv.INK, lw=1.5, ls="--", label="transformation finish")
    ax.fill_betweenx(T, logt_start, logt_finish, color=pv.AQUA, alpha=0.10)

    ax.axhline(720, color=pv.MUTED, ls=":", lw=1.1)
    ax.text(3.15, 727, "$A_{e1}$ (equilibrium line)", color=pv.INK2,
            fontsize=8.5, va="bottom", ha="right")
    ax.axhline(220, color=pv.MUTED, ls=":", lw=1.1)
    ax.text(3.15, 227, "$M_s$ (martensite start)", color=pv.INK2,
            fontsize=8.5, va="bottom", ha="right")

    # cooling paths (straight lines in log-time vs T)
    ax.plot([-0.55, 3.05], [795, 250], color=pv.AQUA, lw=2.8)      # slow cool
    ax.plot([-0.9, 0.18], [795, 150], color=pv.RED, lw=2.8)        # fast quench
    ax.text(2.02, 490, "slow cool $\\to$ pearlite\n(soft, tough)",
            fontsize=9, color="#0f7f57", fontweight="bold", ha="left", va="center")
    ax.annotate("fast quench $\\to$ martensite\n(hard, brittle)", (0.11, 200),
                xytext=(0.42, 176), fontsize=9, color=pv.RED,
                fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=pv.RED, lw=1.2))
    ax.annotate("nose", (0.03, Tn), xytext=(0.62, 600), fontsize=8.6,
                color=pv.INK2, arrowprops=dict(arrowstyle="->", color=pv.MUTED, lw=1.1))

    ax.set_xlabel("log time $\\to$")
    ax.set_ylabel("temperature")
    ax.set_xlim(-1.0, 3.2)
    ax.set_ylim(140, 800)
    # framed legend at upper left so it does not collide with the A_e1 label;
    # the frame masks the cooling paths passing beneath it
    leg = ax.legend(fontsize=8.4, loc="upper left", frameon=True)
    leg.get_frame().set_facecolor(pv.SURFACE)
    leg.get_frame().set_edgecolor(pv.GRID)
    pv.titles(ax, "Time--temperature--transformation (TTT) diagram",
              "cooling rate decides the product: pearlite vs martensite")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ms_quench.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_griffith():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: energy balance U(a) = 4*gamma*a - pi*sigma^2*a^2/E  (per unit thickness)
    a = np.linspace(0, 2.4, 400)
    gamma, E, sigma = 1.0, 1.0, 1.0
    U_surf = 4 * gamma * a
    U_strain = -np.pi * sigma**2 * a**2 / E
    U_tot = U_surf + U_strain
    a_star = 2 * gamma * E / (np.pi * sigma**2)

    axL.axhline(0, color=pv.BASELINE, lw=1)
    axL.plot(a, U_surf, color=pv.AQUA, label="surface energy  +4$\\gamma a$")
    axL.plot(a, U_strain, color=pv.BLUE, label="released strain energy  $-\\pi\\sigma^2 a^2/E$")
    axL.plot(a, U_tot, color=pv.INK, lw=2.6, label="total energy $U(a)$")
    axL.axvline(a_star, color=pv.RED, ls="--", lw=1.4)
    imax = int(np.argmin(np.abs(a - a_star)))
    axL.plot([a_star], [U_tot[imax]], "o", color=pv.RED, ms=8, zorder=6)
    axL.annotate("critical crack $a^{*}=\\dfrac{2\\gamma E}{\\pi\\sigma^{2}}$\n"
                 "(energy maximum $\\Rightarrow$ unstable)",
                 (a_star, U_tot[imax]), xytext=(1.35, 4.4), textcoords="data",
                 fontsize=8.5, color=pv.RED, fontweight="bold", va="top",
                 arrowprops=dict(arrowstyle="->", color=pv.RED, lw=1.1))
    axL.set_xlabel("crack half-length $a$")
    axL.set_ylabel("energy")
    axL.set_yticks([])
    axL.legend(fontsize=8, loc="lower left")
    pv.titles(axL, "Griffith energy balance",
              "a flaw grows only past the energy peak")

    # right: critical stress vs crack size  sigma_c = sqrt(2 E gamma / (pi a))
    a2 = np.linspace(0.05, 3.0, 400)
    sig_c = np.sqrt(2 * E * gamma / (np.pi * a2))
    axR.plot(a2, sig_c, color=pv.INK, lw=2.4)
    axR.fill_between(a2, sig_c, sig_c.max()*1.05, color=pv.AQUA, alpha=0.10)
    axR.fill_between(a2, 0, sig_c, color=pv.RED, alpha=0.08)
    axR.text(2.0, sig_c[int(0.66*len(a2))]+0.55, "safe\n(sub-critical)", color=pv.INK2,
             fontsize=9, ha="center")
    axR.text(2.2, 0.35, "fracture", color=pv.RED, fontsize=9.5, fontweight="bold", ha="center")
    axR.set_xlabel("flaw size $a$")
    axR.set_ylabel("applied stress $\\sigma$")
    axR.set_ylim(0, sig_c.max()*1.05)
    axR.annotate("$\\sigma_c\\propto a^{-1/2}$: raising the applied stress\n"
                 "makes a pre-existing, harmless flaw supercritical",
                 (0.9, np.sqrt(2*E*gamma/(np.pi*0.9))), xytext=(0.9, 3.4),
                 fontsize=8.5, color=pv.INK, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=pv.INK, lw=1.2))
    pv.titles(axR, "Critical crack size vs applied stress",
              "$\\sigma_c\\propto a^{-1/2}$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ms_griffith.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_paris():
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    dK = np.logspace(0, 2, 500)
    dKth, Kc = 3.0, 70.0
    m, C = 3.2, 1e-9
    paris = C * dK**m
    # region I threshold blow-down and region III blow-up
    f_low = np.clip((dK - dKth) / dK, 1e-6, None)**1.6
    f_high = np.clip(Kc / (Kc - dK), 1, 1e6)
    dadN = paris * f_low * np.clip(f_high, 1, 40)
    good = (dK > dKth*1.02) & (dK < Kc*0.985)
    ax.loglog(dK[good], dadN[good], color=pv.INK, lw=2.6)

    # highlight the Paris (linear) middle regime
    mid = (dK > 6) & (dK < 35)
    ax.loglog(dK[mid], (C*dK**m)[mid], color=pv.BLUE, lw=4, alpha=0.35)
    ax.axvline(dKth, color=pv.MUTED, ls=":", lw=1.2)
    ax.axvline(Kc, color=pv.RED, ls="--", lw=1.4)
    ax.text(4.1, 1.15e-9, "threshold\n$\\Delta K_{th}$ (endurance limit)",
            color=pv.INK2, fontsize=8.5, va="bottom")
    ax.text(Kc*0.62, 2e-4, "fast fracture\n$\\to K_c$", color=pv.RED,
            fontsize=8.5, ha="right", fontweight="bold")
    ax.annotate("Region II: Paris law\n$da/dN = C\\,(\\Delta K)^{m}$\nslope $=m$",
                (14, C*14**m), xytext=(4.3, 3e-4),
                fontsize=9.5, color=pv.BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=pv.BLUE, lw=1.3))
    ax.set_xlabel("stress-intensity range $\\Delta K$")
    ax.set_ylabel("crack growth per cycle $da/dN$")
    ax.set_ylim(1e-9, 1e-2)
    ax.set_xticks([3, 5, 10, 20, 40, 70])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    pv.titles(ax, "The Paris fatigue law",
              "each sub-yield cycle grows the crack; below $\\Delta K_{th}$ it does not")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ms_paris.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_weibull():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    s = np.linspace(0, 2.2, 500)
    s0 = 1.0
    for m, col, lab, dy in [(1, pv.YELLOW, "$m=1$ (broad)", 6),
                            (4, pv.BLUE, "$m=4$", -18),
                            (20, pv.AQUA, "$m=20$ (narrow)", 6)]:
        Pf = 1 - np.exp(-(s / s0)**m)
        axL.plot(s, Pf, color=col, lw=2.4)
        idx = int(0.62 * len(s))
        pv.direct_label(axL, s[idx], Pf[idx], lab, col, dx=4, dy=dy, fontsize=8.5,
                        va="top" if dy < 0 else "center")
    axL.axvline(1.0, color=pv.MUTED, ls=":", lw=1)
    axL.set_xlabel("stress $\\sigma / \\sigma_0$")
    axL.set_ylabel("failure probability $P_f$")
    axL.text(0.06, 0.97, "$P_f = 1-\\exp[-(\\sigma/\\sigma_0)^{m}]$\n"
             "low modulus $\\Rightarrow$ breaks\nat wildly varying stress",
             fontsize=8.3, color=pv.INK, va="top")
    pv.titles(axL, "Weibull failure statistics",
              "a higher modulus $m$ means a narrower strength distribution")

    # right: size effect  sigma_mean ~ V^(-1/m)
    V = np.logspace(0, 4, 400)
    for m, col in [(4, pv.BLUE), (12, pv.AQUA)]:
        strength = V**(-1.0/m)
        axR.semilogx(V, strength, color=col, lw=2.4)
        pv.direct_label(axR, V[280], V[280]**(-1.0/m), f"$m={m}$", col, dx=4, dy=4, fontsize=9)
    axR.set_xlabel("specimen volume $V$")
    axR.set_ylabel("mean strength  $\\propto V^{-1/m}$")
    axR.annotate("weakest-link scaling:\nlarger specimens are weaker",
                 (30, 30**(-1.0/4)), xytext=(400, 0.245), fontsize=8.3,
                 color=pv.INK, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=pv.INK, lw=1.1))
    pv.titles(axR, "The statistical size effect",
              "larger bodies sample more of the flaw distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ms_weibull.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_nucleation():
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    r = np.linspace(0, 3.2, 500)
    gamma = 1.0
    dGv = 1.0  # magnitude of (negative) volume free-energy density
    surf = 4 * np.pi * gamma * r**2
    vol = -(4.0/3.0) * np.pi * dGv * r**3
    tot = surf + vol
    r_star = 2 * gamma / dGv
    istar = int(np.argmin(np.abs(r - r_star)))

    ax.axhline(0, color=pv.BASELINE, lw=1)
    ax.plot(r, surf, color=pv.AQUA, lw=2, label="surface term $+4\\pi r^{2}\\gamma$")
    ax.plot(r, vol, color=pv.BLUE, lw=2, label="volume term $-\\frac{4}{3}\\pi r^{3}\\Delta G_v$")
    ax.plot(r, tot, color=pv.INK, lw=2.8, label="total $\\Delta G(r)$")
    ax.plot([r_star], [tot[istar]], "o", color=pv.RED, ms=9, zorder=6)
    ax.annotate("$r^{*}=\\dfrac{2\\gamma}{\\Delta G_v}$,   barrier $\\Delta G^{*}$",
                (r_star, tot[istar]), xytext=(8, 8), textcoords="offset points",
                color=pv.RED, fontsize=9.5, fontweight="bold")
    ax.annotate("", (r_star, 0), (r_star, tot[istar]),
                arrowprops=dict(arrowstyle="<->", color=pv.RED, lw=1.2))
    # heterogeneous: lowered barrier
    tot_het = 0.35 * tot
    ax.plot(r, tot_het, color=pv.YELLOW, lw=2, ls="--")
    ihet = int(np.argmin(np.abs(r - 2.35)))
    ax.annotate("heterogeneous (on defects):\nlower barrier",
                (r[ihet], tot_het[ihet]), xytext=(1.95, tot[istar]*1.60),
                color=pv.YELLOW, fontsize=8.3, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=pv.YELLOW, lw=1.1))
    ax.set_ylim(tot.min()*1.4, surf.max()*0.5)
    ax.set_xlabel("nucleus radius $r$")
    ax.set_ylabel("free-energy change $\\Delta G$")
    ax.set_yticks([])
    ax.legend(fontsize=8.4, loc="upper right")
    ax.text(0.35, tot.min()*1.15, "below $r^{*}$: redissolves\nabove $r^{*}$: grows spontaneously",
            fontsize=8.6, color=pv.INK, fontweight="bold")
    pv.titles(ax, "Classical nucleation of a new phase",
              "sub-critical nuclei redissolve; super-critical ones grow")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ms_nucleation.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_jmak():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    t = np.linspace(0, 3, 500)
    tau = 1.0
    for n, col in [(1, pv.YELLOW), (2, pv.BLUE), (4, pv.AQUA)]:
        X = 1 - np.exp(-(t / tau)**n)
        axL.plot(t, X, color=col, lw=2.4, label=f"$n={n}$")
    # linear null for contrast
    axL.plot(t, np.clip(0.5*t, 0, 1), color=pv.MUTED, ls=":", lw=1.6,
             label="linear null")
    axL.legend(fontsize=8.5, loc="lower right")
    axL.set_xlabel("time $t$")
    axL.set_ylabel("transformed fraction $X$")
    axL.annotate("$X=1-\\exp[-(t/\\tau)^{n}]$",
                 (0.1, 0.9), xytext=(0.1, 0.9), fontsize=9.5, color=pv.INK)
    pv.titles(axL, "JMAK / Avrami transformation kinetics",
              "sigmoid, not a line; the exponent $n$ names the mechanism")

    # right: Avrami plot ln(-ln(1-X)) vs ln t -> slope n
    tt = np.linspace(0.15, 2.5, 200)
    for n, col in [(1, pv.YELLOW), (2, pv.BLUE), (4, pv.AQUA)]:
        X = 1 - np.exp(-(tt / tau)**n)
        y = np.log(-np.log(1 - X))
        axR.plot(np.log(tt), y, color=col, lw=2.4)
    axR.text(0.88, 3.75, "slope $n=4$", color=pv.AQUA, fontsize=8.5,
             fontweight="bold", ha="right", va="bottom")
    axR.text(0.88, 1.62, "slope $n=2$", color=pv.BLUE, fontsize=8.5,
             fontweight="bold", ha="right", va="top")
    axR.text(0.88, 0.35, "slope $n=1$", color=pv.YELLOW, fontsize=8.5,
             fontweight="bold", ha="right", va="top")
    axR.set_xlabel("$\\ln t$")
    axR.set_ylabel("$\\ln(-\\ln(1-X))$")
    axR.text(-1.95, 3.95, "mechanism diagnosis:\n$n\\approx1$ site-saturated growth\n"
             "$n\\approx4$ continuous nucleation, 3-D growth",
             fontsize=8.3, color=pv.INK, fontweight="bold", va="top")
    pv.titles(axR, "Avrami plot", "a straight line whose slope is $n$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ms_jmak.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_grain_boundary():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    # left: Read-Shockley gamma(theta) = gamma0 * theta * (A - ln theta)
    th = np.linspace(0.3, 62, 500)  # degrees
    thm = 15.0
    thr = np.radians(th)
    thmr = np.radians(thm)
    # Read-Shockley up to theta_m, then saturate
    A = 1 - np.log(thmr)
    gamma_low = thr * (A - np.log(thr))
    g_at_m = thmr * (A - np.log(thmr))
    gamma = np.where(th <= thm, gamma_low, g_at_m)
    gamma = gamma / g_at_m  # normalise saturation to 1
    axL.plot(th, gamma, color=pv.INK, lw=2.6)
    axL.axvline(thm, color=pv.MUTED, ls=":", lw=1.2)
    axL.text(thm+1, 0.35, "low-angle\n$\\to$ high-angle", color=pv.INK2, fontsize=8.5)
    axL.scatter([3], [np.interp(3, th, gamma)], color=pv.AQUA, s=60, zorder=6)
    pv.direct_label(axL, 3, np.interp(3, th, gamma), "low-angle\nboundary", pv.AQUA, dx=6, dy=-2, fontsize=8)
    axL.scatter([55], [np.interp(55, th, gamma)], color=pv.RED, s=60, zorder=6)
    pv.direct_label(axL, 55, np.interp(55, th, gamma), "high-angle\nboundary", pv.RED, dx=-6, dy=-14, fontsize=8, ha="right", va="top")
    axL.set_xlabel("misorientation angle $\\theta$  (degrees)")
    axL.set_ylabel("boundary energy $\\gamma$ (normalised)")
    axL.annotate("Read–Shockley\n$\\gamma=\\gamma_0\\,\\theta\\,(A-\\ln\\theta)$",
                 (8, np.interp(8, th, gamma)), xytext=(20, 0.55), fontsize=8.6,
                 color=pv.INK, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=pv.INK, lw=1.1))
    pv.titles(axL, "Grain-boundary energy vs misorientation",
              "energy rises with misorientation, then saturates")

    # right: Hall-Petch and inverse Hall-Petch  sigma_y = s0 + k d^-1/2
    d = np.linspace(4, 3000, 600)  # nm
    dinv = d**-0.5
    sy = 1.0 + 6.0 * dinv
    dc = 18.0            # critical grain size for inverse HP (nm)
    dcinv = dc**-0.5     # critical d^-1/2 (peak location)
    peak = 1.0 + 6.0 * dcinv
    # below dc (dinv > dcinv): grains too small -> boundary sliding -> strength FALLS
    sy_inv = peak - 3.2 * (dinv - dcinv)
    axR.plot(dinv[d >= dc], sy[d >= dc], color=pv.BLUE, lw=2.6)
    axR.plot(dinv[d < dc], sy_inv[d < dc], color=pv.RED, lw=2.6, ls="--")
    axR.axvline(dcinv, color=pv.MUTED, ls=":", lw=1.1)
    axR.set_xlabel("$d^{-1/2}$  (finer grains $\\to$)")
    axR.set_ylabel("yield strength $\\sigma_y$")
    axR.set_ylim(1.0, peak + 0.3)
    pv.direct_label(axR, dinv[440], sy[440], "Hall–Petch:\nfiner = stronger\n$\\sigma_y=\\sigma_0+k\\,d^{-1/2}$",
                    pv.BLUE, dx=40, dy=-6, fontsize=8, ha="left")
    axR.annotate("inverse Hall–Petch:\ngrain-boundary sliding weakens",
                 (0.5, peak - 3.2*(0.5 - dcinv)), xytext=(0.30, 1.35),
                 fontsize=8, color=pv.RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=pv.RED, lw=1.1))
    pv.titles(axR, "Hall–Petch strengthening (and its reversal)",
              "finer grains are stronger, until they get too small")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ms_grain_boundary.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_arrhenius():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    T = np.linspace(300, 1400, 500)
    Q = 8000.0  # K (Q/k)
    Xv = np.exp(-Q / T)
    axL.plot(T, Xv, color=pv.INK, lw=2.6)
    axL.set_xlabel("temperature $T$")
    axL.set_ylabel("vacancy fraction $X_v$")
    axL.annotate("$X_v = \\exp(-Q_f/kT)$\nvacancies rise\nexponentially with $T$",
                 (1150, np.exp(-Q/1150)), xytext=(430, np.exp(-Q/1150)*0.9),
                 fontsize=8.6, color=pv.INK, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=pv.INK, lw=1.1))
    pv.titles(axL, "Equilibrium vacancy concentration",
              "the thermally generated defect population")

    # right: linearised Arrhenius  ln Xv vs 1/T
    invT = 1.0 / T
    axR.plot(invT * 1000, np.log(Xv), color=pv.BLUE, lw=2.6)
    axR.set_xlabel("$1000/T$")
    axR.set_ylabel("$\\ln X_v$")
    axR.annotate("straight line:\nslope $=-Q_f/k$",
                 (invT[250]*1000, np.log(Xv)[250]), xytext=(1.55, -21.5),
                 fontsize=8.8, color=pv.BLUE, fontweight="bold", va="top",
                 arrowprops=dict(arrowstyle="->", color=pv.BLUE, lw=1.1))
    pv.titles(axR, "Arrhenius linearisation",
              "the same activated exponential governs every thermal rate")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ms_arrhenius.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_potential_well()
    fig_griffith()
    fig_paris()
    fig_weibull()
    fig_nucleation()
    fig_jmak()
    fig_grain_boundary()
    fig_arrhenius()
    fig_quench()
    print("materials-science figures written to", os.path.abspath(FIGS))
