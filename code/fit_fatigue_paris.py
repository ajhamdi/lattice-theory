"""Case study D — the Paris-Sarajevo fatigue law.

If each managed crisis is a fatigue cycle growing the crack (da/dN = C*(dK)^m),
crack growth accelerates toward fracture, so inter-crisis intervals should
compress as a power of remaining time: dt_n = c * (t_f - t_n)^p with p > 0.

Test 1 (calibration): pre-WWI Europe, where t_f = July 1914 is known.
Test 2 (falsification): India-Pakistan 1947-2025 and Taiwan Strait 1954-2022,
where the law predicts monotone interval compression (Kendall tau < 0 between
cycle index and interval length).
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import politviz as pv

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.environ.get("LT_FIGDIR", os.path.join(HERE, "..", "figures"))

pv.apply_style()

df = pd.read_csv(os.path.join(DATA, "crisis_sequences.csv"), comment="#")
results = {}

# --- pre-WWI power-law fit ---
pw = df[df.arena == "prewwi"].year.values
t_f = pw[-1]                       # July 1914 fracture
onsets = pw[:-1]                   # the five sub-critical cycles
dt = np.diff(pw)[: len(onsets) - 0]  # interval after each onset (last one ends at fracture)
rem = t_f - onsets                 # remaining time at each onset
# drop the final interval (it ends *at* fracture by construction -> not independent)
x, y = np.log(rem[:-1]), np.log(dt[:-1])
sl, ic, r, p, se = stats.linregress(x, y)
results["prewwi"] = {"exponent_p": float(sl), "se": float(se), "r2": float(r**2),
                     "pvalue": float(p), "n_intervals": int(len(x))}

# --- compression tests for open-ended arenas ---
for arena in ("indpak", "taiwan"):
    yrs = df[df.arena == arena].year.values
    ivals = np.diff(yrs)
    tau, pk = stats.kendalltau(np.arange(len(ivals)), ivals)
    results[arena] = {"intervals": [float(v) for v in ivals],
                      "kendall_tau": float(tau), "kendall_p": float(pk),
                      "n_intervals": int(len(ivals))}

# ---- figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
if pv.SHOW_TITLES:
    fig.suptitle("The Paris–Sarajevo fatigue law: do crisis cycles compress toward fracture?",
                 x=0.005, ha="left", fontsize=13, fontweight="bold", color=pv.INK)
    fig.text(0.005, 0.92,
             "Left: pre-WWI Europe (known fracture, July 1914). Right: open-ended rivalries — "
             "the law demands monotone interval compression",
             fontsize=9, color=pv.INK2)

# panel 1
ax1.scatter(rem[:-1], dt[:-1], s=30, color=pv.BLUE, zorder=3,
            edgecolor=pv.SURFACE, linewidth=0.8)
labels = ["Tangier 1905", "Bosnia 1908", "Agadir 1911", "1st Balkan 1912"]
for xi, yi, lab in zip(rem[:-1], dt[:-1], labels):
    ax1.annotate(lab, (xi, yi), xytext=(5, 4), textcoords="offset points",
                 fontsize=7.5, color=pv.MUTED)
xx = np.linspace(rem[:-1].min() * 0.8, rem[:-1].max() * 1.2, 50)
ax1.plot(xx, np.exp(ic) * xx ** sl, color=pv.AQUA, lw=2.2)
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xticks([2, 3, 4, 6, 10])
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.xaxis.set_minor_formatter(NullFormatter())
ax1.text(0.03, 0.95, f"Δt ∝ (t_f − t)^p,  p = {sl:.2f} ± {se:.2f}\nR² = {r**2:.2f}  (n = 4; calibration only)",
         transform=ax1.transAxes, fontsize=8.5, va="top", color=pv.INK2)
ax1.text(0.60, 0.20, "fatigue power law", transform=ax1.transAxes, color=pv.AQUA,
         fontsize=8.5, fontweight="bold")
ax1.set_xlabel("time remaining to fracture, t_f − t (years)")
ax1.set_ylabel("interval to next crisis (years)")
ax1.set_title("Pre-WWI Europe, 1905–1914", loc="left", fontsize=10, color=pv.INK)

# panel 2
series = [("prewwi", "pre-WWI Europe", pv.BLUE),
          ("indpak", "India–Pakistan", pv.AQUA),
          ("taiwan", "Taiwan Strait", pv.YELLOW)]
for arena, lab, color in series:
    yrs = df[df.arena == arena].year.values
    ivals = np.diff(yrs)
    idx = np.arange(1, len(ivals) + 1)
    ax2.plot(idx, ivals, marker="o", ms=5, color=color, lw=1.8,
             markeredgecolor=pv.SURFACE, markeredgewidth=0.8)
    if arena == "taiwan":
        # label below the sequence tail, clear of the compression-test note
        pv.direct_label(ax2, 2.7, 19, lab, color, dx=0)
    else:
        pv.direct_label(ax2, idx[-1], ivals[-1], lab, color, dx=6)
ax2.set_yscale("log")
ax2.set_xlabel("cycle number")
ax2.set_ylabel("interval between crises (years)")
ax2.set_title("Interval sequences (log scale)", loc="left", fontsize=10, color=pv.INK)
ax2.set_xlim(0.5, 14.5)
tau_ip = results["indpak"]["kendall_tau"]
p_ip = results["indpak"]["kendall_p"]
ax2.text(0.98, 0.97, f"India–Pakistan compression test:\nKendall τ = {tau_ip:.2f} (p = {p_ip:.2f})",
         transform=ax2.transAxes, fontsize=8.5, color=pv.INK2, ha="right", va="top")

fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(os.path.join(FIGS, "D_paris_sarajevo.png"))

# Validated-only variant for the journal main text (LT_MAIN_VARIANTS=1): the
# pre-WWI calibration arena alone (known fracture date, no healing). The
# open-ended rivalry arenas stay on the composite figure above, which the
# unsuccessful-attempts appendix uses.
if os.environ.get("LT_MAIN_VARIANTS", "0") == "1":
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    pv.titles(ax, "Pre-WWI Europe, 1905–1914",
              "inter-crisis intervals vs time remaining to the July 1914 fracture")
    ax.scatter(rem[:-1], dt[:-1], s=34, color=pv.BLUE, zorder=3,
               edgecolor=pv.SURFACE, linewidth=0.8)
    for xi, yi, lab in zip(rem[:-1], dt[:-1], labels):
        ax.annotate(lab, (xi, yi), xytext=(5, 4), textcoords="offset points",
                    fontsize=8, color=pv.MUTED)
    xxm = np.linspace(rem[:-1].min() * 0.8, rem[:-1].max() * 1.2, 50)
    ax.plot(xxm, np.exp(ic) * xxm ** sl, color=pv.AQUA, lw=2.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([2, 3, 4, 6, 10])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.text(0.03, 0.95,
            f"Δt ∝ (t_f − t)^p,  p = {sl:.2f} ± {se:.2f}\nR² = {r**2:.2f}  (n = 4; calibration only)",
            transform=ax.transAxes, fontsize=9, va="top", color=pv.INK2)
    ax.text(0.60, 0.20, "fatigue power law", transform=ax.transAxes, color=pv.AQUA,
            fontsize=9, fontweight="bold")
    ax.set_xlabel("time remaining to fracture, t_f − t (years)")
    ax.set_ylabel("interval to next crisis (years)")
    fig.savefig(os.path.join(FIGS, "D_paris_sarajevo_main.png"))
    plt.close(fig)

with open(os.path.join(DATA, "fit_D_fatigue.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
