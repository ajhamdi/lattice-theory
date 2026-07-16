"""Case study A — the Avrami-Huntington transformation law.

Tests whether the Third Wave of democratization (1974->peak) follows
JMAK phase-transformation kinetics X(t) = X0 + (Xf-X0)*(1 - exp(-((t-t0)/tau)^n))
against logistic and linear null models. Data: OWID / V-Dem Regimes of the
World counts, 1789-2024.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import politviz as pv

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.environ.get("LT_FIGDIR", os.path.join(HERE, "..", "figures"))

pv.apply_style()

df = pd.read_csv(os.path.join(DATA, "row_regimes.csv"), comment="#")
w = df[df.entity == "World"].copy()
cols = [c for c in w.columns if c.startswith("num_countries")]
w["total"] = w[cols].sum(axis=1)
w["dem"] = (w["num_countries_regime__category_electoral_democracy"]
            + w["num_countries_regime__category_liberal_democracy"])
w["X"] = w.dem / w.total
w = w.sort_values("year").reset_index(drop=True)

T0 = 1974  # onset of Huntington's Third Wave (Carnation Revolution)
post = w[w.year >= T0]
peak_year = int(post.loc[post.X.idxmax(), "year"])
win = w[(w.year >= T0) & (w.year <= peak_year)]
t, X = win.year.values.astype(float), win.X.values


def jmak(t, X0, Xf, tau, n):
    return X0 + (Xf - X0) * (1 - np.exp(-((t - T0) / tau) ** n))


def logistic(t, X0, Xf, tm, s):
    return X0 + (Xf - X0) / (1 + np.exp(-(t - tm) / s))


def linear(t, a, b):
    return a + b * (t - T0)


def aic(y, yhat, k):
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    return n * np.log(rss / n) + 2 * k, rss


fits = {}
p_j, _ = curve_fit(jmak, t, X, p0=[0.25, 0.6, 20, 1.5],
                   bounds=([0, 0, 1, 0.2], [1, 1, 200, 6]), maxfev=20000)
p_l, _ = curve_fit(logistic, t, X, p0=[0.25, 0.6, 1990, 5],
                   bounds=([0, 0, 1974, 0.5], [1, 1, 2030, 50]), maxfev=20000)
p_lin, _ = curve_fit(linear, t, X)

for name, fn, p in [("jmak", jmak, p_j), ("logistic", logistic, p_l), ("linear", linear, p_lin)]:
    yhat = fn(t, *p)
    a, rss = aic(X, yhat, len(p))
    ss_tot = float(np.sum((X - X.mean()) ** 2))
    fits[name] = {"params": [float(v) for v in p], "AIC": a, "RSS": rss,
                  "R2": 1 - rss / ss_tot}

n_AH = fits["jmak"]["params"][3]

# ---- figure ----
fig, ax = plt.subplots(figsize=(9.5, 5.2))
pv.titles(ax, "The Avrami–Huntington transformation law",
          f"Democratic fraction of all states, 1789–{int(w.year.max())} (V-Dem/RoW via OWID) — "
          f"JMAK fit on the Third Wave {T0}–{peak_year}")

ax.plot(w.year, w.X, color=pv.BLUE, lw=1.8)
ax.text(1855, 0.29, "observed democratic\nfraction X_D", color=pv.BLUE,
        fontsize=9, fontweight="bold", ha="center")

tt = np.linspace(T0, 2024, 400)
ax.plot(tt, jmak(tt, *p_j), color=pv.AQUA, ls="-", lw=2.2)
ax.plot(tt, logistic(tt, *p_l), color=pv.YELLOW, ls="--", lw=1.8)
ax.text(2028, 0.555, f"JMAK fit  n_AH = {n_AH:.2f}", color=pv.AQUA,
        fontsize=9, fontweight="bold", ha="left", va="bottom")
ax.text(2028, 0.505, "logistic null", color=pv.YELLOW,
        fontsize=9, fontweight="bold", ha="left", va="center")

# actual post-peak divergence (the reverse transformation)
post_peak = w[w.year >= peak_year]
ax.plot(post_peak.year, post_peak.X, color=pv.RED, lw=2.0)
ax.text(2028, 0.455, "reverse\ntransformation", color=pv.RED,
        fontsize=9, fontweight="bold", ha="left", va="top")

for yr, lab in [(1828, "1st wave"), (1922, "1st reverse"), (1943, "2nd wave"),
                (1958, "2nd reverse"), (1974, "3rd wave")]:
    ax.axvline(yr, color=pv.GRID, lw=0.8)
    ax.text(yr + 1, 0.66, lab, rotation=90, fontsize=7.5, color=pv.MUTED, va="top")

ax.axvspan(T0, peak_year, color=pv.AQUA, alpha=0.06)
ax.set_xlim(1789, 2058)
ax.set_ylim(0, 0.70)
ax.set_xlabel("year")
ax.set_ylabel("fraction of states democratic")

txt = (f"AIC:  JMAK {fits['jmak']['AIC']:.0f}   logistic {fits['logistic']['AIC']:.0f}   "
       f"linear {fits['linear']['AIC']:.0f}    (lower = better; JMAK R² = {fits['jmak']['R2']:.3f})")
ax.text(0.01, -0.20, txt, transform=ax.transAxes, fontsize=8.5, color=pv.INK2)

fig.savefig(os.path.join(FIGS, "A_avrami_huntington.png"))

# Validated-only variant for the journal main text (LT_MAIN_VARIANTS=1): the
# JMAK/logistic fits drawn over the fit window only, without the post-peak
# reverse-transformation overlay (that panel belongs to the unsuccessful-
# attempts appendix, which uses the full figure above).
if os.environ.get("LT_MAIN_VARIANTS", "0") == "1":
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    pv.titles(ax, "The Avrami–Huntington transformation law",
              f"Democratic fraction of all states, 1789–{int(w.year.max())} — "
              f"JMAK fit on the Third Wave {T0}–{peak_year}")
    ax.plot(w.year, w.X, color=pv.BLUE, lw=1.8)
    ax.text(1855, 0.29, "observed democratic\nfraction X_D", color=pv.BLUE,
            fontsize=9, fontweight="bold", ha="center")
    tw = np.linspace(T0, peak_year, 300)
    ax.plot(tw, jmak(tw, *p_j), color=pv.AQUA, ls="-", lw=2.4)
    ax.plot(tw, logistic(tw, *p_l), color=pv.YELLOW, ls="--", lw=1.8)
    ax.text(2033, 0.585, f"JMAK fit  n_AH = {n_AH:.2f}", color=pv.AQUA,
            fontsize=9, fontweight="bold", ha="right", va="bottom")
    ax.text(2033, 0.43, "logistic null", color=pv.YELLOW,
            fontsize=9, fontweight="bold", ha="right", va="top")
    for yr, lab in [(1828, "1st wave"), (1922, "1st reverse"), (1943, "2nd wave"),
                    (1958, "2nd reverse"), (1974, "3rd wave")]:
        ax.axvline(yr, color=pv.GRID, lw=0.8)
        ax.text(yr + 1, 0.66, lab, rotation=90, fontsize=7.5, color=pv.MUTED, va="top")
    ax.axvspan(T0, peak_year, color=pv.AQUA, alpha=0.06)
    ax.set_xlim(1789, 2035)
    ax.set_ylim(0, 0.70)
    ax.set_xlabel("year")
    ax.set_ylabel("fraction of states democratic")
    ax.text(0.01, -0.20, txt, transform=ax.transAxes, fontsize=8.5, color=pv.INK2)
    fig.savefig(os.path.join(FIGS, "A_avrami_huntington_main.png"))
    plt.close(fig)

fits["meta"] = {"third_wave_start": T0, "peak_year": peak_year,
                "n_points": len(t), "n_AH": n_AH}
with open(os.path.join(DATA, "fit_A_avrami.json"), "w") as f:
    json.dump(fits, f, indent=2)
print(json.dumps(fits, indent=2))
