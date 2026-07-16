"""Case study C — the imperial Weibull law (weakest-link statistics of imperial failure).

Two tests on 43 empires (Taagepera peak areas, conventional lifetimes):
  1. Distribution test: are imperial lifetimes Weibull-distributed, and is the
     shape parameter beta (the "imperial Weibull modulus") > 1 (wear-out ageing)
     or ~= 1 (memoryless failure, i.e. the exponential null of Arbesman 2011)?
  2. Size-effect test: weakest-link scaling predicts lifetime ~ Area^(-1/beta).
     OLS on log-log tests whether bigger empires actually die sooner.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import politviz as pv

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.environ.get("LT_FIGDIR", os.path.join(HERE, "..", "figures"))

pv.apply_style()
rng = np.random.default_rng(42)

df = pd.read_csv(os.path.join(DATA, "empires.csv"), comment="#")
df["lifetime"] = df.end_year - df.start_year
tau = df.lifetime.values.astype(float)
A = df.peak_area_Mm2.values.astype(float)
n = len(tau)

# --- 1. Weibull MLE vs exponential null ---
beta, loc, eta = stats.weibull_min.fit(tau, floc=0)
ll_w = np.sum(stats.weibull_min.logpdf(tau, beta, 0, eta))
lam = tau.mean()
ll_e = np.sum(stats.expon.logpdf(tau, 0, lam))
aic_w, aic_e = -2 * ll_w + 2 * 2, -2 * ll_e + 2 * 1

boot = [stats.weibull_min.fit(rng.choice(tau, n, replace=True), floc=0)[0]
        for _ in range(2000)]
b_lo, b_hi = np.percentile(boot, [2.5, 97.5])

# --- 2. size effect ---
sl, ic, r, p, se = stats.linregress(np.log(A), np.log(tau))
pred_slope = -1 / beta

results = {
    "beta": float(beta), "beta_CI95": [float(b_lo), float(b_hi)],
    "eta_years": float(eta), "mean_lifetime": float(lam),
    "AIC_weibull": float(aic_w), "AIC_exponential": float(aic_e),
    "size_slope": float(sl), "size_slope_se": float(se),
    "size_r": float(r), "size_p": float(p),
    "predicted_slope": float(pred_slope), "n_empires": n,
}

# ---- figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
if pv.SHOW_TITLES:
    fig.suptitle("The imperial Weibull law: weakest-link statistics of imperial failure",
                 x=0.005, ha="left", fontsize=13, fontweight="bold", color=pv.INK)
    fig.text(0.005, 0.92, f"{n} empires, peak areas from Taagepera; lifetimes = conventional founding→dissolution",
             fontsize=9, color=pv.INK2)

# panel 1: Weibull probability plot
ts = np.sort(tau)
F = (np.arange(1, n + 1) - 0.3) / (n + 0.4)      # Bernard median ranks
x, y = np.log(ts), np.log(-np.log(1 - F))
ax1.scatter(x, y, s=24, color=pv.BLUE, zorder=3, edgecolor=pv.SURFACE, linewidth=0.8)
xx = np.linspace(x.min() - 0.2, x.max() + 0.2, 100)
ax1.plot(xx, beta * (xx - np.log(eta)), color=pv.AQUA, lw=2.2)
ax1.plot(xx, 1.0 * (xx - np.log(lam)), color=pv.YELLOW, ls="--", lw=1.6)
ax1.text(0.03, 0.95,
         f"imperial Weibull modulus β = {beta:.2f}\n95% CI [{b_lo:.2f}, {b_hi:.2f}]\n"
         f"AIC: Weibull {aic_w:.0f} vs exponential {aic_e:.0f}",
         transform=ax1.transAxes, fontsize=8.5, va="top", color=pv.INK2)
ax1.text(0.66, 0.30, "Weibull fit", transform=ax1.transAxes, color=pv.AQUA,
         fontsize=8.5, fontweight="bold")
ax1.text(0.66, 0.18, "exponential null\n(memoryless, β=1)", transform=ax1.transAxes,
         color=pv.YELLOW, fontsize=8.5, fontweight="bold")
ax1.set_xlabel("ln lifetime (years)")
ax1.set_ylabel("ln(−ln(1−F))")
ax1.set_title("Do empires age? (probability plot)", loc="left", fontsize=10, color=pv.INK)

# panel 2: size effect
ax2.scatter(np.log10(A), np.log10(tau), s=24, color=pv.BLUE, zorder=3,
            edgecolor=pv.SURFACE, linewidth=0.8)
for _, row in df.iterrows():
    label_offsets = {"British Empire": (4, 4), "Macedonian Empire": (6, -2),
                     "Byzantine Empire": (4, 4), "Xin dynasty": (-6, -10),
                     "Aztec Empire": (4, 4), "Mongol Empire": (4, 4),
                     "Ottoman Empire": (4, 4)}
    if row["name"] in label_offsets:
        ax2.annotate(row["name"].replace(" Empire", "").replace(" dynasty", ""),
                     (np.log10(row.peak_area_Mm2), np.log10(row.lifetime)),
                     xytext=label_offsets[row["name"]], textcoords="offset points",
                     fontsize=7.5, color=pv.MUTED)
la = np.linspace(np.log10(A.min()), np.log10(A.max()), 50)
ax2.plot(la, (ic + sl * (la / np.log10(np.e))) / np.log(10), color=pv.YELLOW, lw=2.0)
med = np.median(np.log(tau))
ax2.plot(la, (med + pred_slope * (la * np.log(10) - np.median(np.log(A)))) / np.log(10),
         color=pv.AQUA, ls="--", lw=2.0)
ax2.text(0.03, 0.24,
         f"observed slope = {sl:.2f} ± {se:.2f}  (p = {p:.2f})",
         transform=ax2.transAxes, fontsize=8.5, color=pv.YELLOW, fontweight="bold")
ax2.text(0.03, 0.145,
         f"weakest-link prediction −1/β = {pred_slope:.2f}",
         transform=ax2.transAxes, fontsize=8.5, color=pv.AQUA, fontweight="bold")
ax2.set_xlabel("log₁₀ peak area (10⁶ km²)")
ax2.set_ylabel("log₁₀ lifetime (years)")
ax2.set_title("Do bigger empires die sooner? (size effect)", loc="left", fontsize=10, color=pv.INK)

fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(os.path.join(FIGS, "C_weibull_empires.png"))

# Validated-only variant for the journal main text (LT_MAIN_VARIANTS=1): the
# Weibull probability plot alone. The falsified size-effect panel stays on the
# composite figure above, which the unsuccessful-attempts appendix uses.
if os.environ.get("LT_MAIN_VARIANTS", "0") == "1":
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    pv.titles(ax, "Do empires age?",
              f"Weibull probability plot, {n} imperial lifetimes (Taagepera)")
    ax.scatter(x, y, s=26, color=pv.BLUE, zorder=3, edgecolor=pv.SURFACE, linewidth=0.8)
    ax.plot(xx, beta * (xx - np.log(eta)), color=pv.AQUA, lw=2.4)
    ax.plot(xx, 1.0 * (xx - np.log(lam)), color=pv.YELLOW, ls="--", lw=1.6)
    ax.text(0.03, 0.95,
            f"imperial Weibull modulus β = {beta:.2f}\n95% CI [{b_lo:.2f}, {b_hi:.2f}]\n"
            f"AIC: Weibull {aic_w:.0f} vs exponential {aic_e:.0f}",
            transform=ax.transAxes, fontsize=9, va="top", color=pv.INK2)
    ax.text(0.66, 0.30, "Weibull fit", transform=ax.transAxes, color=pv.AQUA,
            fontsize=9, fontweight="bold")
    ax.text(0.66, 0.16, "exponential null\n(memoryless, β=1)", transform=ax.transAxes,
            color=pv.YELLOW, fontsize=9, fontweight="bold")
    ax.set_xlabel("ln lifetime (years)")
    ax.set_ylabel("ln(−ln(1−F))")
    fig.savefig(os.path.join(FIGS, "C_weibull_empires_main.png"))
    plt.close(fig)

with open(os.path.join(DATA, "fit_C_weibull.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
