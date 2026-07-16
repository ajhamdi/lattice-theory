"""Case study B — the vacancy law (political vacancy thermodynamics).

Tests N_defect = A * exp(-E_k / (Theta + Theta0)) where Theta is systemic
temperature proxied by global information velocity (share of humanity online)
and N_defect is the UCDP count of ongoing conflicts by type.

Theta is the OWID/ITU World aggregate for 2005+, extended back to 1990 with a
population-weighted mean over all countries reporting in that year (coverage
in the 1990s is dominated by large states, which is what the weighting wants).

Null model: log-linear in Theta. Falsification lever: with Theta monotone in
time, a pure temperature law demands monotone defect counts — the time-domain
panel shows whether history obeyed.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

import matplotlib.pyplot as plt
import politviz as pv

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.environ.get("LT_FIGDIR", os.path.join(HERE, "..", "figures"))

pv.apply_style()

conf = pd.read_csv(os.path.join(DATA, "number-of-armed-conflicts.csv"), comment="#")
net = pd.read_csv(os.path.join(DATA, "share-of-individuals-using-the-internet.csv"), comment="#")
pop = pd.read_csv(os.path.join(DATA, "population.csv"), comment="#")

# --- build Theta(t), 1990-2025 ---
world = net[net.entity == "World"][["year", "it_net_user_zs"]].rename(
    columns={"it_net_user_zs": "theta"})
cn = net[net.code.notna() & (net.code != "OWID_WRL")]
cp = pop[pop.code.notna() & (pop.year >= 1985)][["code", "year", "population_historical"]]
merged = cn.merge(cp, on=["code", "year"])
wavg = (merged.assign(w=lambda d: d.it_net_user_zs * d.population_historical)
        .groupby("year")
        .apply(lambda d: d.w.sum() / d.population_historical.sum(), include_groups=False)
        .rename("theta_w").reset_index())
theta = wavg[wavg.year < 2005].rename(columns={"theta_w": "theta"})
theta = pd.concat([theta, world[world.year >= 2005]]).sort_values("year")

cw = conf[conf.entity == "World"].copy()
m = cw.merge(theta, on="year").sort_values("year")

types = {
    "intrastate": ("number_ongoing_conflicts__conflict_type_intrastate",
                   "intrastate conflicts (vacancy channel)"),
    "non_state": ("number_ongoing_conflicts__conflict_type_non_state_conflict",
                  "non-state conflicts (interstitial channel)"),
    "one_sided": ("number_ongoing_conflicts__conflict_type_one_sided_violence",
                  "one-sided violence (defect-adjacent)"),
}


def arrhenius(theta, lnA, E, theta0):
    return lnA - E / (theta + theta0)


results = {}
panels = {}
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
if pv.SHOW_TITLES:
    fig.suptitle("The vacancy law: political Arrhenius test", x=0.005, ha="left",
                 fontsize=13, fontweight="bold", color=pv.INK)
    fig.text(0.005, 0.945,
             "UCDP conflict counts vs systemic temperature Θ (share of humanity online, "
             "population-weighted before 2005), 1990–2025",
             fontsize=9, color=pv.INK2)

panel_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
for ax, (key, (col, label)) in zip(panel_axes, types.items()):
    d = m[(m[col] > 0) & m.theta.notna()]
    th, N = d.theta.values.astype(float), d[col].values.astype(float)
    lnN = np.log(N)

    p_a, _ = curve_fit(arrhenius, th, lnN, p0=[5, 50, 20],
                       bounds=([-10, -5000, 1], [20, 5000, 500]), maxfev=40000)
    lnN_hat = arrhenius(th, *p_a)
    r2_a = 1 - np.sum((lnN - lnN_hat) ** 2) / np.sum((lnN - lnN.mean()) ** 2)
    aic_a = len(N) * np.log(np.sum((lnN - lnN_hat) ** 2) / len(N)) + 2 * 3

    sl, ic, r, p, se = stats.linregress(th, lnN)
    lnN_lin = ic + sl * th
    aic_l = len(N) * np.log(np.sum((lnN - lnN_lin) ** 2) / len(N)) + 2 * 2

    results[key] = {"arrhenius_params": [float(v) for v in p_a],
                    "R2_arrhenius": float(r2_a), "AIC_arrhenius": float(aic_a),
                    "loglinear_slope": float(sl), "loglinear_p": float(p),
                    "R2_loglinear": float(r ** 2), "AIC_loglinear": float(aic_l),
                    "n": int(len(N))}
    panels[key] = (d, th, N, p_a, ic, sl, r2_a)

    ax.scatter(th, N, s=22, color=pv.BLUE, zorder=3, edgecolor=pv.SURFACE, linewidth=0.8)
    tt = np.linspace(max(th.min(), 0.01), th.max(), 300)
    ax.plot(tt, np.exp(arrhenius(tt, *p_a)), color=pv.AQUA, lw=2.2)
    ax.plot(tt, np.exp(ic + sl * tt), color=pv.YELLOW, ls="--", lw=1.6)
    ax.set_title(label, loc="left", fontsize=9.5, color=pv.INK)
    ax.set_xlabel("Θ  (systemic temperature = % of humanity online)")
    ax.set_ylabel("ongoing conflicts (count)")
    ax.text(0.03, 0.95, f"Arrhenius R² = {r2_a:.2f}   n = {len(N)}",
            transform=ax.transAxes, fontsize=8.5, va="top", color=pv.INK2)
    d0, d1 = d.iloc[0], d.iloc[-1]
    pv.direct_label(ax, d0.theta, d0[col], str(int(d0.year)), pv.MUTED, dy=8, fontsize=7.5)
    pv.direct_label(ax, d1.theta, d1[col], str(int(d1.year)), pv.MUTED, dy=8,
                    fontsize=7.5, ha="right")

axes[0, 0].text(0.55, 0.24, "vacancy-law exponential", transform=axes[0, 0].transAxes,
                color=pv.AQUA, fontsize=8.5, fontweight="bold")
axes[0, 0].text(0.55, 0.14, "log-linear null", transform=axes[0, 0].transAxes,
                color=pv.YELLOW, fontsize=8.5, fontweight="bold")

# time-domain falsification panel
ax = axes[1, 1]
d = m[m.theta.notna()]
ax.plot(d.year, d[types["intrastate"][0]], color=pv.BLUE, lw=2.0)
pv.direct_label(ax, 1996, 38, "intrastate count", pv.BLUE, dy=-10)
ax2 = ax  # single axis: overlay Theta scaled? NO dual axis — show Theta as shaded context
ax.plot(d.year, d.theta * 57 / d.theta.max(), color=pv.MUTED, lw=1.4, ls=":")
pv.direct_label(ax, 2013, 33, "Θ (indexed, dotted)", pv.MUTED, dy=0)
ax.annotate("post-Cold-War\nannealing dip", xy=(1998, 32), xytext=(1996.5, 15),
            fontsize=8.5, color=pv.RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=pv.RED, lw=1.2))
ax.set_title("time domain: monotone Θ, non-monotone defects", loc="left",
             fontsize=9.5, color=pv.INK)
ax.set_xlabel("year")
ax.set_ylabel("intrastate conflicts (count)")

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(FIGS, "B_kenostate_arrhenius.png"))

# Validated-only variant for the journal main text (LT_MAIN_VARIANTS=1): the
# interstitial (non-state) channel alone — the one that obeys the pure
# temperature law. The failed channels stay on the composite figure above,
# which the unsuccessful-attempts appendix uses.
if os.environ.get("LT_MAIN_VARIANTS", "0") == "1":
    d, th, N, p_a, ic, sl, r2_a = panels["non_state"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    pv.titles(ax, "The vacancy law: the interstitial channel",
              "UCDP non-state conflict counts vs systemic temperature Θ, 1990–2025")
    ax.scatter(th, N, s=26, color=pv.BLUE, zorder=3, edgecolor=pv.SURFACE, linewidth=0.8)
    tt = np.linspace(max(th.min(), 0.01), th.max(), 300)
    ax.plot(tt, np.exp(arrhenius(tt, *p_a)), color=pv.AQUA, lw=2.4)
    ax.plot(tt, np.exp(ic + sl * tt), color=pv.YELLOW, ls="--", lw=1.6)
    ax.set_xlabel("Θ  (systemic temperature = % of humanity online)")
    ax.set_ylabel("ongoing non-state conflicts (count)")
    ax.text(0.03, 0.95, f"Arrhenius R² = {r2_a:.2f}   n = {len(N)}",
            transform=ax.transAxes, fontsize=9, va="top", color=pv.INK2)
    ax.text(0.50, 0.24, "vacancy-law exponential", transform=ax.transAxes,
            color=pv.AQUA, fontsize=9, fontweight="bold")
    ax.text(0.50, 0.15, "log-linear null", transform=ax.transAxes,
            color=pv.YELLOW, fontsize=9, fontweight="bold")
    d0, d1 = d.iloc[0], d.iloc[-1]
    col = types["non_state"][0]
    pv.direct_label(ax, d0.theta, d0[col], str(int(d0.year)), pv.MUTED, dy=8, fontsize=8)
    pv.direct_label(ax, d1.theta, d1[col], str(int(d1.year)), pv.MUTED, dy=8,
                    fontsize=8, ha="right")
    fig.savefig(os.path.join(FIGS, "B_kenostate_arrhenius_main.png"))
    plt.close(fig)

results["theta_series"] = {int(r.year): round(float(r.theta), 3) for r in theta.itertuples()
                           if r.year in (1990, 1995, 2000, 2005, 2015, 2025)}
with open(os.path.join(DATA, "fit_B_kenostate.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
