"""Part IV — registered forward predictions, 2026 -> 2031.

Every coefficient is FROZEN from the Part III fits (read from data/fit_*.json
or refit on the identical windows); nothing is tuned to make the future look
good. Outputs: figures/E_predictions_2031.png and data/fit_E_predictions.json.

P1  Non-state conflicts 2030 from the vacancy-law interstitial Arrhenius channel.
P2  Intrastate conflicts 2028-2030 floor from the amended (post-annealing) law.
P3  Democratic fraction 2030 from the reverse-transformation trend.
P4  Next India-Pakistan crisis window from the Paris-Sarajevo compression fit.
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
pred = {"registered": "2026-07-08"}

# ---------- rebuild Theta(t) exactly as in Case Study B ----------
net = pd.read_csv(os.path.join(DATA, "share-of-individuals-using-the-internet.csv"), comment="#")
pop = pd.read_csv(os.path.join(DATA, "population.csv"), comment="#")
world = net[net.entity == "World"][["year", "it_net_user_zs"]].rename(
    columns={"it_net_user_zs": "theta"})
cn = net[net.code.notna() & (net.code != "OWID_WRL")]
cp = pop[pop.code.notna() & (pop.year >= 1985)][["code", "year", "population_historical"]]
merged = cn.merge(cp, on=["code", "year"])
wavg = (merged.assign(w=lambda d: d.it_net_user_zs * d.population_historical)
        .groupby("year")
        .apply(lambda d: d.w.sum() / d.population_historical.sum(), include_groups=False)
        .rename("theta").reset_index())
theta = pd.concat([wavg[wavg.year < 2005], world[world.year >= 2005]]).sort_values("year")
theta = theta[theta.year >= 1990]

# Theta projection: logistic saturation fit on 1990-2025
def logi(t, L, tm, s):
    return L / (1 + np.exp(-(t - tm) / s))

pL, _ = curve_fit(logi, theta.year, theta.theta, p0=[85, 2012, 6],
                  bounds=([74, 2000, 2], [100, 2030, 20]))
years_f = np.arange(2026, 2032)
theta_f = logi(years_f, *pL)
pred["theta_projection"] = {int(y): round(float(v), 1) for y, v in zip(years_f, theta_f)}

# ---------- P1 / P2: conflict channels from frozen Arrhenius params ----------
with open(os.path.join(DATA, "fit_B_kenostate.json")) as f:
    fitB = json.load(f)

conf = pd.read_csv(os.path.join(DATA, "number-of-armed-conflicts.csv"), comment="#")
cw = conf[conf.entity == "World"].merge(theta, on="year")
COLS = {"non_state": "number_ongoing_conflicts__conflict_type_non_state_conflict",
        "intrastate": "number_ongoing_conflicts__conflict_type_intrastate"}

proj = {}
for key, col in COLS.items():
    lnA, E, th0 = fitB[key]["arrhenius_params"]
    lnN_hat = lnA - E / (cw.theta.values + th0)
    resid = np.log(cw[col].values) - lnN_hat
    sig = float(np.std(resid, ddof=3))
    Nf = np.exp(lnA - E / (theta_f + th0))
    proj[key] = {"point": Nf, "lo": Nf * np.exp(-1.96 * sig), "hi": Nf * np.exp(1.96 * sig),
                 "sigma_ln": sig}
    pred[f"P_{key}_2030"] = {"point": round(float(Nf[years_f == 2030][0]), 1),
                             "lo95": round(float((Nf * np.exp(-1.96 * sig))[years_f == 2030][0]), 1),
                             "hi95": round(float((Nf * np.exp(1.96 * sig))[years_f == 2030][0]), 1)}

# ---------- P3: reverse transformation of the democratic fraction ----------
row = pd.read_csv(os.path.join(DATA, "row_regimes.csv"), comment="#")
w = row[row.entity == "World"].copy()
cols = [c for c in w.columns if c.startswith("num_countries")]
w["total"] = w[cols].sum(axis=1)
w["X"] = (w["num_countries_regime__category_electoral_democracy"]
          + w["num_countries_regime__category_liberal_democracy"]) / w.total
w = w.sort_values("year")
rev = w[w.year >= 2016]                      # the sustained decline window
slr = stats.linregress(rev.year, rev.X)
X2030 = slr.intercept + slr.slope * 2030
# prediction band from residual scatter + slope uncertainty
Xres = rev.X - (slr.intercept + slr.slope * rev.year)
sigX = float(np.sqrt(np.std(Xres, ddof=2) ** 2 + (slr.stderr * (2030 - rev.year.mean())) ** 2))
peak2016 = float(w[w.year.between(2005, 2020)].X.max())
pred["P_demfrac_2030"] = {"point": round(float(X2030), 3),
                          "lo95": round(float(X2030 - 1.96 * sigX), 3),
                          "hi95": round(float(X2030 + 1.96 * sigX), 3),
                          "slope_per_year": round(float(slr.slope), 4),
                          "slope_p": round(float(slr.pvalue), 4),
                          "falsification_line_peak": round(peak2016, 3)}

# ---------- P4: India-Pakistan compression -> next-crisis window ----------
cr = pd.read_csv(os.path.join(DATA, "crisis_sequences.csv"), comment="#")
ip = cr[cr.arena == "indpak"].year.values
onsets, ivals = ip[:-1], np.diff(ip)

def sse_for_tf(tf):
    x = np.log(tf - onsets)
    res = stats.linregress(x, np.log(ivals))
    return np.sum((np.log(ivals) - (res.intercept + res.slope * x)) ** 2), res

tf_grid = np.arange(2026.0, 2101.0, 0.25)
sses = np.array([sse_for_tf(tf)[0] for tf in tf_grid])
tf_hat = float(tf_grid[np.argmin(sses)])
_, res_hat = sse_for_tf(tf_hat)
# next interval predicted from the last crisis (2025.35)
last = ip[-1]
dt_next = float(np.exp(res_hat.intercept) * (tf_hat - last) ** res_hat.slope)
# a flat SSE profile means tf is weakly identified; report the profile width
tf_band = tf_grid[sses <= sses.min() * 1.10]
pred["P_indpak"] = {"tf_hat": tf_hat, "tf_profile_10pct": [float(tf_band.min()), float(tf_band.max())],
                    "exponent_p": round(float(res_hat.slope), 2),
                    "dt_next_years": round(dt_next, 2),
                    "next_crisis_point": round(last + dt_next, 2),
                    "last_crisis": float(last)}

# ---------- figure ----------
fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))
if pv.SHOW_TITLES:
    fig.suptitle("Registered predictions, 2026 → 2031 (coefficients frozen from Part III)",
                 x=0.005, ha="left", fontsize=13, fontweight="bold", color=pv.INK)
    fig.text(0.005, 0.945, "Filed 2026-07-08 — each panel carries its numeric falsification "
             "criterion; resolution 2029–2031", fontsize=9, color=pv.INK2)

# panel 1: Theta
ax = axes[0, 0]
ax.scatter(theta.year, theta.theta, s=16, color=pv.BLUE, zorder=3,
           edgecolor=pv.SURFACE, linewidth=0.6)
tt = np.linspace(1990, 2031, 300)
ax.plot(tt, logi(tt, *pL), color=pv.AQUA, lw=2.0, ls="--")
ax.axvspan(2026, 2031, color=pv.AQUA, alpha=0.07)
pv.direct_label(ax, 2014, 12, "observed Θ", pv.BLUE, dx=0)
pv.direct_label(ax, 1991, 62, "saturation fit\n→ projection", pv.AQUA, dx=0)
ax.text(0.97, 0.06, f"Θ(2030) ≈ {logi(2030, *pL):.0f}",
        transform=ax.transAxes, fontsize=8.5, color=pv.INK2, ha="right")
ax.set_title("systemic temperature Θ and its projection", loc="left", fontsize=9.5, color=pv.INK)
ax.set_xlabel("year"); ax.set_ylabel("Θ (systemic temperature, 0–100)")

# panel 2: conflict channels
ax = axes[0, 1]
series = [("non_state", "non-state (interstitial)", pv.BLUE),
          ("intrastate", "intrastate (vacancy)", pv.AQUA)]
for key, lab, color in series:
    col = COLS[key]
    ax.plot(cw.year, cw[col], color=color, lw=1.8)
    ax.plot(years_f, proj[key]["point"], color=color, lw=2.0, ls="--")
    ax.fill_between(years_f, proj[key]["lo"], proj[key]["hi"], color=color, alpha=0.12, lw=0)
    pv.direct_label(ax, 2031, proj[key]["point"][-1], lab, color, dx=6)
ax.axvspan(2026, 2031, color=pv.MUTED, alpha=0.05)
ax.set_xlim(1989, 2043)
ax.set_title("P1/P2: defect counts under frozen Arrhenius laws", loc="left",
             fontsize=9.5, color=pv.INK)
ax.set_xlabel("year"); ax.set_ylabel("ongoing conflicts (count)")
p1 = pred["P_non_state_2030"]
ax.text(0.03, 0.95, f"P1: non-state 2030 = {p1['point']:.0f}  [{p1['lo95']:.0f}, {p1['hi95']:.0f}]",
        transform=ax.transAxes, fontsize=8.5, va="top", color=pv.INK2)

# panel 3: democratic fraction
ax = axes[1, 0]
w9 = w[w.year >= 1990]
ax.plot(w9.year, w9.X, color=pv.BLUE, lw=1.8)
tt = np.linspace(2016, 2031, 60)
ax.plot(tt, slr.intercept + slr.slope * tt, color=pv.RED, lw=2.0, ls="--")
ax.fill_between(tt, slr.intercept + slr.slope * tt - 1.96 * sigX,
                slr.intercept + slr.slope * tt + 1.96 * sigX,
                color=pv.RED, alpha=0.10, lw=0)
ax.axhline(peak2016, color=pv.MUTED, lw=1.2, ls=":")
ax.text(2042, peak2016 + 0.004, f"falsification line: return to peak ({peak2016:.2f})",
        fontsize=8, color=pv.MUTED, ha="right", va="bottom")
pv.direct_label(ax, 1996, 0.44, "observed X_D", pv.BLUE, dx=0)
pv.direct_label(ax, 2031, slr.intercept + slr.slope * 2031, "reverse-transformation\ntrend", pv.RED, dx=6)
p3 = pred["P_demfrac_2030"]
ax.text(0.03, 0.08, f"P3: X_D(2030) = {p3['point']:.3f}  [{p3['lo95']:.3f}, {p3['hi95']:.3f}]",
        transform=ax.transAxes, fontsize=8.5, color=pv.INK2)
ax.set_xlim(1990, 2043)
ax.set_title("P3: the reverse transformation continues", loc="left", fontsize=9.5, color=pv.INK)
ax.set_xlabel("year"); ax.set_ylabel("fraction of states democratic")

# panel 4: India-Pakistan next-crisis window
ax = axes[1, 1]
ax.scatter(onsets, ivals, s=26, color=pv.BLUE, zorder=3,
           edgecolor=pv.SURFACE, linewidth=0.8)
tt = np.linspace(1947, 2031, 400)
ax.plot(tt, np.exp(res_hat.intercept) * np.maximum(tf_hat - tt, 0.3) ** res_hat.slope,
        color=pv.AQUA, lw=2.0, ls="--")
nxt = pred["P_indpak"]["next_crisis_point"]
ax.axvspan(2027, 2031.9, color=pv.RED, alpha=0.10)
ax.axvline(nxt, color=pv.RED, lw=1.6, ls="--")
ax.text(nxt - 3.0, 15.5, f"P4: next crisis ≈ {nxt:.0f}\n(window 2027–2031)",
        fontsize=8.5, color=pv.RED, fontweight="bold", ha="right")
pv.direct_label(ax, 1972, 12.4, "observed intervals", pv.BLUE, dx=6)
pv.direct_label(ax, 1955, 4.2, "compression fit", pv.AQUA, dx=0)
ax.set_title("P4: India–Pakistan interval compression", loc="left", fontsize=9.5, color=pv.INK)
ax.set_xlabel("crisis onset year"); ax.set_ylabel("interval to next crisis (years)")

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(FIGS, "E_predictions_2031.png"))

with open(os.path.join(DATA, "fit_E_predictions.json"), "w") as f:
    json.dump(pred, f, indent=2)
print(json.dumps(pred, indent=2))
