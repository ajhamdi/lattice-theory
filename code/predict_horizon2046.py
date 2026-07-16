"""Part IV, far cohort — registered 20-year forecasts (2026 -> 2046).

Unlike the near cohort (predict_2031.py), these forecasts bring their OWN
datasets (polities.csv, un_admissions.csv, war_durations.csv,
nuclear_states.csv, two new arenas in crisis_sequences.csv) but reuse the
SAME counter-equations, and where a coefficient exists from Part III it is
frozen (beta, eta from fit_C; Arrhenius params from fit_B). Every forecast
returns an outcome AND an expected year derived from the equations.

F1  Imperial Weibull hazard (Eq. 3): conditional 20-yr rupture probability for
    15 current constitutional orders; median first-rupture year.
F2  Reverse-transformation trough (Eq. 1): trough year of the third reverse
    wave and nucleation window of the fourth wave.
F3  Nucleation freeze (Eq. 5): UN-admission gap statistics; burst
    prediction conditional on F1.
V   Out-of-sample validation: Paris-Sarajevo timing machinery retrodicts the
    Armenia-Azerbaijan fracture (fit through 2022, fracture was 2023.72).
F4  Kosovo-Serbia compression (Eq. 4): fracture-or-settlement window.
F5  Thermal plateau (Eq. 2 + Theta saturation): conflict counts stop rising.
F6  10th nuclear state (Eq. 5 with healing): probability and median year.
F7  Ukraine frozen conflict (Eq. 3 on war durations): armistice year, no treaty.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from scipy.special import gamma as Gamma

import matplotlib.pyplot as plt
import politviz as pv

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.environ.get("LT_FIGDIR", os.path.join(HERE, "..", "figures"))

pv.apply_style()
NOW = 2026.5
out = {"registered": "2026-07-08", "horizon": 2046}

# ================= F1: imperial Weibull hazard on current polities =================
with open(os.path.join(DATA, "fit_C_weibull.json")) as f:
    fitC = json.load(f)
BETA, ETA, MEANLIFE = fitC["beta"], fitC["eta_years"], fitC["mean_lifetime"]
B_LO, B_HI = fitC["beta_CI95"]


def cum_hazard(t, beta=BETA, eta=ETA):
    return (np.asarray(t, dtype=float) / eta) ** beta


def p_rupture(age, horizon, beta=BETA, eta=ETA):
    """P(rupture within `horizon` yrs | survived to `age`)."""
    return 1 - np.exp(cum_hazard(age, beta, eta) - cum_hazard(age + horizon, beta, eta))


def median_remaining(age, beta=BETA, eta=ETA, q=0.5):
    return eta * (cum_hazard(age, beta, eta) - np.log(q)) ** (1 / beta) - age


pol = pd.read_csv(os.path.join(DATA, "polities.csv"), comment="#")
pol["age"] = NOW - pol.founded
pol["p20"] = p_rupture(pol.age, 20)
# sensitivity: sweep beta over its bootstrap CI holding the observed mean lifetime fixed
for tag, b in [("lo", B_LO), ("hi", B_HI)]:
    eta_b = MEANLIFE / Gamma(1 + 1 / b)
    pol[f"p20_{tag}"] = p_rupture(pol.age, 20, b, eta_b)
pol["median_year"] = NOW + median_remaining(pol.age)
pol = pol.sort_values("p20", ascending=True).reset_index(drop=True)

# aggregate: P(at least one rupture among the 15) as a function of year
yy = np.arange(0, 60.25, 0.25)
ln_surv = np.array([(cum_hazard(pol.age) - cum_hazard(pol.age + y)).sum() for y in yy])
p_any = 1 - np.exp(ln_surv)
p_any_2046 = float(np.interp(20, yy, p_any))
median_first = float(NOW + np.interp(0.5, p_any, yy))
expected_count = float(pol.p20.sum())

out["F1_hazard"] = {
    "beta_frozen": round(BETA, 3), "eta_frozen": round(ETA, 1),
    "table": {r["name"]: {"age": round(r.age, 0), "p20": round(r.p20, 3),
                          "p20_range": [round(r.p20_lo, 3), round(r.p20_hi, 3)],
                          "median_rupture_year": round(r.median_year, 0)}
              for _, r in pol.iterrows()},
    "P_at_least_one_by_2046": round(p_any_2046, 2),
    "expected_ruptures_by_2046": round(expected_count, 2),
    "median_first_rupture_year": round(median_first, 1),
}

# ================= F2: reverse-transformation trough =================
row = pd.read_csv(os.path.join(DATA, "row_regimes.csv"), comment="#")
w = row[row.entity == "World"].copy()
cols = [c for c in w.columns if c.startswith("num_countries")]
w["total"] = w[cols].sum(axis=1)
w["X"] = (w["num_countries_regime__category_electoral_democracy"]
          + w["num_countries_regime__category_liberal_democracy"]) / w.total
w = w.sort_values("year").reset_index(drop=True)
# onset of the sustained decline: the last year attaining the post-2004 maximum (2016) -
# the same window the near cohort's P3 uses. The 2003-04 global peak is followed by an
# overshoot-relaxation plateau, not yet the wave.
sub = w[w.year.between(2005, 2025)]
T0R = float(sub.year[sub.X >= sub.X.max() - 0.005].max())   # last touch of the high-water mark
X0R = float(sub.loc[sub.year == T0R, "X"].iloc[0])
rev = w[w.year >= T0R]


def jmak_rev(t, Xf, tau, n):
    return Xf + (X0R - Xf) * np.exp(-(np.maximum(t - T0R, 0) / tau) ** n)


p_rev, _ = curve_fit(jmak_rev, rev.year, rev.X, p0=[0.40, 15, 1.5],
                     bounds=([0.10, 3, 0.6], [0.50, 60, 4]), maxfev=20000)
Xf_r, tau_r, n_r = [float(v) for v in p_rev]
jmak_usable = (tau_r < 55) and (Xf_r > 0.105)            # bound-pinned params = no curvature yet
trough_jmak = T0R + tau_r * np.log(20) ** (1 / n_r)     # 95% of the decay done
# primary timing estimator: mean duration of the two historical reverse waves
REV_HIST = [(1942 - 1922), (1975 - 1958)]
trough_hist = T0R + float(np.mean(REV_HIST))
trough_window = [T0R + min(REV_HIST) - 4, T0R + max(REV_HIST) + 4]
# floor estimators: JMAK asymptote and the P3 linear trend evaluated at the trough
slr_rev = stats.linregress(rev.year, rev.X)
floor_linear = float(slr_rev.intercept + slr_rev.slope * trough_hist)
resid_r = rev.X - jmak_rev(rev.year, *p_rev)
out["F2_fourth_wave"] = {
    "onset_year": T0R, "onset_X": round(X0R, 3),
    "jmak_reverse": {"Xf": round(Xf_r, 3), "tau": round(tau_r, 1), "n": round(n_r, 2),
                     "floor_and_timing_identified": bool(jmak_usable),
                     "R2": round(1 - float((resid_r ** 2).sum())
                                 / float(((rev.X - rev.X.mean()) ** 2).sum()), 3)},
    "trough_year_jmak": round(trough_jmak, 1),
    "trough_year_registered": round(trough_hist, 1),
    "trough_window": [round(v, 0) for v in trough_window],
    "floor_jmak": round(Xf_r, 3), "floor_linear_at_trough": round(floor_linear, 3),
    "fourth_wave_onset_by": round(trough_window[1] + 5, 0),
}

# ================= F3: nucleation freeze (UN admissions) =================
un = pd.read_csv(os.path.join(DATA, "un_admissions.csv"), comment="#")
adm_years = un[un.year > 1945].year.values          # exclude the founding cohort
gaps = np.diff(adm_years)
open_gap = NOW - adm_years.max()
# burstiness: Fano factor of admissions per 5-yr bin, 1946-2025
bins = np.arange(1946, 2027, 5)
counts, _ = np.histogram(np.repeat(un[un.year > 1945].year, un[un.year > 1945].n_admitted),
                         bins=bins)
fano = float(counts.var(ddof=1) / counts.mean())
out["F3_freeze"] = {
    "last_admission": int(adm_years.max()), "open_gap_years": round(float(open_gap), 1),
    "max_closed_gap_years": int(gaps.max()), "median_closed_gap_years": float(np.median(gaps)),
    "fano_factor_5yr": round(fano, 1),
    "expected_first_admission": round(median_first + 2, 0),
}

# ================= V + F4: Paris-Sarajevo timing, two new arenas =================
cr = pd.read_csv(os.path.join(DATA, "crisis_sequences.csv"), comment="#")


def fit_tf(events, tf_grid):
    onsets, ivals = events[:-1], np.diff(events)
    best = None
    sses = []
    for tf in tf_grid:
        x = np.log(tf - onsets)
        res = stats.linregress(x, np.log(ivals))
        sse = float(np.sum((np.log(ivals) - (res.intercept + res.slope * x)) ** 2))
        sses.append(sse)
        if best is None or sse < best[0]:
            best = (sse, float(tf), res)
    sses = np.array(sses)
    band = tf_grid[sses <= sses.min() * 1.10]
    return best[1], best[2], (float(band.min()), float(band.max())), onsets, ivals


# validation: fit Armenia-Azerbaijan on cycles through 2022 only; fracture was 2023.72
az = cr[cr.arena == "armaz"].year.values
az_fit = az[:-1]                                    # hold out the terminal event
tf_az, res_az, band_az, on_az, iv_az = fit_tf(az_fit, np.arange(2023.0, 2061.0, 0.25))
out["V_armaz_retrodiction"] = {
    "fit_through": float(az_fit[-1]), "tf_hat": tf_az, "tf_profile_10pct": list(band_az),
    "exponent_p": round(float(res_az.slope), 2),
    "actual_fracture": float(az[-1]),
    "error_years": round(tf_az - float(az[-1]), 2),
}

ko = cr[cr.arena == "kosovo"].year.values
tf_ko, res_ko, band_ko, on_ko, iv_ko = fit_tf(ko, np.arange(2025.5, 2061.0, 0.25))
tau_kendall = stats.kendalltau(np.arange(len(iv_ko)), iv_ko)
dt_ko = float(np.exp(res_ko.intercept) * max(tf_ko - ko[-1], 0.05) ** res_ko.slope)
out["F4_kosovo"] = {
    "n_events": len(ko), "last_event": float(ko[-1]),
    "kendall_tau": round(float(tau_kendall.statistic), 2),
    "kendall_p": round(float(tau_kendall.pvalue), 3),
    "tf_hat": tf_ko, "tf_profile_10pct": list(band_ko),
    "exponent_p": round(float(res_ko.slope), 2),
    "next_event_point": round(ko[-1] + dt_ko, 1),
}

# ================= F5: thermal plateau of the defect budget =================
net = pd.read_csv(os.path.join(DATA, "share-of-individuals-using-the-internet.csv"), comment="#")
popdf = pd.read_csv(os.path.join(DATA, "population.csv"), comment="#")
world = net[net.entity == "World"][["year", "it_net_user_zs"]].rename(
    columns={"it_net_user_zs": "theta"})
cn = net[net.code.notna() & (net.code != "OWID_WRL")]
cp = popdf[popdf.code.notna() & (popdf.year >= 1985)][["code", "year", "population_historical"]]
merged = cn.merge(cp, on=["code", "year"])
wavg = (merged.assign(w=lambda d: d.it_net_user_zs * d.population_historical)
        .groupby("year")
        .apply(lambda d: d.w.sum() / d.population_historical.sum(), include_groups=False)
        .rename("theta").reset_index())
theta = pd.concat([wavg[wavg.year < 2005], world[world.year >= 2005]]).sort_values("year")
theta = theta[theta.year >= 1990]


def logi(t, L, tm, s):
    return L / (1 + np.exp(-(t - tm) / s))


pL, _ = curve_fit(logi, theta.year, theta.theta, p0=[85, 2012, 6],
                  bounds=([74, 2000, 2], [100, 2030, 20]))
L_theta = float(pL[0])
plateau_theta_year = float(pL[1] + pL[2] * np.log(0.99 / 0.01))   # Theta = 0.99 L

with open(os.path.join(DATA, "fit_B_kenostate.json")) as f:
    fitB = json.load(f)
conf = pd.read_csv(os.path.join(DATA, "number-of-armed-conflicts.csv"), comment="#")
cw = conf[conf.entity == "World"].merge(theta, on="year")
NS_COL = "number_ongoing_conflicts__conflict_type_non_state_conflict"
lnA, E, th0 = fitB["non_state"]["arrhenius_params"]
sig_ns = float(np.std(np.log(cw[NS_COL].values) - (lnA - E / (cw.theta.values + th0)), ddof=3))
years_far = np.arange(2026, 2047)
theta_far = logi(years_far, *pL)
ns_far = np.exp(lnA - E / (theta_far + th0))
ns_2046 = float(ns_far[-1])
ratio_law = float(ns_far[(years_far >= 2040)].mean() / ns_far[(years_far >= 2030)
                                                              & (years_far <= 2035)].mean())
# linear null: 2010-2025 trend extrapolated
recent = cw[cw.year >= 2010]
lin = stats.linregress(recent.year, recent[NS_COL])
null_2046 = float(lin.intercept + lin.slope * 2046)
ratio_null = float((lin.intercept + lin.slope * np.arange(2040, 2047)).mean()
                   / (lin.intercept + lin.slope * np.arange(2030, 2036)).mean())
out["F5_plateau"] = {
    "theta_ceiling_L": round(L_theta, 1), "theta_99pct_year": round(plateau_theta_year, 0),
    "non_state_2046": {"point": round(ns_2046, 0),
                       "lo95": round(ns_2046 * np.exp(-1.96 * sig_ns), 0),
                       "hi95": round(ns_2046 * np.exp(1.96 * sig_ns), 0)},
    "plateau_ratio_law": round(ratio_law, 2), "plateau_ratio_null": round(ratio_null, 2),
    "linear_null_2046": round(null_2046, 0),
}

# ================= F6: the 10th nuclear state =================
nuc = pd.read_csv(os.path.join(DATA, "nuclear_states.csv"), comment="#")
tests = np.sort(nuc[nuc.status == "declared"].first_test.values)
NPT = 1970.17
gaps_n = np.diff(tests)
pre = gaps_n[tests[1:] <= NPT]
post = gaps_n[tests[1:] > NPT]
lam_post, lam_pre = 1 / post.mean(), 1 / pre.mean()
p_by_2046 = 1 - np.exp(-lam_post * (2046 - NOW))
median_10th = NOW + np.log(2) / lam_post
out["F6_nuclear"] = {
    "n_declared": len(tests), "last": float(tests[-1]),
    "elapsed_since_last": round(NOW - tests[-1], 1),
    "mean_gap_preNPT": round(float(pre.mean()), 1),
    "mean_gap_postNPT": round(float(post.mean()), 1),
    "healing_factor": round(float(post.mean() / pre.mean()), 1),
    "P_10th_by_2046": round(float(p_by_2046), 2),
    "median_10th_year": round(float(median_10th), 1),
    "median_if_healing_collapses": round(float(np.log(2) / lam_pre), 1),
}

# ================= F7: Ukraine frozen conflict =================
wars = pd.read_csv(os.path.join(DATA, "war_durations.csv"), comment="#")
done = wars[wars.termination != "ongoing_censored"]
dur = done.duration_yrs.astype(float).values
bw, _, ew = stats.weibull_min.fit(dur, floc=0)
ukr_age = NOW - float(wars[wars.termination == "ongoing_censored"].start.iloc[0])
m50 = median_remaining(ukr_age, bw, ew, q=0.5)
m90 = median_remaining(ukr_age, bw, ew, q=0.10)     # 90% of wars this old end within m90
treaty_at_term = int((done.termination == "treaty_at_termination").sum())
frozen = int(done.termination.str.contains("frozen").sum())
out["F7_ukraine"] = {
    "n_completed_wars": len(dur), "weibull_shape": round(float(bw), 2),
    "weibull_scale": round(float(ew), 2),
    "ukraine_age_2026": round(ukr_age, 2),
    "median_cessation_year": round(NOW + m50, 1),
    "p90_cessation_year": round(NOW + m90, 1),
    "share_treaty_at_termination": round(treaty_at_term / len(dur), 2),
    "share_frozen": round(frozen / len(dur), 2),
}

# =========================== figure 1: hazard ===========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.6),
                               gridspec_kw={"width_ratios": [1.15, 1]})
if pv.SHOW_TITLES:
    fig.suptitle("F1 — The imperial Weibull hazard of the present order (Eq. 3, β and η frozen from the 43-empire fit)",
                 x=0.005, ha="left", fontsize=12.5, fontweight="bold", color=pv.INK)
    fig.text(0.005, 0.925, "conditional P(rupture within 20 yr | survived to 2026); whiskers sweep β over its 95% CI "
             "holding the observed mean lifetime fixed", fontsize=9, color=pv.INK2)

ypos = np.arange(len(pol))
ax1.barh(ypos, pol.p20, height=0.62, color=pv.BLUE, zorder=3)
ax1.hlines(ypos, pol.p20_lo, pol.p20_hi, color=pv.INK2, lw=1.1, zorder=4)
for i, r in pol.iterrows():
    ax1.text(max(r.p20, r.p20_hi) + 0.009, i, f"{r.p20:.0%}", va="center",
             fontsize=8, color=pv.INK2)
ax1.set_yticks(ypos)
ax1.set_yticklabels([n.split(" (")[0] for n in pol.name], fontsize=8.5)
ax1.set_xlabel("P(rupture by 2046)")
ax1.set_xlim(0, max(pol.p20_hi.max(), pol.p20.max()) * 1.18)
ax1.set_title("the oldest orders carry the highest hazard", loc="left",
              fontsize=9.5, color=pv.INK)

ax2.plot(NOW + yy, p_any, color=pv.AQUA, lw=2.2)
pv.direct_label(ax2, NOW + 34, float(np.interp(34, yy, p_any)),
                "P(≥1 of the 15 ruptures)", pv.AQUA, dx=6, dy=-6)
ax2.axvspan(NOW, 2046, color=pv.MUTED, alpha=0.06)
ax2.axhline(0.5, color=pv.GRID, lw=1.0)
ax2.axvline(median_first, color=pv.RED, lw=1.6, ls="--")
ax2.text(median_first + 0.8, 0.03, f"median first rupture ≈ {median_first:.0f}",
         fontsize=8.5, color=pv.RED, fontweight="bold", va="bottom")
ax2.text(0.03, 0.93, f"P(≥1 rupture by 2046) = {p_any_2046:.2f}\n"
         f"expected ruptures by 2046 = {expected_count:.1f}",
         transform=ax2.transAxes, fontsize=9, va="top", color=pv.INK2)
ax2.set_xlabel("year"); ax2.set_ylabel("cumulative probability")
ax2.set_xlim(NOW, NOW + 40); ax2.set_ylim(0, 1)
ax2.set_title("when does the first one land?", loc="left", fontsize=9.5, color=pv.INK)

fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(FIGS, "F_hazard_2046.png"))

# =========================== figure 2: timing panels ===========================
fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.8))
if pv.SHOW_TITLES:
    fig.suptitle("F2–F7 — Timing forecasts from the counter-equations (registered 2026-07-08, horizon 2046)",
                 x=0.005, ha="left", fontsize=13, fontweight="bold", color=pv.INK)
    fig.text(0.005, 0.945, "each panel yields an outcome AND an expected year; new datasets, same equations, "
             "frozen coefficients where they exist", fontsize=9, color=pv.INK2)

# (a) F2 reverse wave trough
ax = axes[0, 0]
w9 = w[w.year >= 1975]
ax.plot(w9.year, w9.X, color=pv.BLUE, lw=1.8)
tt = np.linspace(T0R, 2046, 200)
ax.plot(tt, jmak_rev(tt, *p_rev), color=pv.AQUA, lw=2.0, ls="--")
ax.axvspan(*trough_window, color=pv.RED, alpha=0.08)
ax.axvline(trough_hist, color=pv.RED, lw=1.5, ls="--")
ax.text(1977, 0.615, f"registered trough ≈ {trough_hist:.0f}\n(historical reverse-wave duration)",
        fontsize=8, color=pv.RED, fontweight="bold", va="top", ha="left")
ax.axhline(0.40, color=pv.RED, lw=1.0, ls=":")
ax.text(1998, 0.394, "floor claim:\nX_D never below 0.40", fontsize=8, color=pv.RED,
        va="top")
pv.direct_label(ax, 1990, 0.42, "observed X_D", pv.BLUE, dx=-8, dy=10)
jmak_lab = ("reverse JMAK\n(floor X_f = %.2f)" % Xf_r if jmak_usable
            else "reverse JMAK\n(floor not yet identifiable)")
pv.direct_label(ax, 2036, jmak_rev(2036, *p_rev), jmak_lab, pv.AQUA, dx=4, dy=-18)
ax.set_xlim(1975, 2046); ax.set_ylim(0.28, 0.62)
ax.set_title("F2: third reverse wave → trough → fourth wave", loc="left", fontsize=9.5, color=pv.INK)
ax.set_xlabel("year"); ax.set_ylabel("democratic fraction X_D")

# (b) F3 UN admissions
ax = axes[0, 1]
ax.bar(un[un.year > 1945].year, un[un.year > 1945].n_admitted, width=0.9,
       color=pv.BLUE, zorder=3)
ax.axvspan(2011.6, NOW, color=pv.RED, alpha=0.10)
ax.text(2028.5, 9.5, "15-yr freeze\n(record; prev.\nmax = 6 yr)", fontsize=8,
        color=pv.RED, fontweight="bold", ha="left")
ax.annotate("decolonization burst\n(UK/French fracture)", (1962, 14.5), fontsize=8,
            color=pv.INK2, ha="left", xytext=(1968, 15.5),
            arrowprops=dict(arrowstyle="-", color=pv.MUTED, lw=0.8))
ax.annotate("Soviet/Yugoslav\nfracture", (1992, 13), fontsize=8, color=pv.INK2,
            xytext=(1997, 14), arrowprops=dict(arrowstyle="-", color=pv.MUTED, lw=0.8))
ax.set_xlim(1945, 2047)
ax.set_title("F3: new states nucleate on fracture surfaces", loc="left", fontsize=9.5, color=pv.INK)
ax.set_xlabel("year"); ax.set_ylabel("UN members admitted")

# (c) V armaz retrodiction
ax = axes[0, 2]
ax.scatter(on_az, iv_az, s=26, color=pv.BLUE, zorder=3, edgecolor=pv.SURFACE, linewidth=0.8)
tt = np.linspace(2007, min(tf_az - 0.05, 2026), 300)
ax.plot(tt, np.exp(res_az.intercept) * (tf_az - tt) ** res_az.slope, color=pv.AQUA, lw=2.0, ls="--")
ax.axvline(az[-1], color=pv.RED, lw=1.6)
ax.axvline(tf_az, color=pv.AQUA, lw=1.6, ls=":")
ax.text(az[-1] - 0.35, 6.3, f"actual fracture {az[-1]:.1f}", rotation=90, fontsize=8,
        color=pv.RED, fontweight="bold", va="top")
ax.text(tf_az + 0.15, 6.3, f"retrodicted t_f = {tf_az:.1f}", rotation=90, fontsize=8,
        color=pv.AQUA, fontweight="bold", va="top")
pv.direct_label(ax, on_az[0], iv_az[0], "observed intervals", pv.BLUE, dx=6, dy=6)
ax.set_xlim(2006, 2029.5); ax.set_ylim(0, 7.4)
ax.set_title("V: out-of-sample — Karabakh fracture,\nfit on 2008–2022 only", loc="left",
             fontsize=9.5, color=pv.INK)
ax.set_xlabel("crisis onset year"); ax.set_ylabel("interval to next crisis (yr)")

# (d) F4 kosovo
ax = axes[1, 0]
ax.scatter(on_ko, iv_ko, s=26, color=pv.BLUE, zorder=3, edgecolor=pv.SURFACE, linewidth=0.8)
tt = np.linspace(2004, min(tf_ko - 0.05, 2032), 300)
ax.plot(tt, np.exp(res_ko.intercept) * (tf_ko - tt) ** res_ko.slope, color=pv.AQUA, lw=2.0, ls="--")
ax.axvspan(band_ko[0], min(band_ko[1], 2033), color=pv.RED, alpha=0.10)
ax.axvline(tf_ko, color=pv.RED, lw=1.6, ls="--")
ax.text(tf_ko - 0.4, 7.1, f"t_f ≈ {tf_ko:.1f}", rotation=90, fontsize=8.5, color=pv.RED,
        fontweight="bold", va="top", ha="right")
pv.direct_label(ax, on_ko[2], iv_ko[2], "observed intervals", pv.BLUE, dx=6, dy=4)
pv.direct_label(ax, 2016, float(np.exp(res_ko.intercept) * (tf_ko - 2016) ** res_ko.slope),
                "compression fit", pv.AQUA, dx=6, dy=-10)
ax.set_xlim(2003, 2033); ax.set_ylim(0, 7.8)
ax.set_title("F4: Serbia–Kosovo interval compression", loc="left", fontsize=9.5, color=pv.INK)
ax.set_xlabel("crisis onset year"); ax.set_ylabel("interval to next crisis (yr)")

# (e) F5 plateau
ax = axes[1, 1]
ax.plot(cw.year, cw[NS_COL], color=pv.BLUE, lw=1.8)
ax.plot(years_far, ns_far, color=pv.AQUA, lw=2.0, ls="--")
ax.fill_between(years_far, ns_far * np.exp(-1.96 * sig_ns), ns_far * np.exp(1.96 * sig_ns),
                color=pv.AQUA, alpha=0.10, lw=0)
tt = np.arange(2010, 2047)
ax.plot(tt, lin.intercept + lin.slope * tt, color=pv.YELLOW, lw=1.8, ls=":")
ax.axvline(plateau_theta_year, color=pv.MUTED, lw=1.2, ls=":")
ax.text(plateau_theta_year + 0.4, 12, f"Θ hits 99% of ceiling ≈ {plateau_theta_year:.0f}",
        rotation=90, fontsize=8, color=pv.MUTED, va="bottom")
pv.direct_label(ax, 2014, 62, "observed non-state conflicts", pv.BLUE, dx=-6, dy=10)
pv.direct_label(ax, 2041, float(ns_far[years_far == 2041][0]), "frozen law:\nplateau", pv.AQUA,
                dx=6, dy=-16)
pv.direct_label(ax, 2028, float(lin.intercept + lin.slope * 2028), "linear null:\nkeeps climbing",
                pv.YELLOW, dx=-4, dy=26, ha="right")
ax.set_xlim(1989, 2047)
ax.set_title("F5: the thermal plateau of the defect budget", loc="left", fontsize=9.5, color=pv.INK)
ax.set_xlabel("year"); ax.set_ylabel("ongoing non-state conflicts")

# (f) F7 war-duration Weibull
ax = axes[1, 2]
ds = np.sort(dur)
Fm = (np.arange(1, len(ds) + 1) - 0.3) / (len(ds) + 0.4)
ax.scatter(np.log(ds), np.log(-np.log(1 - Fm)), s=24, color=pv.BLUE, zorder=3,
           edgecolor=pv.SURFACE, linewidth=0.8)
xx = np.linspace(np.log(ds.min()) - 0.3, np.log(ds.max()) + 0.5, 100)
ax.plot(xx, bw * (xx - np.log(ew)), color=pv.AQUA, lw=2.2)
ax.text(0.03, 0.97,
        f"25 interstate wars since 1945\nWeibull β_w = {bw:.2f} < 1:\nwars that do not end fast\nlast long",
        transform=ax.transAxes, fontsize=8.5, va="top", color=pv.INK2)
ax.text(0.97, 0.05,
        f"Ukraine at age {ukr_age:.1f} yr →\n"
        f"median cessation ≈ {NOW + m50:.0f}\n90% by ≈ {NOW + m90:.0f}\n"
        f"treaty at termination: {treaty_at_term}/{len(dur)} wars",
        transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right", color=pv.INK2)
ax.text(-4.6, bw * (-4.6 - np.log(ew)) + 0.35, "Weibull fit", color=pv.AQUA,
        fontsize=9, fontweight="bold")
ax.set_title("F7: how interstate wars end (frozen, not settled)", loc="left", fontsize=9.5, color=pv.INK)
ax.set_xlabel("ln war duration (yr)"); ax.set_ylabel("ln(−ln(1−F))")

fig.tight_layout(rect=[0, 0, 1, 0.935])
fig.savefig(os.path.join(FIGS, "G_timing_2046.png"))

with open(os.path.join(DATA, "fit_F_horizon.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
