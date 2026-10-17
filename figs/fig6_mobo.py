import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import *
R = os.path.join(HERE, "..", "sim", "results")
rd = lambda f: pd.read_csv(os.path.join(R, f))
A = pd.concat([rd(f"mobo_v2_allowed_s{s}.csv") for s in (7, 11)], ignore_index=True)
B = pd.concat([rd(f"mobo_v2_free_s{s}.csv") for s in (7, 11)], ignore_index=True)
RE = rd("mobo_v2_reeval.csv")
P = rd("mobo_v2_premium.csv")
W = rd("emulator_v2_kappa_max_sweep.csv") if os.path.exists(os.path.join(R, "emulator_v2_kappa_max_sweep.csv")) else pd.DataFrame()
res = rd("resolution_scenarios_v2.csv")
mor = res[(res.hp == 12) & res.label.str.startswith("MOR")].iloc[0]
base = RE[RE.chem == "baseline"].iloc[0]
Af, Bf = A[A.feasible], B[B.feasible]
oranges = LinearSegmentedColormap.from_list("or", ["#fbe3d8", "#f3a581", "#eb6834", "#a8401a"])
KMIN_040 = 0.369


def front(df):
    df = df[df.feasible].assign(_d=lambda x: x.dose.round(1)).sort_values(["_d", "LWR"]); best, keep = np.inf, []
    for i, r in df.iterrows():
        if r.LWR < best - 1e-12:
            keep.append(i); best = r.LWR
    return df.loc[keep]


# f_F = 0 lies inside the fluorine-allowed design space: the attainable fluorine-allowed front is the
# non-dominated set of both campaigns (union front); the fluorine-free front uses fluorine-free designs only.
REa, REb = front(RE[RE.chem.isin(["allowed", "free"])]), front(RE[RE.chem == "free"])
fig, axs2 = plt.subplots(2, 2, figsize=(13.5, 10.4), gridspec_kw=dict(wspace=0.32, hspace=0.4))
axs = axs2.ravel()

# a: objective space
ax = axs[0]
sc = ax.scatter(Af.dose, Af.LWR, c=Af.F_ug, cmap=oranges, norm=Normalize(0, 130), s=30, edgecolor="white", lw=0.6, zorder=3)
ax.scatter(Bf.dose, Bf.LWR, color=C1, marker="s", s=26, edgecolor="white", lw=0.6, zorder=3, alpha=0.55)
for df, c, mk in ((REa, C2, "o"), (REb, C1, "s")):
    ax.step(np.r_[df.dose.values, 80], np.r_[df.LWR.values, df.LWR.values[-1]], where="post", color=c, lw=2.2, zorder=4)
    for _, r in df.iterrows():
        m = "D" if r.get("selection", "") == "emulator front" else mk
        ax.errorbar(r.dose, r.LWR, xerr=r.dose_se, yerr=r.LWR_se, fmt=m, ms=9 if m != "D" else 8.5, color=c, mec=INK, mew=1.3,
                    ecolor=INK, elinewidth=1, capsize=2, zorder=5)

ax.errorbar(base.dose, base.LWR, xerr=base.dose_se, yerr=base.LWR_se, fmt="*", ms=19, color=INK, mec="white", mew=1.2,
            ecolor=INK, capsize=2, zorder=6)
ax.plot(mor.dose, mor.LWR, marker="^", ms=12, color=C3, mec="white", mew=1.5, zorder=6)
ax.annotate("fluorinated baseline", (base.dose, base.LWR), xytext=(base.dose + 10, base.LWR + 1.45), fontsize=12.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.annotate("tin-oxo MOR (illustrative)", (mor.dose, mor.LWR), xytext=(mor.dose + 2, mor.LWR), fontsize=12, va="center")
ax.set_xlabel("Dose-to-size (mJ cm$^{-2}$)"); ax.set_ylabel("LWR at size (nm)")
lo = np.nanmin(np.r_[Af.dose, Bf.dose]); ax.set_xlim(max(5, lo - 3), 80)
ax.set_ylim(min(2.9, mor.LWR - 0.4), 9.8)
cax = ax.inset_axes([0.5, 0.86, 0.46, 0.035])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_label("Film F (µg per wafer-layer)", fontsize=12); cb.ax.xaxis.set_label_position("top")
cb.ax.tick_params(labelsize=11.5); cb.outline.set_visible(False)
panel(ax, "a", x=-0.18)

# b: design variables of re-evaluated Pareto designs
ax = axs[1]
vars_ = [("f_F", 0, 1, "$f_F$"), ("n_I", 0, 3, "$n_I$"), ("s_PG", 1, 3, "$s_{PG}$"), ("L_D", 3, 10, "$L_D$"), ("c_PAG", 0.2, 0.4, "$c_{PAG}$")]
xs = np.arange(len(vars_))
for df, c in ((REa, C2), (REb, C1)):
    for _, r in df.iterrows():
        ax.plot(xs, [(r[k] - lo_) / (hi - lo_) for k, lo_, hi, _ in vars_], color=c, lw=2.0, alpha=0.9, marker="o", ms=7,
                mec="white", mew=1)
for x in xs:
    ax.axvline(x, color=GRID, lw=1.2, zorder=0)
ax.set_xticks(xs); ax.set_xticklabels([v[3] for v in vars_], fontsize=16)
ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["lower\nbound", "", "upper\nbound"], fontsize=12.5)
ax.set_ylim(-0.06, 1.06); ax.tick_params(axis="x", length=0); ax.spines["bottom"].set_visible(False)
ax.text(0.98, 0.02, "fluorine allowed", color=C2, transform=ax.transAxes, ha="right", fontsize=12.5, fontweight="bold")
ax.text(0.98, 0.09, "fluorine-free", color=C1, transform=ax.transAxes, ha="right", fontsize=12.5, fontweight="bold")
panel(ax, "b", x=-0.22)

# c: exposure-dose premium with bootstrap interval
ax = axs[2]
Pm = P[P.premium.notna()]
ax.errorbar(Pm.LWR_target, Pm.premium, yerr=[Pm.premium - Pm.p05, Pm.p95 - Pm.premium], fmt="o", ms=9, color=C1, mec="white",
            mew=1.4, ecolor=C1, elinewidth=1.6, capsize=5)
ax.axhline(0, color=MUTED, lw=1.0)
ax.set_xlabel("LWR target (nm)"); ax.set_ylabel("Exposure-dose premium, fluorine-free (%)")
ax.set_xticks(Pm.LWR_target.values); ax.set_xticklabels([f"{t:g}" for t in Pm.LWR_target.values])
panel(ax, "c", x=-0.2)

# d: emulator premium vs largest fluorine-free effective activity
ax = axs[3]
shades = {4.5: BLUES[3], 4.75: BLUES[4], 5.0: BLUES[6]}
for t, c in shades.items():
    col = f"premium_{t}"
    if col in W:
        ax.plot(W.kappa_max_free, W[col], color=c, marker="o", ms=7, mec="white", mew=1.2, label=f"LWR $\\leq$ {t:g} nm")
for (t, c), dx in zip(shades.items(), (-0.045, 0.0, 0.045)):
    r = P[np.isclose(P.LWR_target, t)]
    if len(r) and np.isfinite(r.premium.iloc[0]):
        r = r.iloc[0]
        ax.errorbar(0.9 + dx, r.premium, yerr=[[r.premium - r.p05], [r.p95 - r.premium]], fmt="D", ms=8, mfc="white", mec=c,
                    mew=2, ecolor=c, elinewidth=1.5, capsize=3, zorder=6, label="_nolegend_")
ax.plot([], [], "D", ms=8, mfc="white", mec=INK2, mew=2, label="direct re-evaluation")
ax.axvspan(0.3, KMIN_040, color="#ecebe7", hatch="///", ec="#b9b8b3", lw=0)
ax.axvline(KMIN_040, color=C2, lw=2.0)
ax.text(KMIN_040 + 0.02, 0.36, "$\\kappa_{min}$ (0.40 mmol g$^{-1}$)", transform=ax.get_xaxis_transform(), fontsize=12, color=C2,
        va="top")
ax.axvline(0.9, color=INK, lw=1.4, ls=(0, (5, 3)))
ax.text(0.92, 0.80, "reference\nassumptions", transform=ax.get_xaxis_transform(), fontsize=12, color=INK, va="top")
ax.set_xlim(0.3, 2.05); ax.set_ylim(bottom=-5)
ax.set_xlabel(r"Largest fluorine-free activity, $\kappa_{aryl}s_{PG,max}$")
ax.set_ylabel("Exposure-dose premium, fluorine-free (%)")
ax.legend(loc="upper right", fontsize=12.5, handlelength=1.6)
panel(ax, "d", x=-0.2)
save(fig, "Fig6_MOBO")
print(P.round(1).to_string())
