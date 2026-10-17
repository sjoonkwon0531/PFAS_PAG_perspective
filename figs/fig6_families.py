"""Fig. 6 (v4): formulation families at 12 nm hp - re-evaluated fronts, premiums with paired bootstrap,
fluorine-dose trade-off and emulator sensitivity."""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import *
R = os.path.join(HERE, "..", "sim", "results")
rd = lambda f: pd.read_csv(os.path.join(R, f))
RE = rd("round3_reeval_all.csv")
P = rd("round2_premium_r3.csv")
W = rd("round2_emu_sweep.csv")
res = rd("resolution_scenarios_v2.csv")
mor = res[(res.hp == 12) & res.label.str.startswith("MOR")].iloc[0]
base = RE[RE.origin.str.startswith("v3:baseline")].iloc[0]
KMIN_040 = 0.369
STY = {"free": dict(c=C1, ls="-", mk="s", lab="fluorine-free"),
       "B": dict(c=C2, ls="-", mk="o", lab="B: blends of existing anion types"),
       "H": dict(c=C4, ls=(0, (6, 2)), mk="D", lab="H: B + hybrid I/F anion"),
       "U": dict(c=MUTED, ls=(0, (2, 2)), mk="^", lab="U: unconstrained benchmark")}


def front(df):
    d = df[df.feasible].assign(_d=lambda x: x.dose.round(1)).sort_values(["_d", "LWR"]); best, keep = np.inf, []
    for i, r in d.iterrows():
        if r.LWR < best - 1e-12:
            keep.append(i); best = r.LWR
    return d.loc[keep]


FR = {f: front(RE[RE[f"in_{f}"]]) for f in STY}
fig, axs2 = plt.subplots(2, 2, figsize=(13.5, 10.6), gridspec_kw=dict(wspace=0.32, hspace=0.42))
axs = axs2.ravel()

# a: fronts
ax = axs[0]
F = RE[RE.feasible & ~RE.origin.str.startswith("v3:baseline")]
ax.scatter(F.dose, F.LWR, s=16, color="#c9c8c4", zorder=1, lw=0)
for f in ("U", "H", "B", "free"):
    d, st = FR[f], STY[f]
    ax.step(np.r_[d.dose.values, 80], np.r_[d.LWR.values, d.LWR.values[-1]], where="post", color=st["c"], ls=st["ls"],
            lw=2.4, zorder=3, label=st["lab"])
    ax.errorbar(d.dose, d.LWR, xerr=d.dose_se, yerr=d.LWR_se, fmt=st["mk"], ms=7.5, color=st["c"], mec=INK, mew=1.0,
                ecolor=INK, elinewidth=0.9, capsize=1.5, zorder=4, ls="none")
ax.errorbar(base.dose, base.LWR, xerr=base.dose_se, yerr=base.LWR_se, fmt="*", ms=18, color=INK, mec="white", mew=1.2,
            ecolor=INK, capsize=2, zorder=6)
ax.annotate("fluorinated baseline", (base.dose, base.LWR), xytext=(base.dose + 8, base.LWR + 0.4), fontsize=12.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.plot(mor.dose, mor.LWR, marker="^", ms=11, color=C3, mec="white", mew=1.5, zorder=6, ls="none")
ax.annotate("tin-oxo MOR (illustrative)", (mor.dose, mor.LWR), xytext=(mor.dose + 2, mor.LWR), fontsize=12, va="center")
ax.set_xlabel("Dose-to-size (mJ cm$^{-2}$)"); ax.set_ylabel("LWR at size (nm)")
ax.set_xlim(8, 65); ax.set_ylim(min(2.9, mor.LWR - 0.4), 8.6)
ax.legend(loc="upper right", fontsize=11, handlelength=2.4, borderaxespad=0.2)
panel(ax, "a", x=-0.17)

# b: premiums by family, paired bootstrap 90% intervals
ax = axs[1]
for j, f in enumerate(("B", "H", "U")):
    g = P[(P.family == f) & P.premium.notna()]
    x = g.LWR_target.values + (j - 1) * 0.045
    st = STY[f]
    ax.errorbar(x, g.premium, yerr=[g.premium - g.paired_p05, g.paired_p95 - g.premium], fmt=st["mk"], ms=9, color=st["c"],
                mec=INK, mew=1.0, ecolor=st["c"], elinewidth=1.8, capsize=4, label=st["lab"].split(":")[0])
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(sorted(P.LWR_target.unique())); ax.set_xticklabels([f"{t:g}" for t in sorted(P.LWR_target.unique())])
ax.set_xlabel("LWR target (nm)"); ax.set_ylabel("Exposure-dose premium, fluorine-free (%)")
ax.legend(loc="upper left", fontsize=12.5, title="compared with", title_fontsize=12.5)
panel(ax, "b", x=-0.2)

# c: lowest dose at LWR <= 5.0 nm versus a cap on film fluorine
ax = axs[2]
caps = np.linspace(0, 135, 271)
for f in ("U", "H", "B"):
    d = RE[RE.feasible & RE[f"in_{f}"] & (RE.LWR <= 5.0)]
    y = [d[d.F_ug <= c + 1e-9].dose.min() if (d.F_ug <= c + 1e-9).any() else np.nan for c in caps]
    st = STY[f]
    ax.step(caps, y, where="post", color=st["c"], ls=st["ls"], lw=2.4, label=st["lab"].split(":")[0])
d0 = RE[RE.feasible & RE.in_free & (RE.LWR <= 5.0)].dose.min()
ax.plot([0], [d0], marker="s", ms=10, color=C1, mec=INK, mew=1.0, ls="none")
ax.annotate("fluorine-free", (0, d0), xytext=(14, d0 - 0.15), fontsize=12.5, color=C1, va="center")
ax.axvline(base.F_ug, color=INK2, lw=1.2, ls=(0, (4, 3)))
ax.text(base.F_ug + 2, 0.97, "baseline\nfilm F", transform=ax.get_xaxis_transform(), fontsize=12, color=INK2, va="top")
ax.set_xlabel("Cap on film fluorine (µg per wafer-layer)")
ax.set_ylabel("Lowest dose at LWR $\\leq$ 5.0 nm (mJ cm$^{-2}$)")
ax.set_xlim(-4, 135)
ax.legend(loc="lower left", fontsize=12.5)
panel(ax, "c", x=-0.2)

# d: emulator sensitivity, families B and U
ax = axs[3]
for f, ls in (("B", "-"), ("U", (0, (2, 2)))):
    for t, c in ((5.0, BLUES[4]), (5.5, BLUES[6])):
        g = W[np.isclose(W.LWR_target, t)].sort_values("kappa_max_free")
        ax.plot(g.kappa_max_free, g[f"premium_{f}"], color=c, ls=ls, marker="o" if f == "B" else None, ms=6, mec="white",
                mew=1, lw=2.2, label=f"{f}, LWR $\\leq$ {t:g} nm")
ax.axvspan(0.3, KMIN_040, color="#ecebe7", hatch="///", ec="#b9b8b3", lw=0)
ax.axvline(KMIN_040, color=C2, lw=2.0)
ax.text(KMIN_040 + 0.02, 0.14, "$\\kappa_{min}$ (0.40 mmol g$^{-1}$)", transform=ax.get_xaxis_transform(), fontsize=12, color=C2,
        va="top")
ax.axvline(0.9, color=INK, lw=1.4, ls=(0, (5, 3)))
ax.text(0.92, 0.80, "reference\nassumptions", transform=ax.get_xaxis_transform(), fontsize=12, color=INK, va="top")
ax.set_xlim(0.3, 2.05); ax.set_ylim(-5, 280)
ax.set_xlabel(r"Largest fluorine-free activity, $\kappa_{aryl}s_{PG,max}$")
ax.set_ylabel("Emulator premium, fluorine-free (%)")
ax.legend(loc="upper right", fontsize=12, handlelength=2.4)
panel(ax, "d", x=-0.2)
save(fig, "Fig6_MOBO")
print({f: len(v) for f, v in FR.items()})
