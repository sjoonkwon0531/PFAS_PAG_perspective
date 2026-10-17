import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import *
R = os.path.join(HERE, "..", "sim", "results")
dt = pd.read_csv(os.path.join(R, "convergence_dt.csv"))
dx = pd.read_csv(os.path.join(R, "convergence_dx.csv"))
tests = pd.read_csv(os.path.join(R, "solver_tests.csv"))
fig, axs = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw=dict(wspace=0.38))
ax = axs[0]
ax.plot(dt.dt, dt.LWR, color=C1, marker="o", mec="white", mew=1.2, label="LWR")
ax.plot(dt.dt, dt.LER, color=C2, marker="s", mec="white", mew=1.2, label="LER")
ax.axvline(0.25, color=MUTED, lw=1.2, ls=(0, (4, 3)))
ax.text(0.27, 4.15, "production\n$\\Delta t$ = 0.25 s", fontsize=12, color=INK2, va="bottom")
ax.set_xscale("log"); ax.set_xticks([0.0625, 0.125, 0.25, 0.5, 1, 2]); ax.set_xticklabels(["0.06", "0.13", "0.25", "0.5", "1", "2"])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("Time step, $\\Delta t$ (s)"); ax.set_ylabel("Roughness (nm), same random field")
ax.set_ylim(3.0, 5.6); ax.legend(loc="lower left", fontsize=12.5)
panel(ax, "a", x=-0.2)
ax = axs[1]
g = dx.groupby("dx")
for i, q in enumerate(["CD", "LWR", "LER"]):
    for j, (d, c) in enumerate(((1.0, C1), (0.5, C2))):
        m, se = g[q].mean()[d], g[q].std()[d] / np.sqrt(g[q].count()[d])
        ax.bar(i + (j - 0.5) * 0.36, m, width=0.34, color=c, yerr=se, capsize=4, error_kw=dict(ecolor=INK, lw=1.2),
               label=f"$\\Delta x$ = {d:g} nm" if i == 0 else None)
ax.set_xticks(range(3)); ax.set_xticklabels(["CD", "LWR", "LER"]); ax.tick_params(axis="x", length=0)
ax.set_ylabel("nm (mean ± s.e., six fields)"); ax.legend(loc="upper right", fontsize=12.5)
panel(ax, "b", x=-0.2)
ax = axs[2]
t6 = tests[tests.test.str.startswith("T6")]
vals = [float(t6.iloc[0].created_Q_frac), max(float(t6.iloc[1].created_Q_frac), 1e-17)]
ax.bar([0, 1], vals, color=[REF, C1], width=0.55)
for x, v in zip([0, 1], vals):
    ax.text(x, v * 2.2, f"{v:.1e}" if v < 1e-3 else f"{100 * v:.1f}%", ha="center", fontsize=13)
ax.set_yscale("log"); ax.set_ylim(1e-17, 3)
ax.set_xticks([0, 1]); ax.set_xticklabels(["v1: continuous\nkernel + clipping", "v2: lattice\npropagator"]); ax.tick_params(axis="x", length=0)
ax.set_ylabel("Quencher created / initial quencher")
panel(ax, "c", x=-0.2)
save(fig, "FigS2_convergence")
print("ok")
