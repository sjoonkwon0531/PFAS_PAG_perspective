"""Fig. S4: design variables of the re-evaluated front designs of each formulation family (12 nm hp)."""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import *
R = os.path.join(HERE, "..", "sim", "results")
F = pd.read_csv(os.path.join(R, "round2_fronts_r3.csv"))
vars_ = [("F_eq", 0, 1, "$f_F$"), ("n_I", 0, 3, "$n_I$"), ("s_PG", 1, 3, "$s_{PG}$"), ("L_D", 3, 10, "$L_D$"), ("c_PAG", 0.2, 0.4, "$c_{PAG}$")]
col = {"free": C1, "B": C2, "H": C4, "U": MUTED}
fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True, gridspec_kw=dict(wspace=0.12))
xs = np.arange(len(vars_))
for ax, f, letter in zip(axs, ("free", "B", "H", "U"), "abcd"):
    g = F[F.front == f]
    for _, r in g.iterrows():
        ax.plot(xs, [(r[k] - lo) / (hi - lo) for k, lo, hi, _ in vars_], color=col[f], lw=1.8, alpha=0.85, marker="o", ms=6,
                mec="white", mew=0.8)
    for x in xs:
        ax.axvline(x, color=GRID, lw=1.1, zorder=0)
    ax.set_xticks(xs); ax.set_xticklabels([v[3] for v in vars_], fontsize=15)
    ax.tick_params(axis="x", length=0); ax.spines["bottom"].set_visible(False)
    ax.set_ylim(-0.06, 1.06)
    ax.text(0.86 if f != "free" else 0.39, 0.015 if f != "free" else 0.4, {"free": "fluorine-free", "B": "family B", "H": "family H", "U": "family U"}[f], transform=ax.transAxes,
            ha="center", fontsize=13, color=col[f] if f != "U" else INK2, fontweight="bold")
    panel(ax, letter, x=-0.08 if letter != "a" else -0.3)
axs[0].set_yticks([0, 0.5, 1]); axs[0].set_yticklabels(["lower\nbound", "", "upper\nbound"], fontsize=12.5)
save(fig, "FigS4_design_variables")
print("ok")
