import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import *
R = os.path.join(HERE, "..", "sim", "results")
T = pd.read_csv(os.path.join(R, "emulator_v2_table.csv"))
F = pd.read_csv(os.path.join(R, "emulator_v2_fronts.csv"))
fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.0), gridspec_kw=dict(wspace=0.34))
shades = {5.0: BLUES[3], 6.5: BLUES[5], 8.0: BLUES[6]}
for ax, q, lab, letter in ((axs[0], "dose", "Dose-to-size (mJ cm$^{-2}$)", "a"), (axs[1], "LWR", "LWR at size (nm)", "b")):
    for L, c in shades.items():
        d = T[(T.L_D == L) & (T.alpha_um == 5.0) & (T.status == "ok")].sort_values("kappa")
        ax.errorbar(d.kappa, d[q], yerr=d[f"{q}_se"], color=c, marker="o", ms=7, mec="white", mew=1.2, capsize=3, label=f"{L:g} nm")
    ax.set_xscale("log"); ax.set_xticks([0.4, 0.6, 1, 2, 5]); ax.set_xticklabels(["0.4", "0.6", "1", "2", "5"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"Relative effective activity, $\kappa$"); ax.set_ylabel(lab)
    panel(ax, letter, x=-0.2)
axs[0].set_yscale("log"); axs[0].set_yticks([10, 20, 50, 100]); axs[0].set_yticklabels(["10", "20", "50", "100"])
axs[0].yaxis.set_minor_formatter(plt.NullFormatter())
axs[0].legend(title="$L_D$", fontsize=12, title_fontsize=13)
ax = axs[2]
cols = {"ref": INK, "ka02": C1, "ka05": C2, "spg2": C3, "spg5": C4}
names = {"ref": "reference (0.9)", "ka02": "$\\kappa_{aryl}$ = 0.2 (0.6)", "ka05": "$\\kappa_{aryl}$ = 0.5 (1.5)",
         "spg2": "$s_{PG}$ $\\leq$ 2 (0.6)", "spg5": "$s_{PG}$ $\\leq$ 5 (1.5)"}
A = F[(F.case == "ref") & (F.campaign == "fluorine-allowed")].sort_values("dose")
ax.step(np.r_[A.dose, 90], np.r_[A.LWR, A.LWR.values[-1]], where="post", color=MUTED, lw=2.4, ls=(0, (4, 2)),
        label="fluorine allowed (reference)")
for k, c in cols.items():
    B = F[(F.case == k) & (F.campaign == "fluorine-free")].sort_values("dose")
    if len(B) == 0:
        continue
    ls = (0, (1.5, 1.5)) if k in ("spg2", "spg5") else "-"
    ax.step(np.r_[B.dose, 90], np.r_[B.LWR, B.LWR.values[-1]], where="post", color=c, lw=2.0, ls=ls, label=names[k])
ax.set_xlim(5, 90)
ax.set_xlabel("Dose-to-size (mJ cm$^{-2}$)"); ax.set_ylabel("LWR at size (nm)")
ax.legend(fontsize=11.5, loc="upper right", handlelength=1.8)
panel(ax, "c", x=-0.2)
save(fig, "FigS1_emulator")
print("ok")
