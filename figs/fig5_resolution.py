import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import *
R = os.path.join(HERE, "..", "sim", "results")
el = pd.read_csv(os.path.join(R, "el_vs_hp_v2.csv"))
st = pd.read_csv(os.path.join(R, "resolution_scenarios_v2.csv"))
vs = pd.read_csv(os.path.join(R, "visibility_v2.csv")) if os.path.exists(os.path.join(R, "visibility_v2.csv")) else None

fig, axs2 = plt.subplots(2, 2, figsize=(13.5, 10.0), gridspec_kw=dict(wspace=0.3, hspace=0.38))
axs = axs2.ravel()
# a: exposure latitude vs half-pitch
ax = axs[0]
LDs = sorted(el.L_D.unique())
shades = [BLUES[2], BLUES[3], BLUES[4], BLUES[5], BLUES[6]]
for L, c in zip(LDs, shades):
    d = el[el.L_D == L].sort_values("hp")
    ax.plot(d.hp, d.EL, color=c, marker="o", ms=7, mec="white", mew=1.2, label=f"{L:g} nm")
ax.axhline(10, color=MUTED, lw=1.2, ls=(0, (4, 3)))
ax.text(15.2, 10.5, "EL = 10%", fontsize=12, color=INK2, va="bottom", ha="left")
ax.legend(title="$L_D$", loc="lower right", bbox_to_anchor=(1.03, -0.02), ncol=2, columnspacing=0.8, fontsize=12,
          title_fontsize=13, handlelength=1.4)
ax.set_xlabel("Half-pitch (nm)"); ax.set_ylabel("Mean-field exposure latitude (%)")
ax.set_xlim(7.5, 24.8); ax.set_ylim(0, 30); ax.set_xticks([8, 12, 16, 20, 24])
panel(ax, "a", x=-0.17)

style = {"S0 Fluorinated baseline": (REF, "o", "Fluorinated baseline"),
         "S2 Idealized bake-equivalent": (C1, "s", "Bake-equivalent (fluorine-free)"),
         "S5 Co-designed PFAS-free": (C2, "D", "Co-designed (fluorine-free)"),
         "MOR tin-oxo cage (illustrative)": (C3, "^", "Tin-oxo MOR (illustrative)")}
for ax, col, se, ylab, letter, yl in [(axs[1], "LWR_over_CD", "LWR_over_CD_se", "LWR / CD (valid rows)", "b", (0, 0.85)),
                                      (axs[2], "missing", "missing_se", "Missing-edge-pair segment fraction", "c", (-0.008, 0.19))]:
    for lab, (c, mk, name) in style.items():
        d = st[st.label == lab].sort_values("hp")
        ax.errorbar(d.hp, d[col], yerr=d[se], color=c, marker=mk, ms=9, mec="white", mew=1.5, capsize=4,
                    elinewidth=1.4, label=name)
    ax.set_xlabel("Half-pitch (nm)"); ax.set_ylabel(ylab)
    ax.set_xticks([10, 12, 16]); ax.set_xlim(9.3, 16.7); ax.set_ylim(*yl)
    ax.invert_xaxis()
    ax.axvline(14, color=GRID, lw=1.2)
    panel(ax, letter, x=-0.17)
axs[1].text(16.55, 0.805, "0.33 NA regime", fontsize=12, color=INK2, ha="left")
axs[1].text(13.7, 0.805, "0.55 NA regime (proxy)", fontsize=12, color=INK2, ha="left")
axs[2].legend(loc="upper left", fontsize=12.5, handlelength=1.6)

# d: image-visibility sensitivity at 12 nm hp
ax = axs[3]
if vs is not None:
    for lab, (c, mk, name) in style.items():
        d = vs[vs.label == lab].sort_values("visibility")
        if len(d) == 0:
            continue
        ax.errorbar(d.visibility, d.LWR / 12.0, yerr=d.LWR_se / 12.0, color=c, marker=mk, ms=9, mec="white", mew=1.5,
                    capsize=4, elinewidth=1.4, label=name)
    ax.set_xlabel("Aerial image visibility, $V$ (12 nm hp)"); ax.set_ylabel("LWR / CD (valid rows)")
    ax.set_xticks([0.70, 0.85, 1.00]); ax.set_xlim(0.66, 1.04)
panel(ax, "d", x=-0.17)
save(fig, "Fig5_resolution")
print("ok")
