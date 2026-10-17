"""Table-of-contents graphic (8 cm x 4 cm)."""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import *
P = pd.read_csv(os.path.join(HERE, "..", "sim", "results", "round2_premium.csv"))
cm = 1 / 2.54
fig = plt.figure(figsize=(8 * cm, 4 * cm))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 8); ax.set_ylim(0, 4); ax.axis("off")
# left: PFAS PAG anion and its four functions
ax.text(0.75, 3.45, "C$_4$F$_9$SO$_3^-$", fontsize=8.5, color=C2, ha="center", va="center", fontweight="bold")
ax.text(0.75, 3.05, "PFAS PAG anion", fontsize=6.5, color=INK2, ha="center", va="center")
funcs = ["absorption", "acid activity", "acid diffusion", "compatibility"]
for i, f in enumerate(funcs):
    y = 2.35 - i * 0.58
    ax.add_patch(FancyBboxPatch((0.08, y - 0.2), 1.35, 0.4, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#eef4fc", ec=C1, lw=0.8))
    ax.text(0.755, y, f, fontsize=6.3, ha="center", va="center", color=INK)
ax.add_patch(FancyArrowPatch((1.55, 1.5), (2.12, 1.5), arrowstyle="-|>", mutation_scale=8, color=INK2, lw=1))
# middle: screening framework as gates
steps = ["photon\nbudget", "activity\ncriterion", "stochastic\nmodel", "family-\nconstrained\nsearch", "fluorine\nbalance"]
for i, s in enumerate(steps):
    x = 2.2 + i * 0.66
    ax.add_patch(FancyBboxPatch((x, 0.95), 0.6, 1.1, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#fdf0e9" if i == 3 else "#f4f3f0",
                                ec=C2 if i == 3 else MUTED, lw=0.8))
    ax.text(x + 0.3, 1.5, s, fontsize=4.6, ha="center", va="center", color=INK, linespacing=0.95)
ax.text(3.73, 2.35, "falsifiable screening framework", fontsize=6.5, ha="center", color=INK, fontweight="bold")
ax.text(3.73, 0.62, "each step tested by a measurement", fontsize=5.6, ha="center", color=INK2)
# right: premium depends on the comparison set (LWR <= 5.0 nm)
ia = fig.add_axes([0.775, 0.25, 0.205, 0.5])
g = P[np.isclose(P.LWR_target, 5.0)].set_index("family")
fam = ["B", "H", "U"]; col = [C2, C4, MUTED]
ia.bar(range(3), [g.loc[f, "premium"] for f in fam], color=col, width=0.65,
       yerr=[[g.loc[f, "premium"] - g.loc[f, "paired_p05"] for f in fam], [g.loc[f, "paired_p95"] - g.loc[f, "premium"] for f in fam]],
       error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK))
ia.set_xticks(range(3)); ia.set_xticklabels(["blend", "hybrid", "uncon-\nstrained"], fontsize=5.2)
ia.tick_params(axis="y", labelsize=5.5, length=2, pad=1); ia.tick_params(axis="x", length=0, pad=1)
ia.set_ylim(0, 45)
for sp in ("left", "bottom"):
    ia.spines[sp].set_linewidth(0.6)
fig.text(0.875, 0.83, "model dose premium (%) of\nfluorine-free PAGs vs", fontsize=5.8, ha="center", color=INK, linespacing=1.0)
fig.text(0.875, 0.035, "12 nm hp, LWR $\\leq$ 5.0 nm", fontsize=5.0, ha="center", color=INK2)
for ext in ("png", "pdf", "svg", "tiff"):
    fig.savefig(os.path.join(HERE, f"TOC_graphic.{ext}"), dpi=600 if ext in ("png", "tiff") else None,
                **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}))
print("ok")
