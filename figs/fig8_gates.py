import os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import *

gates = [("1", "Chemical\nscreening", "effective activity $\\kappa$,\nacid yield $\\phi$, absorption $\\alpha$,\nEq. 17 criterion", "Which candidates\nto synthesize", "computed here"),
         ("2", "Film\nkinetics", "deprotection vs bake\ntime at saturating dose;\nacid diffusion", "Is $\\kappa = \\kappa_{acid}s_{PG}$\nseparable?", "measurements\nproposed here"),
         ("3", "Patterning\nwindow", "dose, LWR, EL and a\ndefect metric at the\ntarget pitch", "Matched-performance\ncomparison", ""),
         ("4", "Etch and\ncontamination", "etch transfer,\noutgassing, metal and\nionic purity", "Process compatibility", ""),
         ("5", "Module\nqualification", "defectivity, stability,\nsupply specification", "Module release", ""),
         ("6", "Lot-level\nvalidation", "yield, reliability,\ninventory and emission\nper accepted layer", "Product qualification\nand LCA", "")]
fig, ax = plt.subplots(figsize=(17, 6.4))
ax.set_xlim(0, 98.5); ax.set_ylim(0, 42); ax.axis("off")
w, gap = 14.2, 2.0
for i, (n, t, obs, dec, note) in enumerate(gates):
    x = 1 + i * (w + gap)
    done = i == 0; part = i == 1
    fc = BLUES[2] if done else (BLUES[0] if part else "#f4f3f0")
    ax.add_patch(FancyBboxPatch((x, 7), w, 31, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc,
                                ec=C1 if (done or part) else GRID, lw=1.6))
    ax.text(x + 1.0, 34.0, n, fontsize=22, fontweight="bold", color=C1 if (done or part) else MUTED, va="center")
    ax.text(x + 3.6, 34.0, t, fontsize=14.5, fontweight="bold", color=INK, va="center", linespacing=1.0)
    ax.text(x + 1.0, 28.3, "Observables", fontsize=12, color=INK2, va="center", style="italic")
    ax.text(x + 1.0, 26.3, obs, fontsize=12.5, color=INK, va="top", linespacing=1.15)
    ax.text(x + 1.0, 15.2, "Decision", fontsize=12, color=INK2, va="center", style="italic")
    ax.text(x + 1.0, 13.2, dec, fontsize=12.5, color=INK, va="top", linespacing=1.15)
    if note:
        ax.text(x + w / 2, 3.2, note, fontsize=12.5, color=C1, ha="center", va="center", fontweight="bold", linespacing=1.1)
    if i < len(gates) - 1:
        ax.annotate("", xy=(x + w + gap - 0.1, 22), xytext=(x + w + 0.5, 22),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6, mutation_scale=16))
save(fig, "Fig8_gates")
print("ok")
