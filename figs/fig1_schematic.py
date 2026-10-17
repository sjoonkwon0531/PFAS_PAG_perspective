import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import *

fig = plt.figure(figsize=(16, 9.2))
axA = fig.add_axes([0.02, 0.36, 0.56, 0.6]); axA.set_xlim(0, 100); axA.set_ylim(0, 62); axA.axis("off")
axB = fig.add_axes([0.61, 0.36, 0.37, 0.6]); axB.set_xlim(0, 100); axB.set_ylim(0, 62); axB.axis("off")
axC = fig.add_axes([0.05, 0.02, 0.9, 0.29]); axC.axis("off")

# ---------------- a: where fluorine sits in an EUV patterning stack ----------------
ax = axA
layers = [(4, 8, "Silicon wafer", "#d9d8d3"),
          (12, 7, "Spin-on carbon / hard mask", "#c4c3bd"),
          (19, 4, "Underlayer (adhesion)", "#b9b8b3"),
          (23, 16, "", "#eef4fc")]
for y, h, lab, c in layers:
    ax.add_patch(Rectangle((6, y), 52, h, facecolor=c, edgecolor="white", lw=2))
    if lab:
        ax.text(32, y + h / 2, lab, ha="center", va="center", fontsize=13, color=INK)
ax.text(32, 37.2, "Chemically amplified resist film (20 to 30 nm)", ha="center", va="center", fontsize=13, color=INK)
rng = np.random.default_rng(3)
# polymer (gray dots), PAG (orange = fluorinated anion), quencher (blue)
for _ in range(110):
    ax.add_patch(Circle((rng.uniform(8, 56), rng.uniform(24.5, 34.5)), 0.45, color="#b9b8b3", lw=0))
for _ in range(16):
    ax.add_patch(Circle((rng.uniform(8, 56), rng.uniform(24.5, 34.5)), 0.85, color=C2, lw=0))
for _ in range(5):
    ax.add_patch(Circle((rng.uniform(8, 56), rng.uniform(24.5, 34.5)), 0.75, color=C1, lw=0))
# surfactant at the surface
for x in np.arange(8, 57, 3.0):
    ax.plot([x, x], [39, 40.6], color=C2, lw=1.6); ax.add_patch(Circle((x, 40.9), 0.45, color=C2, lw=0))
# EUV photons
for x in (16, 32, 48):
    ax.annotate("", xy=(x, 41.8), xytext=(x - 4, 55), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8,
                                                                       mutation_scale=16))
ax.text(10, 57.5, "EUV, 13.5 nm (91.8 eV)", fontsize=13.5, color=INK)
# legend chips
ax.add_patch(Circle((8, 1.5), 0.9, color=C2, lw=0)); ax.text(9.6, 1.5, "fluorinated PAG anion / surfactant", va="center", fontsize=12.5)
ax.add_patch(Circle((38, 1.5), 0.9, color=C1, lw=0)); ax.text(39.6, 1.5, "quencher", va="center", fontsize=12.5)
ax.add_patch(Circle((50, 1.5), 0.6, color="#b9b8b3", lw=0)); ax.text(51.4, 1.5, "polymer", va="center", fontsize=12.5)

roles = [(54, "Catalytic activity", "perfluoroalkyl sulfonic acids drive deprotection"),
         (44, "Diffusion control", "anion size sets acid blur"),
         (34, "Solubility and inertness", "no side reactions in the film"),
         (24, "Surface activity", "coating uniformity, rinse, pattern collapse"),
         (14, "EUV absorption", "modest in the model: $\\alpha$ about 8% lower without F")]
for y, t, sub in roles:
    ax.add_patch(FancyBboxPatch((63, y - 4), 35.5, 8.2, boxstyle="round,pad=0.25,rounding_size=1.2",
                                facecolor="white", edgecolor=GRID, lw=1.2))
    ax.text(65, y + 1.5, t, fontsize=14, fontweight="bold", color=INK, va="center")
    ax.text(65, y - 1.8, sub, fontsize=12, color=INK2, va="center")
for y in (54, 44, 34, 14):
    ax.add_patch(FancyArrowPatch((57.5, 31), (62.5, y), arrowstyle="-", color=C2, lw=1.2,
                                 connectionstyle="arc3,rad=0.0", alpha=0.7))
ax.add_patch(FancyArrowPatch((57.5, 40.5), (62.5, 24), arrowstyle="-", color=C2, lw=1.2, alpha=0.7))
panel(ax, "a", x=0.0, y=0.98)

# ---------------- b: what this Perspective quantifies ----------------
ax = axB
items = [(53, "Photon budget", "Henke cross sections, Eqs. 1 to 3", "Fig. 2"),
         (43, "Stochastic patterning", "verified solver; activity criterion, Eqs. 4 to 17", "Figs. 3, 4"),
         (33, "Shrinking-pitch stress test", "fixed aerial image; NA labels are proxies", "Fig. 5"),
         (23, "Design-space exploration", "formulation families; fluorine as objective", "Fig. 6"),
         (13, "Screening inventory", "allocation to waste pathways, not emission", "Fig. 7")]
for y, t, sub, f in items:
    ax.add_patch(FancyBboxPatch((4, y - 4), 92, 8.2, boxstyle="round,pad=0.25,rounding_size=1.2",
                                facecolor="#eef4fc", edgecolor="none"))
    ax.text(7, y + 1.5, t, fontsize=14, fontweight="bold", color=INK, va="center")
    ax.text(7, y - 1.8, sub, fontsize=12, color=INK2, va="center")
    ax.text(94, y, f, fontsize=13, color=C1, va="center", ha="right", fontweight="bold")
for y0, y1 in [(49, 47.3), (39, 37.3), (29, 27.3), (19, 17.3)]:
    ax.annotate("", xy=(50, y1), xytext=(50, y0), arrowprops=dict(arrowstyle="-|>", color=C1, lw=1.5))
panel(ax, "b", x=0.0, y=0.98)

# ---------------- c: regulatory and industrial timeline (stage-coded) ----------------
ax = axC
ax.set_xlim(2020.6, 2028.2); ax.set_ylim(-3.9, 2.9)
ax.plot([2021, 2027.8], [0, 0], color=INK2, lw=2)
for yr in range(2021, 2028):
    ax.plot([yr, yr], [-0.06, 0.06], color=INK2, lw=1.5)
    ax.text(yr, -0.12, str(yr), ha="center", va="top", fontsize=13, color=INK2, zorder=6,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
STAGE = {"D": ("#8a8985", "o", "definition"), "P": (C2, "s", "proposal"), "O": (C1, "D", "committee opinion"),
         "R": (C3, "o", "final rule (reporting)"), "I": ("#52514e", "^", "industry statement"),
         "E": ("white", "o", "expected step")}
events = [(2021.5, 1, "D", "OECD PFAS definition"),
          (2022.97, 2, "I", "3M announces exit from\nPFAS manufacturing"),
          (2023.1, -1, "P", "EU universal PFAS\nrestriction proposal"),
          (2023.45, 1, "I", "SIA: PAG replacement\n15 to 20+ years (estimate)"),
          (2023.78, -2, "R", "US TSCA PFAS\nreporting rule"),
          (2025.63, 1, "P", "Updated EU proposal"),
          (2025.99, -1, "I", "3M reports exit\ncompleted"),
          (2026.18, 2, "O", "RAC opinion;\nSEAC draft opinion"),
          (2026.28, -2, "R", "TSCA submission\nstart moved"),
          (2026.95, 1, "E", "SEAC final opinion\nexpected; then\nCommission proposal")]
for x, s_, st, lab in events:
    c, mk, _ = STAGE[st]
    h = ({1: 0.4, 2: 1.6} if s_ > 0 else {1: 0.55, 2: 1.95})[abs(s_)] * np.sign(s_)
    ax.plot([x, x], [0, h], color=INK2 if st == "E" else c, lw=1.4, ls=(0, (2, 2)) if st == "E" else "-")
    ax.plot(x, 0, marker=mk, ms=10, color=c, mec=INK2 if st == "E" else "white", mew=1.5, zorder=5)
    ax.text(x, h + 0.06 * np.sign(s_), lab, ha="center", va="bottom" if s_ > 0 else "top", fontsize=12, color=INK,
            linespacing=1.05)
from matplotlib.lines import Line2D
hd = [Line2D([], [], ls="", marker=STAGE[k][1], ms=9, color=STAGE[k][0], mec=INK2 if k == "E" else "white", mew=1.2,
             label=STAGE[k][2]) for k in ("D", "P", "O", "R", "I", "E")]
ax.legend(handles=hd, loc="lower left", bbox_to_anchor=(0.02, -0.12), ncol=6, fontsize=12, handletextpad=0.3,
          columnspacing=1.2, frameon=False)
ax.text(2020.62, 0.0, "c", fontsize=22, fontweight="bold", va="center", ha="left")
save(fig, "Fig1_overview")
print("ok")
