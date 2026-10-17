import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
atoms = pd.read_csv(os.path.join(ROOT, "data", "henke_f2_13p5nm.csv"))
car = pd.read_csv(os.path.join(ROOT, "data", "car_formulations_13p5nm.csv"))

fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.0), gridspec_kw=dict(width_ratios=[0.9, 1.15, 1.0], wspace=0.75))

# a: per-atom photoabsorption cross section at 13.5 nm
ax = axs[0]
a = atoms.copy(); a["Mb"] = a.sigma_cm2 / 1e-18
a = a[a.element.isin(["H", "C", "N", "O", "F", "S", "Si", "Cl", "Br", "Zr", "Hf", "Zn", "Bi", "Sn", "Sb", "I", "Te"])]
a = a.sort_values("Mb")
col = [C2 if e == "F" else (C1 if e in ("I", "Sn", "Te", "Sb", "Bi", "Zn", "Hf") else "#b9b8b3") for e in a.element]
ax.barh(a.element, a.Mb, color=col, height=0.72)
ax.set_xscale("log")
ax.set_xlabel(r"$\sigma_a$ at 13.5 nm (Mb atom$^{-1}$)")
ax.tick_params(axis="y", length=0)
for e, v in zip(a.element, a.Mb):
    if e in ("F", "I", "Sn", "H", "C", "O"):
        ax.text(v * 1.15, list(a.element).index(e), f"{v:.2g}", va="center", fontsize=12, color=INK2)
ax.set_xlim(0.015, 80)
panel(ax, "a", x=-0.2)

# b: formulation absorption coefficient by PAG anion (equal molar loading 0.27 mmol/g)
ax = axs[1]
lab = {"PFOS": "TPS PFOS", "Nonaflate": "TPS nonaflate", "Triflate": "TPS triflate", "Tosylate": "TPS tosylate",
       "Camphorsulfonate": "TPS camphorsulfonate", "4-Iodobenzenesulfonate": "TPS 4-iodobenzenesulfonate",
       "2,4,6-Triiodobenzenesulfonate": "TPS 2,4,6-triiodobenzenesulfonate"}
c2 = car.iloc[::-1].reset_index(drop=True)
y = np.arange(len(c2))
cols = [REF if f else C1 for f in c2.fluorinated]
ax.barh(y, c2.alpha_um_inv - 4.0, left=4.0, color=cols, height=0.68)
poly = 4.43   # 65:35 (molar) HS/tBA copolymer, data/euv_absorption.py
ax.axvline(poly, color=MUTED, lw=1.2, ls=(0, (4, 3)))
ax.text(poly + 0.015, len(c2) - 0.42, "polymer only", ha="left", va="center", fontsize=12, color=INK2)
for i, r in c2.iterrows():
    t = f"{r.alpha_um_inv:.2f}" + (f"  (F {r.F_wt_pct:.1f} wt%)" if r.F_wt_pct > 0 else "")
    ax.text(r.alpha_um_inv + 0.02, i, t, ha="left", va="center", fontsize=12.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in c2.anion], fontsize=13.5)
ax.tick_params(axis="y", length=0)
ax.set_xlim(4.0, 5.75)
ax.set_xlabel(r"Absorption coefficient, $\alpha$ ($\mu$m$^{-1}$)")
panel(ax, "b", x=-0.62)

# c: absorbed photons per hp x hp resist column at 35 mJ/cm2
ax = axs[2]
E_PH = 6.62607015e-34 * 2.99792458e8 / 13.5e-9
phi = 35e-3 / E_PH * 1e-14          # photons nm^-2
hp = np.array([10, 12, 16])
thk_car = np.array([18, 22, 30]); thk_mor = np.array([12, 15, 20])
cases = [("Nonaflate CAR", 4.74, thk_car, REF, "o", (0, ())),
         ("Tosylate CAR", 4.36, thk_car, C1, "s", (0, ())),
         ("I$_3$-aryl sulfonate CAR", 5.27, thk_car, C2, "D", (0, ())),
         ("Tin-oxo MOR", 13.71, thk_mor, C3, "^", (0, ()))]
for name, al, th, c, mk, ls in cases:
    N = phi * hp ** 2 * (1 - np.exp(-al * 1e-3 * th))
    ax.plot(hp, N, color=c, marker=mk, ls=ls, mec="white", mew=1.5, ms=9)
    ax.lines[-1].set_label(name)
ax.set_xlabel("Half-pitch (nm)")
ax.set_ylabel(r"Absorbed photons per hp$^2$ column")
ax.set_xticks(hp); ax.set_xlim(9.4, 16.6)
ax.legend(loc="lower right", handlelength=1.6, fontsize=12.5)
ax.set_yscale("log"); ax.set_ylim(60, 2000)
panel(ax, "c", x=-0.2)
save(fig, "Fig2_absorption")
print("ok")
