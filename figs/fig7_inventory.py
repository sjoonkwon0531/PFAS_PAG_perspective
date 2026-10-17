import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "sim"))
from style import *
from green_inventory import car_inventory, mor_inventory
R = os.path.join(HERE, "..", "sim", "results")
inv = pd.read_csv(os.path.join(R, "fluorine_inventory.csv"))

fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.0), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.15], wspace=0.5))

# a: film-bound fluorine per wafer-layer by PAG anion
ax = axs[0]
f = inv[inv.F_film_ug > 0].iloc[::-1]
names = {"PFOS": "TPS PFOS", "Nonaflate": "TPS nonaflate", "Triflate": "TPS triflate"}
ax.barh([names[a] for a in f.anion], f.F_film_ug, color=REF, height=0.6)
for i, (v, d) in enumerate(zip(f.F_film_ug, f.F_dispensed_ug)):
    ax.text(v + 4, i, f"{v:.0f} µg", va="center", fontsize=13, color=INK)
ax.barh(["Any F-free anion"], [0], color=C1)
ax.text(4, 3, "0 µg", va="center", fontsize=13, color=C1, fontweight="bold")
ax.set_xlabel("Fluorine in film (µg per wafer-layer)")
ax.set_xlim(0, 290); ax.tick_params(axis="y", length=0)
panel(ax, "a", x=-0.62)

# b: dispensed element mass vs spin-coat utilization
ax = axs[1]
eta = np.logspace(np.log10(0.01), np.log10(0.5), 60)
Fn = np.array([car_inventory("Nonaflate", eta_u=e)["F_dispensed_ug"] for e in eta])
Fp = np.array([car_inventory("PFOS", eta_u=e)["F_dispensed_ug"] for e in eta])
Sn = np.array([mor_inventory(eta_u=e)["Sn_dispensed_ug"] for e in eta])
ax.plot(100 * eta, Fp / 1e3, color=REF, ls=(0, (5, 3)))
ax.plot(100 * eta, Fn / 1e3, color=REF)
ax.plot(100 * eta, Sn / 1e3, color=C3)
ax.text(1.15, Fp[0] / 1e3 * 1.25, "F, TPS PFOS", fontsize=12.5, color=INK, va="bottom")
ax.text(1.0, Fn[0] / 1e3 * 0.16, "F, TPS nonaflate", fontsize=12.5, color=INK, va="top")
ax.text(1.15, Sn[0] / 1e3 * 1.25, "Sn, tin-oxo MOR", fontsize=12.5, color=INK, va="bottom")
ax.axvspan(2, 10, color=GRID, alpha=0.6, lw=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Spin-coat utilization, $\\eta_u$ (%)")
ax.set_ylabel("Dispensed mass (mg per wafer-layer)")
ax.set_xticks([1, 2, 5, 10, 20, 50]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
ax.set_ylim(0.05, 600)
panel(ax, "b", x=-0.24)

# c: allocation of dispensed fluorine (nonaflate) in absolute mass for three utilizations
ax = axs[2]
cases = [0.02, 0.05, 0.20]
streams = [("F_spin_waste_ug", "Spin-off and edge-bead waste", C1),
           ("F_developer_ug", "Developer", C2),
           ("F_strip_ug", "Plasma strip", C3)]
h = 0.24
for i, e in enumerate(cases):
    r = car_inventory("Nonaflate", eta_u=e)
    tot = r["F_dispensed_ug"]
    for j, (key, lab, c) in enumerate(streams):
        y = i + (j - 1) * (h + 0.04)
        v = r[key]
        ax.barh(y, v, height=h, color=c, label=lab if i == 0 else None)
        ax.text(v * 1.12, y, f"{v:,.0f} µg ({100 * v / tot:.0f}%)" if v >= 100 else f"{v:.0f} µg ({100 * v / tot:.1f}%)",
                va="center", fontsize=11.5, color=INK)
ax.set_xscale("log"); ax.set_xlim(10, 6e4)
ax.set_yticks(range(3)); ax.set_yticklabels([f"$\\eta_u$ = {100 * e:g}%" for e in cases])
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Fluorine allocated (µg per wafer-layer)")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=12, handlelength=1.2, columnspacing=1.0)
ax.invert_yaxis()
panel(ax, "c", x=-0.3)
save(fig, "Fig7_inventory")
print("ok")
