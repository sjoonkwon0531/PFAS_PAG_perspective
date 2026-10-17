import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import *
R = os.path.join(HERE, "..", "sim", "results")
df = pd.read_csv(os.path.join(R, "map_kappa_LD_v2.csv"))
cm = LinearSegmentedColormap.from_list("blues", BLUES)
cm_r = LinearSegmentedColormap.from_list("blues_r", BLUES[::-1])

LDs = np.sort(df.L_D.unique()); ks = np.sort(df.kappa.unique())


def edges(v, log=False):
    v = np.log(v) if log else np.asarray(v, float)
    mid = (v[1:] + v[:-1]) / 2
    e = np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])
    return np.exp(e) if log else e


xe, ye = edges(LDs), edges(ks, log=True)
G0 = 1.266                                   # amplification number of the baseline (kappa = 1)
k_min = np.log(2) / G0
s = np.linspace(k_min / 0.3, (11.6 / 6.0) ** 2, 50)
traj_k, traj_L = 0.3 * s, 6.0 * np.sqrt(s)

fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.4), sharey=True, gridspec_kw=dict(wspace=0.32))
spec = [("dose", "Dose-to-size (mJ cm$^{-2}$)", cm, (20, 120), "{:.0f}"),
        ("LWR", "LWR at size (nm)", cm, (3.5, 11), "{:.1f}"),
        ("EL", "Mean-field exposure latitude (%)", cm_r, (10, 28), "{:.0f}")]
for ax, (col, lab, cmap, (vmin, vmax), fmt), letter in zip(axs, spec, "abc"):
    Z = df.pivot(index="kappa", columns="L_D", values=col).loc[ks, LDs].values
    Zm = np.ma.masked_invalid(Z)
    pc = ax.pcolormesh(xe, ye, Zm, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
                       edgecolors="white", linewidth=1.5)
    # no-pattern region
    for i, k in enumerate(ks):
        for j, L in enumerate(LDs):
            if not np.isfinite(Z[i, j]):
                ax.fill_between([xe[j], xe[j + 1]], ye[i], ye[i + 1], facecolor="#ecebe7",
                                hatch="///", edgecolor="#b9b8b3", linewidth=0)
            elif not (k == 1.0 and L == 6.0):
                v = Z[i, j]
                frac = (v - vmin) / (vmax - vmin)
                if cmap is cm_r:
                    frac = 1 - frac
                ax.text(L, k, fmt.format(v), ha="center", va="center", fontsize=11.5,
                        color="white" if frac > 0.55 else INK)
    ax.axhline(k_min, color=C2, lw=2.0)
    ax.plot(traj_L, traj_k, color=INK, lw=2.0, ls=(0, (5, 3)))
    ax.annotate("", xy=(traj_L[-1], traj_k[-1]), xytext=(traj_L[-6], traj_k[-6]),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.0, mutation_scale=18))
    ax.plot(6.0, 1.0, marker="*", ms=19, color=INK, mec="white", mew=1.5, zorder=5)
    ax.set_yscale("log")
    ax.set_yticks(ks); ax.set_yticklabels([f"{k:g}" for k in ks]); ax.minorticks_off()
    ax.set_xticks(LDs); ax.set_xticklabels([f"{L:g}" for L in LDs])
    ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(ye[0], ye[-1])
    ax.set_xlabel("Acid diffusion length, $L_D$ (nm)")
    if letter == "a":
        ax.set_ylabel(r"Relative effective activity, $\kappa$")
    else:
        ax.tick_params(axis="y", labelleft=False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    cb = fig.colorbar(pc, ax=ax, fraction=0.05, pad=0.03); cb.set_label(lab); cb.outline.set_visible(False)
    panel(ax, letter, x=-0.2, y=1.02)
axs[0].text(xe[-1] - 0.15, k_min * 0.965, r"$\Gamma$ = ln 2", color=C2, ha="right", va="top", fontsize=13,
            fontweight="bold")
axs[0].text(3.2, ye[0] * 1.03, "not sized below 300 mJ cm$^{-2}$", color=INK2, ha="left", va="bottom", fontsize=12.5)
axs[0].text(9.55, 0.88, "bake-equivalent", color=INK, fontsize=12.5, ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
save(fig, "Fig4_RLS_maps")
print(k_min)
