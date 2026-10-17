import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import *
R = os.path.join(HERE, "..", "sim", "results")
H = pd.read_csv(os.path.join(R, "emulator_v2_holdout.csv"))
N = pd.read_csv(os.path.join(R, "emulator_v2_nuisance_premium_rel.csv")) if os.path.exists(os.path.join(R, "emulator_v2_nuisance_premium_rel.csv")) else None
fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.0), gridspec_kw=dict(wspace=0.36))
ax = axs[0]
ax.errorbar(H.dose, H.dose_pred, xerr=H.dose_se, fmt="o", color=C1, mec="white", mew=1.2, ms=8, capsize=2)
lim = [H.dose.min() * 0.9, H.dose.max() * 1.1]; ax.plot(lim, lim, color=MUTED, lw=1.2, ls=(0, (4, 3)))
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xscale("log"); ax.set_yscale("log")
from matplotlib.ticker import FixedLocator, NullLocator
for a_ in (ax.xaxis, ax.yaxis):
    a_.set_major_locator(FixedLocator([15, 20, 30, 40])); a_.set_minor_locator(NullLocator())
ax.set_xticklabels(["15", "20", "30", "40"]); ax.set_yticklabels(["15", "20", "30", "40"])
ax.set_xlabel("Simulated dose-to-size (mJ cm$^{-2}$)"); ax.set_ylabel("Emulator prediction (mJ cm$^{-2}$)")
ax.text(0.04, 0.96, f"RMSE {np.sqrt(np.mean(H.dose_err_pct ** 2)):.1f}%", transform=ax.transAxes, fontsize=13, va="top")
panel(ax, "a", x=-0.2)
ax = axs[1]
ax.errorbar(H.LWR, H.LWR_pred, xerr=H.LWR_se, fmt="o", color=C1, mec="white", mew=1.2, ms=8, capsize=2)
lim = [H.LWR.min() - 0.3, H.LWR.max() + 0.3]; ax.plot(lim, lim, color=MUTED, lw=1.2, ls=(0, (4, 3)))
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Simulated LWR (nm)"); ax.set_ylabel("Emulator prediction (nm)")
ax.text(0.04, 0.96, f"RMSE {np.sqrt(np.mean(H.LWR_err ** 2)):.2f} nm\nmean s.e. {H.LWR_se.mean():.2f} nm", transform=ax.transAxes,
        fontsize=13, va="top")
panel(ax, "b", x=-0.2)
ax = axs[2]
if N is not None:
    col = {"ref": INK, "phi1.5": C1, "phi3": C2, "sigma3": C3, "mth0.6": C4}
    lab = {"ref": "reference", "phi1.5": "$\\phi$ = 1.5", "phi3": "$\\phi$ = 3", "sigma3": "$\\sigma_e$ = 3 nm", "mth0.6": "$m_{th}$ = 0.6"}
    tsel = "premium_rel0.25"
    for k in ["ref", "phi1.5", "phi3", "mth0.6", "sigma3"]:
        g = N[N.variant == k]
        ax.plot(g.kappa_max_free, g[tsel], color=col.get(k, MUTED), marker="o", ms=7, mec="white", mew=1.2, label=lab.get(k, k),
                lw=2.4 if k == "ref" else 1.8)
    ax.set_xlabel(r"Largest fluorine-free activity, $\kappa_{aryl}s_{PG,max}$")
    ax.set_ylabel("Premium at best LWR + 0.25 nm (%)")
    ax.legend(fontsize=12, loc="upper right")
panel(ax, "c", x=-0.2)
save(fig, "FigS3_emulator_checks")
print("ok")
