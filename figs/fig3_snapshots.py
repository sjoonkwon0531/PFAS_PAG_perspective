import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "sim"))
from style import *
from car_stochastic import Params, MORParams, simulate, simulate_mor, dose_to_size_meanfield, mor_dose_to_size

KP0 = 0.12725
cm = LinearSegmentedColormap.from_list("blues", BLUES)
cm_ph = LinearSegmentedColormap.from_list("ink", ["#fcfcfb", "#b9b8b3", "#52514e", "#0b0b0b"])
XW = 96  # nm shown along the line

p0 = Params(kp=KP0, alpha_um=4.74, kappa=1.0, L_D=6.0, Lx=XW, seed=11)
D0 = dose_to_size_meanfield(p0)
r0 = simulate(p0, D0, return_fields=True)
p1 = Params(kp=KP0, alpha_um=4.36, kappa=0.3, L_D=6.0, Lx=XW, seed=11)
r1 = simulate(p1, D0)
p2 = Params(kp=KP0, alpha_um=4.36, kappa=0.999, L_D=10.95, Lx=XW, seed=11)
r2 = simulate(p2, dose_to_size_meanfield(p2))
p5 = Params(kp=KP0, alpha_um=5.27, kappa=1.0, L_D=4.0, Lx=XW, seed=11)
r5 = simulate(p5, dose_to_size_meanfield(p5))

fig, axs = plt.subplots(2, 3, figsize=(14.5, 9.6), gridspec_kw=dict(hspace=0.42, wspace=0.28))
ext = [0, 64, 0, XW]


def show(ax, img, cmap, vmin, vmax, letter, contour=None, note=None):
    im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
                   interpolation="nearest")
    if contour is not None:
        yy, xx = np.meshgrid(np.arange(img.shape[1]) + 0.5, np.arange(img.shape[0]) + 0.5)
        ax.contour(yy, xx, img, levels=[contour], colors=[C2], linewidths=1.8)
    ax.set_xticks([0, 32, 64]); ax.set_yticks([0, 32, 64, 96])
    ax.set_xlabel("y (nm)"); ax.set_ylabel("x (nm)")
    if False:  # panel headings removed (panel letters only; details in the caption)
        ax.text(0.5, 1.015, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=14, color=INK)
    panel(ax, letter, x=-0.42, y=1.0)
    return im


z = p0.thickness // 2
im = show(axs[0, 0], r0["nph"].sum(axis=2), cm_ph, 0, None, "a")
cb = fig.colorbar(im, ax=axs[0, 0], fraction=0.05, pad=0.04); cb.set_label("Absorbed photons per nm$^2$ column")
im = show(axs[0, 1], r0["acid0"].sum(axis=2), cm_ph, 0, None, "b")
cb = fig.colorbar(im, ax=axs[0, 1], fraction=0.05, pad=0.04); cb.set_label("Photoacids per nm$^2$ column")
im = show(axs[0, 2], r0["m"][:, :, z], cm, 0, 1, "c", contour=0.5,
          note=f"Fluorinated baseline, {D0:.0f} mJ cm$^{{-2}}$")
cb = fig.colorbar(im, ax=axs[0, 2], fraction=0.05, pad=0.04); cb.set_label("Deprotection, m")
im = show(axs[1, 0], r1["m"][:, :, z], cm, 0, 1, "d", contour=0.5,
          note=f"Weak-acid drop-in, $\\Gamma$ < ln 2")
im = show(axs[1, 1], r2["m"][:, :, z], cm, 0, 1, "e", contour=0.5,
          note="Bake-equivalent, $L_D$ = 11 nm")
im = show(axs[1, 2], r5["m"][:, :, z], cm, 0, 1, "f", contour=0.5,
          note="Co-designed, $L_D$ = 4 nm proxy, I$_3$ anion")
for ax in axs[1]:
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04).set_label("Deprotection, m")
save(fig, "Fig3_snapshots")
print("doses", D0, float(r1["m"].max()))
