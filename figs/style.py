"""Shared figure style: Arial Narrow (PT Sans Narrow as on-screen fallback), large fonts,
panel labels a, b, c only. Vector outputs keep text as text, so the SVG/PDF render in
Arial Narrow on any machine that has it installed."""
import matplotlib as mpl
import matplotlib.pyplot as plt

FONT = ["Arial Narrow", "PT Sans Narrow", "Liberation Sans Narrow", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT,
    "mathtext.fontset": "custom",
    "mathtext.rm": "PT Sans Narrow",
    "mathtext.it": "PT Sans Narrow:italic",
    "mathtext.bf": "PT Sans Narrow:bold",
    "mathtext.fallback": "stixsans",
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#52514e",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.dpi": 600,
    "figure.dpi": 150,
})

# reference categorical palette (validated slots 1-3) + neutral reference ink
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8985"; GRID = "#e4e3df"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
REF = "#52514e"          # fluorinated baseline = neutral reference
BLUES = ["#eef4fc", "#cfe0f6", "#a4c5ee", "#6fa3e3", "#2a78d6", "#1d5aa6", "#123c70"]


def panel(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=22, fontweight="bold",
            va="bottom", ha="left", color=INK)


def save(fig, name):
    import os
    d = os.path.dirname(os.path.abspath(__file__))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
