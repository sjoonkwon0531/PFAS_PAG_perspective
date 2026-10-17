"""
Emulator-based sensitivity of the fluorine-free dose premium (12 nm half-pitch, c_PAG = 0.40 mmol/g).

The stochastic response tabulated in results/emulator_table.csv is interpolated in
(alpha, ln kappa) at each diffusion-length node. Every assumption set is then evaluated on the
same dense design grid, which removes the run-to-run variability of the optimizer.
Output: results/emulator_sensitivity_summary.csv, results/emulator_fronts.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
T = pd.read_csv(os.path.join(R, "emulator_table.csv"))
ALPHAS = np.sort(T.alpha_um.unique()); KAPPAS = np.sort(T.kappa.unique()); LDS = np.sort(T.L_D.unique())
CPAG, EL_MIN, MISS_MAX = 0.40, 12.0, 0.05
F_PER_FF = 117.48 / 0.27 * (22 / 30) * CPAG          # film fluorine (ug per wafer-layer) at f_F = 1

interp = {}
for L in LDS:
    sub = T[T.L_D == L]
    for q, tr in (("dose", np.log), ("LWR", None), ("EL", None), ("missing", None)):
        Z = sub.pivot(index="alpha_um", columns="kappa", values=q).loc[ALPHAS, KAPPAS].values
        if tr is not None:
            Z = tr(Z)
        interp[(L, q)] = RegularGridInterpolator((ALPHAS, np.log(KAPPAS)), Z, bounds_error=False, fill_value=np.nan)

CASES = [  # key, label, kappa_aryl, s_PG max, mapping
    ("ref", "Reference", 0.3, 3.0, "lin"),
    ("ka02", "Weaker aryl acid (κ_aryl = 0.2)", 0.2, 3.0, "lin"),
    ("ka05", "Stronger non-F acid (κ_aryl = 0.5)", 0.5, 3.0, "lin"),
    ("spg2", "Less labile groups (s_PG ≤ 2)", 0.3, 2.0, "lin"),
    ("spg5", "More labile groups (s_PG ≤ 5)", 0.3, 5.0, "lin"),
    ("logk", "Log-linear κ mapping", 0.3, 3.0, "log"),
]
TARGETS = [4.2, 4.3, 4.4, 4.5, 4.75, 5.0]


def kappa_acid(fF, ka, mode):
    return ka ** (1 - fF) if mode == "log" else ka + (1 - ka) * fF


def evaluate(ka, smax, mode, free):
    fF = np.array([0.0]) if free else np.linspace(0, 1, 21)
    nI = np.linspace(0, 3, 13)
    sPG = np.linspace(1, smax, 21)
    out = []
    for L in LDS:
        F, N, S = np.meshgrid(fF, nI, sPG, indexing="ij")
        F, N, S = F.ravel(), N.ravel(), S.ravel()
        alpha = 4.43 + (4.36 + 0.38 * F + 0.303 * N - 4.43) * (CPAG / 0.27)
        kap = kappa_acid(F, ka, mode) * S
        pts = np.c_[alpha, np.log(kap)]
        d = np.exp(interp[(L, "dose")](pts)); lwr = interp[(L, "LWR")](pts)
        el = interp[(L, "EL")](pts); ms = interp[(L, "missing")](pts)
        out.append(pd.DataFrame(dict(f_F=F, n_I=N, s_PG=S, L_D=L, alpha=alpha, kappa=kap, dose=d, LWR=lwr,
                                     EL=el, missing=ms, F_ug=F_PER_FF * F)))
    df = pd.concat(out, ignore_index=True)
    df["feasible"] = np.isfinite(df.dose) & np.isfinite(df.LWR) & (df.EL >= EL_MIN) & (df.missing < MISS_MAX)
    return df


def pareto_front(df):
    f = df[df.feasible].sort_values(["dose", "LWR"])
    best, keep = np.inf, []
    for i, r in f.iterrows():
        if r.LWR < best - 1e-9:
            keep.append(i); best = r.LWR
    return f.loc[keep]


rows, fronts = [], []
for key, label, ka, smax, mode in CASES:
    A = pareto_front(evaluate(ka, smax, mode, free=False))
    B = pareto_front(evaluate(ka, smax, mode, free=True))
    for nm, f in (("fluorine-allowed", A), ("fluorine-free", B)):
        g = f.copy(); g["case"] = key; g["campaign"] = nm; fronts.append(g)
    row = dict(case=key, label=label, kappa_aryl=ka, sPG_max=smax, mapping=mode,
               min_LWR_allowed=A.LWR.min(), min_LWR_free=B.LWR.min() if len(B) else np.nan,
               min_dose_allowed=A.dose.min(), min_dose_free=B.dose.min() if len(B) else np.nan)
    for t in TARGETS:
        da = A[A.LWR <= t].dose.min() if (A.LWR <= t).any() else np.nan
        db = B[B.LWR <= t].dose.min() if len(B) and (B.LWR <= t).any() else np.nan
        row[f"dA_{t}"], row[f"dB_{t}"] = da, db
        row[f"premium_{t}"] = 100 * (db / da - 1) if np.isfinite(da) and np.isfinite(db) else np.nan
    rows.append(row)

S = pd.DataFrame(rows)
S.to_csv(os.path.join(R, "emulator_sensitivity_summary.csv"), index=False)
pd.concat(fronts, ignore_index=True).to_csv(os.path.join(R, "emulator_fronts.csv"), index=False)
pd.set_option("display.width", 250)
print(S.round(2).to_string())

# Premium as a function of the largest acid strength reachable without fluorine,
# kappa_max,free = kappa_aryl * s_PG,max (here s_PG,max = 3 and kappa_aryl varied).
sweep = []
for kmax in np.round(np.arange(0.6, 2.01, 0.1), 2):
    ka = kmax / 3.0
    A = pareto_front(evaluate(ka, 3.0, "lin", free=False))
    B = pareto_front(evaluate(ka, 3.0, "lin", free=True))
    r = dict(kappa_max_free=kmax, min_LWR_free=B.LWR.min() if len(B) else np.nan,
             min_dose_free=B.dose.min() if len(B) else np.nan)
    for t in (4.4, 4.5, 4.75, 5.0):
        da = A[A.LWR <= t].dose.min() if (A.LWR <= t).any() else np.nan
        db = B[B.LWR <= t].dose.min() if len(B) and (B.LWR <= t).any() else np.nan
        r[f"premium_{t}"] = 100 * (db / da - 1) if np.isfinite(da) and np.isfinite(db) else np.nan
    sweep.append(r)
W = pd.DataFrame(sweep)
W.to_csv(os.path.join(R, "emulator_kappa_max_sweep.csv"), index=False)
print(W.round(1).to_string())
