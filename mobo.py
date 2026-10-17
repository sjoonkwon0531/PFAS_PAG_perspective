"""
Multi-objective Bayesian optimization (ParEGO) of a CAR formulation with fluorine as an explicit objective.

Why MOBO: each stochastic evaluation is expensive and noisy, and the targets conflict
(dose, roughness, persistent-fluorine inventory). ParEGO (Knowles, IEEE Trans. Evol. Comput. 2006)
scalarizes the normalized objectives with a random augmented Tchebycheff weight each iteration
and maximizes expected improvement of a Gaussian-process surrogate.

Design variables (Table 3 of the draft):
  f_F   in [0, 1]  fluorination of the PAG anion (0 = aryl sulfonate, 1 = nonaflate)
  n_I   in [0, 3]  iodine atoms on the anion
  lam   in [1, 3]  protecting-group lability multiplier s_PG (lower activation energy)
  L_D   in [3, 10] nm acid diffusion length (anion bulk / polymer binding)
  c_PAG in [0.20, 0.40] mmol/g PAG loading
Mappings (assumptions, stated in the manuscript):
  kappa = kappa_acid(f_F) * lam,  kappa_acid = 0.3 + 0.7 f_F
  alpha = 4.43 + (4.36 + 0.38 f_F + 0.303 n_I - 4.43) * c_PAG/0.27  (1/um; fit to data/car_formulations_13p5nm.csv)
Objectives (minimize): LWR at size, dose-to-size, film fluorine inventory (ug/wafer).
Constraint: exposure latitude EL >= 12 %.
"""
import os, sys, json
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from car_stochastic import Params, simulate, metrics, dose_to_size_meanfield, latent_image_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
KP0, HP, THK = 0.12725, 12, 22
LB = np.array([0.0, 0.0, 1.0, 3.0, 0.20])
UB = np.array([1.0, 3.0, 3.0, 10.0, 0.40])
NAMES = ["f_F", "n_I", "lam", "L_D", "c_PAG"]
F_FILM_NONAFLATE_UG_PER_MMOLG = 117.48 / 0.27 * (THK / 30)   # from green_inventory.py, scaled to 22 nm film
EL_MIN = 12.0
# Sensitivity-analysis switches (defaults reproduce the main campaigns)
KAPPA_ARYL = float(os.environ.get("KAPPA_ARYL", "0.3"))   # relative strength of the non-fluorinated acid
SPG_MAX = float(os.environ.get("SPG_MAX", "3.0"))         # upper bound of protecting-group lability
KMAP = os.environ.get("KMAP", "lin")                      # 'lin': linear in f_F; 'log': log-linear in f_F
SENS_TAG = os.environ.get("SENS_TAG", "")
UB[2] = SPG_MAX


def kappa_acid(fF):
    if KMAP == "log":
        return KAPPA_ARYL ** (1.0 - fF)                    # rate log-linear in fluorination (pKa-like)
    return KAPPA_ARYL + (1.0 - KAPPA_ARYL) * fF


def decode(u):
    return LB + u * (UB - LB)


def formulation(x):
    fF, nI, lam, LD, c = x
    alpha_pag_part = (0.38 * fF + 0.303 * nI)
    # alpha(c) = alpha_poly + [alpha(0.27) - alpha_poly] * c / 0.27, alpha_poly = 4.43 (65:35 HS/tBA copolymer)
    alpha = 4.43 + (4.36 + alpha_pag_part - 4.43) * (c / 0.27)
    kappa = kappa_acid(fF) * lam
    F_ug = F_FILM_NONAFLATE_UG_PER_MMOLG * c * fF
    return dict(alpha_um=alpha, kappa=kappa, L_D=LD, pag_mmol_g=c), F_ug


def evaluate(x):
    over, F_ug = formulation(x)
    p = Params(hp=HP, thickness=THK, kp=KP0, **over)
    d0 = dose_to_size_meanfield(p)
    res = dict(zip(NAMES, x)); res.update(over); res["F_ug"] = F_ug
    if not np.isfinite(d0):
        res.update(dose=np.nan, LWR=np.nan, EL=np.nan, feasible=False); return res
    lim = latent_image_metrics(p)
    rows = []
    for f in (0.9, 1.1):
        acc = []
        for s in (1, 2):
            p.seed = s
            acc.append(metrics(simulate(p, d0 * f)["m"], p))
        rows.append(dict(dose=d0 * f, **{k: np.nanmean([a[k] for a in acc]) for k in acc[0]}))
    df = pd.DataFrame(rows)
    cd, dose = df.CD.values, df.dose.values
    if np.any(~np.isfinite(cd)) or cd[0] == cd[1]:
        res.update(dose=np.nan, LWR=np.nan, EL=lim["EL"], feasible=False); return res
    ds = dose[0] + (HP - cd[0]) * (dose[1] - dose[0]) / (cd[1] - cd[0])      # linear interpolation
    lwr = np.interp(ds, dose, df.LWR.values) if dose[0] <= ds <= dose[1] else \
        df.LWR.values[0] + (ds - dose[0]) * (df.LWR.values[1] - df.LWR.values[0]) / (dose[1] - dose[0])
    miss = float(df.missing.max())
    feas = (lim["EL"] >= EL_MIN) and (miss < 0.05) and (5 < ds < 200)
    res.update(dose=float(ds), LWR=float(lwr), EL=lim["EL"], missing=miss, feasible=bool(feas))
    return res


def objectives(df):
    Y = df[["LWR", "dose", "F_ug"]].values.astype(float)
    bad = ~df.feasible.values | ~np.all(np.isfinite(Y), axis=1)
    return Y, bad


def violation(df):
    """Constraint-violation score used while no feasible design has been found."""
    v = np.where(np.isfinite(df.dose.values), 0.0, 3.0)
    el = np.nan_to_num(df.EL.values.astype(float), nan=0.0)
    miss = np.nan_to_num(df.get("missing", pd.Series(np.zeros(len(df)))).values.astype(float), nan=1.0)
    return v + np.maximum(0, EL_MIN - el) / EL_MIN + 5 * np.maximum(0, miss - 0.05)


def parego_step(U, Y, bad, rng, n_cand=6000, df=None):
    good = ~bad
    if good.sum() == 0:                                   # feasibility search first
        s = violation(df)
    else:
        Yg = Y[good]
        lo, hi = Yg.min(0), Yg.max(0)
        Yn = (Y - lo) / np.where(hi > lo, hi - lo, 1)
        Yn[bad] = 1.5                                     # penalize infeasible points
        w = rng.dirichlet(np.ones(3))
        s = np.max(w * Yn, axis=1) + 0.05 * np.sum(w * Yn, axis=1)
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(5) * 0.3, nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=0)
    gp.fit(U, s)
    C = rng.random((n_cand, 5))
    # local candidates around the current best
    best = U[np.argmin(s)]
    C[: n_cand // 3] = np.clip(best + 0.08 * rng.standard_normal((n_cand // 3, 5)), 0, 1)
    mu, sd = gp.predict(C, return_std=True)
    imp = s.min() - mu
    z = imp / np.maximum(sd, 1e-9)
    ei = imp * norm.cdf(z) + sd * norm.pdf(z)
    return C[np.argmax(ei)]


if __name__ == "__main__":
    n_init = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tag = ""
    if len(sys.argv) > 3 and sys.argv[3] == "nofluorine":
        UB[0] = 0.0; tag = "_Ffree"                   # fluorine-free design space (f_F = 0)
    rng = np.random.default_rng(7)
    U = qmc.LatinHypercube(d=5, seed=7).random(n_init)
    # anchor points: fluorinated baseline (S0-like) and its drop-in
    anchors = np.array([[1.0, 0.0, 1.0, 6.0, 0.27], [0.0, 0.0, 1.0, 6.0, 0.27]])
    U = np.vstack([np.clip((anchors - LB) / np.where(UB > LB, UB - LB, 1), 0, 1), U])
    with Pool(2) as pool:
        recs = pool.map(evaluate, [decode(u) for u in U])
    df = pd.DataFrame(recs); df["iter"] = 0
    for it in range(1, n_iter + 1):
        Y, bad = objectives(df)
        newU = np.array([parego_step(U, Y, bad, rng, df=df) for _ in range(2)])
        with Pool(2) as pool:
            recs = pool.map(evaluate, [decode(u) for u in newU])
        d2 = pd.DataFrame(recs); d2["iter"] = it
        df = pd.concat([df, d2], ignore_index=True)
        U = np.vstack([U, newU])
        df.to_csv(os.path.join(OUT, f"mobo_history{SENS_TAG}{tag}.csv"), index=False)
        fb = df[df.feasible]
        print(f"iter {it:2d}  n={len(df)}  feasible={len(fb)}  best LWR={fb.LWR.min():.2f}  "
              f"best F-free LWR={fb[fb.F_ug < 1].LWR.min() if (fb.F_ug < 1).any() else np.nan:.2f}", flush=True)
    df.to_csv(os.path.join(OUT, f"mobo_history{SENS_TAG}{tag}.csv"), index=False)
