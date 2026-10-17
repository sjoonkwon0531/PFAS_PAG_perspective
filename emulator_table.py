"""
Response table for an emulator-based sensitivity analysis at 12 nm half-pitch.

Why an emulator: the optimization assumptions (how fluorination maps onto acid strength, how
labile a protecting group may be) enter the simulator only through the absorption coefficient
alpha and the relative acid strength kappa. Tabulating the stochastic response over
(alpha, kappa, L_D) once, at the PAG loading where all Pareto designs converged (0.40 mmol/g),
lets every assumption set be evaluated on the same data by dense enumeration, free of
optimizer noise.

Protocol per node: mean-field dose-to-size, three-dose ladder (0.85, 1.0, 1.15 x), three seeds,
interpolation to CD = hp; EL from the mean-field model.
Output: results/emulator_table.csv
"""
import os, itertools
import numpy as np
import pandas as pd
from multiprocessing import Pool
from car_stochastic import Params, simulate, metrics, dose_to_size_meanfield, latent_image_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
KP0, HP, THK, CPAG = 0.12725, 12, 22, 0.40
ALPHAS = [4.3, 5.0, 5.7, 6.4]
KAPPAS = [0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.6, 2.0, 3.0, 5.0]
LDS = [5.0, 6.0, 7.0, 8.0]
SEEDS = (21, 22, 23)


def node(args):
    a, k, L = args
    p = Params(hp=HP, thickness=THK, kp=KP0, alpha_um=a, kappa=k, L_D=L, pag_mmol_g=CPAG)
    d0 = dose_to_size_meanfield(p)
    out = dict(alpha_um=a, kappa=k, L_D=L, dose_mf=d0)
    if not np.isfinite(d0):
        out.update(dose=np.nan, LWR=np.nan, missing=np.nan, EL=np.nan); return out
    lim = latent_image_metrics(p)
    rows = []
    for f in (0.85, 1.0, 1.15):
        acc = []
        for s in SEEDS:
            p.seed = s
            acc.append(metrics(simulate(p, d0 * f)["m"], p))
        rows.append(dict(dose=d0 * f, **{q: np.nanmean([x[q] for x in acc]) for q in acc[0]}))
    df = pd.DataFrame(rows).dropna().sort_values("dose")
    cd, dose = df.CD.values, df.dose.values
    if len(cd) < 2:
        out.update(dose=np.nan, LWR=np.nan, missing=np.nan, EL=lim["EL"]); return out
    ds = float(np.interp(HP, cd[::-1], dose[::-1]))
    out.update(dose=ds, LWR=float(np.interp(ds, dose, df.LWR.values)),
               missing=float(np.interp(ds, dose, df.missing.values)), EL=lim["EL"],
               bracketed=bool(cd.min() <= HP <= cd.max()))
    return out


if __name__ == "__main__":
    jobs = list(itertools.product(ALPHAS, KAPPAS, LDS))
    with Pool(2) as pool:
        res = pd.DataFrame(pool.map(node, jobs, chunksize=4))
    res.to_csv(os.path.join(OUT, "emulator_table.csv"), index=False)
    pd.set_option("display.width", 200)
    print(res.pivot_table(index="kappa", columns="L_D", values="LWR").round(2))
    print(res.pivot_table(index="kappa", columns="L_D", values="dose").round(1))
