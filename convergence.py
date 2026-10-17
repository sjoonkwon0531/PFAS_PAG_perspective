"""
Time-step and grid convergence of the v2 solver (review item E2).
(a) Same realization (seed fixed), dx = 1 nm, n_steps = 30 ... 960 at fixed dose: CD, LWR, LER, missing.
(b) Grid: dx = 1.0 and 0.5 nm, ensemble of independent seeds at fixed dose (different grids give
    different Poisson realizations, so ensemble means and standard errors are compared).
(c) Mean-field dose-to-size versus n_steps and dx.
Output: results/convergence_dt.csv, results/convergence_dx.csv, results/convergence_meanfield.csv
"""
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from car_stochastic import Params, simulate, metrics, dose_to_size_meanfield

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
KP = float(os.environ.get("KP", "0.12725"))
DOSE = float(os.environ.get("DOSE", "35.0"))


def run_dt(n):
    p = Params(kp=KP, seed=1, n_steps=n)
    return dict(n_steps=n, dt=p.t_peb / n, **metrics(simulate(p, DOSE)["m"], p))


def run_dx(args):
    dx, seed = args
    p = Params(kp=KP, seed=seed, dx=dx, n_steps=120)
    return dict(dx=dx, seed=seed, **metrics(simulate(p, DOSE)["m"], p))


def run_mf(args):
    dx, n = args
    p = Params(kp=KP, dx=dx, n_steps=n)
    return dict(dx=dx, n_steps=n, dose_to_size=dose_to_size_meanfield(p))


if __name__ == "__main__":
    with Pool(2) as pool:
        a = pd.DataFrame(pool.map(run_dt, [30, 60, 120, 240, 480, 960]))
        b = pd.DataFrame(pool.map(run_dx, [(dx, s) for dx in (1.0, 0.5) for s in range(101, 107)]))
        c = pd.DataFrame(pool.map(run_mf, [(dx, n) for dx in (1.0, 0.5, 0.25) for n in (60, 240, 960)]))
    a.to_csv(os.path.join(OUT, "convergence_dt.csv"), index=False)
    b.to_csv(os.path.join(OUT, "convergence_dx.csv"), index=False)
    c.to_csv(os.path.join(OUT, "convergence_meanfield.csv"), index=False)
    pd.set_option("display.width", 200)
    print(a.round(3).to_string())
    g = b.groupby("dx")[["CD", "LWR", "LER", "missing"]]
    print(pd.concat([g.mean().add_suffix("_mean"), (g.std() / np.sqrt(g.count())).add_suffix("_se")], axis=1).round(3))
    print(c.round(3).to_string())
