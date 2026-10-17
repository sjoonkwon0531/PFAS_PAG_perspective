"""
Re-evaluate the Pareto-optimal designs of both MOBO campaigns, and the fluorinated baseline,
with the higher-fidelity protocol used for Table 2 and Fig. 5 (three-dose ladder at 0.85, 1.0
and 1.15 times the mean-field dose, three seeds, interpolation to CD = hp). This removes the
optimistic bias of selecting the best of single noisy evaluations.
Output: results/mobo_pareto_reeval.csv
"""
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from mobo import formulation, HP, THK, KP0, EL_MIN, SENS_TAG
from car_stochastic import Params, simulate, metrics, dose_to_size_meanfield, latent_image_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
SEEDS = (11, 12, 13)


def pareto(Y):
    keep = np.ones(len(Y), bool)
    for i in range(len(Y)):
        if np.any(np.all(Y <= Y[i], 1) & np.any(Y < Y[i], 1)):
            keep[i] = False
    return keep


def reeval(job):
    tag, x = job
    over, F_ug = formulation(x)
    p = Params(hp=HP, thickness=THK, kp=KP0, **over)
    d0 = dose_to_size_meanfield(p)
    lim = latent_image_metrics(p)
    rows = []
    for f in (0.85, 1.0, 1.15):
        acc = []
        for s in SEEDS:
            p.seed = s
            acc.append(metrics(simulate(p, d0 * f)["m"], p))
        rows.append(dict(dose=d0 * f, **{k: np.nanmean([a[k] for a in acc]) for k in acc[0]}))
    df = pd.DataFrame(rows).dropna().sort_values("dose")
    cd, dose = df.CD.values, df.dose.values
    ds = float(np.interp(HP, cd[::-1], dose[::-1]))
    out = dict(campaign=tag, f_F=x[0], n_I=x[1], s_PG=x[2], L_D=x[3], c_PAG=x[4], F_ug=F_ug,
               alpha_um=over["alpha_um"], kappa=over["kappa"], dose=ds,
               LWR=float(np.interp(ds, dose, df.LWR.values)),
               missing=float(np.interp(ds, dose, df.missing.values)), EL=lim["EL"],
               extrapolated=bool(not (cd.min() <= HP <= cd.max())))
    out["feasible"] = (out["EL"] >= EL_MIN) and (out["missing"] < 0.05)
    return out


if __name__ == "__main__":
    jobs = [("baseline", np.array([1.0, 0.0, 1.0, 6.0, 0.27]))]
    for tag, fn in (("fluorine-allowed", f"mobo_history{SENS_TAG}.csv"),
                    ("fluorine-free", f"mobo_history{SENS_TAG}_Ffree.csv")):
        d = pd.read_csv(os.path.join(OUT, fn))
        f = d[d.feasible].copy()
        f = f[pareto(f[["dose", "LWR"]].values)]
        for _, r in f.iterrows():
            jobs.append((tag, r[["f_F", "n_I", "lam", "L_D", "c_PAG"]].values.astype(float)))
    with Pool(2) as pool:
        res = pd.DataFrame(pool.map(reeval, jobs))
    res.to_csv(os.path.join(OUT, f"mobo_pareto_reeval{SENS_TAG}.csv"), index=False)
    pd.set_option("display.width", 220)
    print(res.round(2).to_string())
