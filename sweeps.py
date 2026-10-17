"""
Scenario sweeps for the PFAS-free lithography Perspective.

Outputs (sim/results/):
  map_kappa_LD.csv     dose-to-size, LWR, LER, Z-factor over (acid strength kappa, diffusion length L_D)
  scenarios.csv        named formulation scenarios (fluorinated baseline -> PFAS-free compensations)
Z-factor (Wallow et al.): Z = hp^3 * LWR^2 * dose  (lower is better)
"""
import os, sys, itertools
import numpy as np
import pandas as pd
from multiprocessing import Pool
from car_stochastic import (Params, simulate, metrics, dose_to_size_meanfield,
                            amplification_number, latent_image_metrics)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
KP0 = 0.12725   # calibrated: mean-field dose-to-size = 35 mJ/cm2 for the nonaflate baseline
SEEDS = (1, 2)


def at_size(pdict):
    """Stochastic dose ladder around the mean-field dose; interpolate to CD = hp."""
    p = Params(**pdict)
    d0 = dose_to_size_meanfield(p)
    G = amplification_number(p)
    if not np.isfinite(d0):
        return dict(pdict, Gamma=G, dose_size=np.nan, LWR=np.nan, LER=np.nan, missing=np.nan,
                    dose_mf=np.nan, Z=np.nan, EL=np.nan, LILS=np.nan)
    lim = latent_image_metrics(p)
    rows = []
    for f in (0.8, 0.95, 1.1, 1.25):
        acc = []
        for s in SEEDS:
            p.seed = s
            acc.append(metrics(simulate(p, d0 * f)["m"], p))
        rows.append(dict(dose=d0 * f, **{k: np.nanmean([a[k] for a in acc]) for k in acc[0]}))
    df = pd.DataFrame(rows).dropna()
    df = df.sort_values("dose")
    cd = df.CD.values; dose = df.dose.values
    if len(cd) < 2 or not (cd.min() <= p.hp <= cd.max()):
        return dict(pdict, Gamma=G, dose_size=np.nan, LWR=np.nan, LER=np.nan,
                    missing=df.missing.max(), dose_mf=d0, Z=np.nan, **lim)
    # CD decreases with dose; interpolate on reversed arrays
    ds = float(np.interp(p.hp, cd[::-1], dose[::-1]))
    lwr = float(np.interp(ds, dose, df.LWR.values))
    ler = float(np.interp(ds, dose, df.LER.values))
    miss = float(np.interp(ds, dose, df.missing.values))
    Z = (p.hp ** 3) * (lwr ** 2) * ds * 1e-9          # (nm^5 * mJ/cm2) scaled by 1e-9
    return dict(pdict, Gamma=G, dose_size=ds, LWR=lwr, LER=ler, missing=miss, dose_mf=d0, Z=Z, **lim)


def run(jobs, name):
    with Pool(2) as pool:
        res = pool.map(at_size, jobs)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(OUT, name), index=False)
    return df


def scenario_jobs():
    base = dict(kp=KP0)
    S = {
        # label: overrides.  alpha from data/car_formulations_13p5nm.csv (equal 0.27 mmol/g PAG)
        "S0 TPS nonaflate (fluorinated baseline)":       dict(alpha_um=4.74, kappa=1.0, L_D=6.0),
        "S1 TPS tosylate drop-in (weaker acid)":          dict(alpha_um=4.36, kappa=0.3, L_D=6.0),
        "S2 S1 + longer PEB (x3.3)":                      dict(alpha_um=4.36, kappa=0.3 * 3.33, L_D=6.0 * np.sqrt(3.33)),
        "S3 S1 + low-Ea protecting group":                dict(alpha_um=4.36, kappa=1.0, L_D=6.0),
        "S4 S3 + polymer-bound anion":                    dict(alpha_um=4.36, kappa=1.0, L_D=4.0),
        "S5 S4 + iodinated aryl sulfonate (3 I)":          dict(alpha_um=5.27, kappa=1.0, L_D=4.0),
    }
    jobs = []
    for k, v in S.items():
        d = dict(base); d.update(v); d["label"] = k
        jobs.append(d)
    return jobs


def strip_label(d):
    d = dict(d); lab = d.pop("label"); return lab, d


def at_size_labeled(d):
    lab, pd_ = strip_label(d)
    r = at_size(pd_); r["label"] = lab; return r


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("scen", "all"):
        with Pool(2) as pool:
            res = pool.map(at_size_labeled, scenario_jobs())
        df = pd.DataFrame(res); df.to_csv(os.path.join(OUT, "scenarios.csv"), index=False)
        print(df[["label", "alpha_um", "kappa", "L_D", "Gamma", "dose_size", "LWR", "LER", "EL", "LILS", "Z",
                  "missing"]].round(3).to_string())
    if which in ("map", "all"):
        kappas = [0.45, 0.55, 0.65, 0.8, 1.0, 1.25, 1.6]
        LDs = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 11.0]
        jobs = [dict(kp=KP0, alpha_um=4.36, kappa=k, L_D=l) for k, l in itertools.product(kappas, LDs)]
        df = run(jobs, "map_kappa_LD.csv")
        print(df.pivot(index="kappa", columns="L_D", values="dose_size").round(1))
        print(df.pivot(index="kappa", columns="L_D", values="LWR").round(2))
        print(df.pivot(index="kappa", columns="L_D", values="EL").round(1))
