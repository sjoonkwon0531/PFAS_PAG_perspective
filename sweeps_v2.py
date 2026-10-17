"""
v2 recomputation of Table 2 (named scenarios) and Fig. 4 (kappa x L_D map) with the conservative
lattice solver, dt = 0.25 s, guarded sizing and seed-based standard errors.

Scenario names follow the internal review (S6): S2 is an idealized bake-equivalent compensation
(effective activity and diffusion length scaled, quencher diffusion fixed); S2b additionally scales
the quencher diffusion length; S4 is a reduced-diffusion proxy for polymer-bound anions.
Outputs: results/scenarios_v2.csv, results/map_kappa_LD_v2.csv
"""
import os, sys, itertools
import numpy as np
import pandas as pd
from multiprocessing import Pool
from car_stochastic import Params, latent_image_metrics
from sizing import size_at

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
KP0 = 0.12725
T = 3.33

SCEN = {
    "S0 Fluorinated baseline (TPS nonaflate)": dict(alpha_um=4.74, kappa=1.0, L_D=6.0),
    "S1 Weak-acid drop-in (TPS tosylate)": dict(alpha_um=4.36, kappa=0.3, L_D=6.0),
    "S2 S1 + idealized bake-equivalent compensation": dict(alpha_um=4.36, kappa=0.3 * T, L_D=6.0 * np.sqrt(T)),
    "S2b S2 with quencher diffusion also scaled": dict(alpha_um=4.36, kappa=0.3 * T, L_D=6.0 * np.sqrt(T),
                                                      L_Q=1.5 * np.sqrt(T)),
    "S3 S1 + more labile protecting group": dict(alpha_um=4.36, kappa=1.0, L_D=6.0),
    "S4 S3 + reduced-diffusion proxy": dict(alpha_um=4.36, kappa=1.0, L_D=4.0),
    "S5 S4 + 2,4,6-triiodobenzenesulfonate": dict(alpha_um=5.27, kappa=1.0, L_D=4.0),
}


def job_scen(item):
    label, over = item
    p = Params(kp=KP0, **over)
    r = size_at(p, seeds=(1, 2, 3, 4, 5, 6))
    lim = latent_image_metrics(p) if r["status"] == "ok" else dict(EL=np.nan, LILS=np.nan)
    return dict(label=label, **over, **r, **lim)


def job_map(args):
    k, L = args
    p = Params(kp=KP0, alpha_um=4.36, kappa=k, L_D=L)
    r = size_at(p, seeds=(1, 2, 3))
    lim = latent_image_metrics(p) if r["status"] == "ok" else dict(EL=np.nan, LILS=np.nan)
    return dict(kappa=k, L_D=L, **r, **lim)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 250)
    if which in ("scen", "all"):
        with Pool(2) as pool:
            df = pd.DataFrame(pool.map(job_scen, list(SCEN.items())))
        df.to_csv(os.path.join(OUT, "scenarios_v2.csv"), index=False)
        print(df[["label", "status", "Gamma", "dose", "dose_se", "LWR", "LWR_se", "missing", "EL"]].round(3).to_string(),
              flush=True)
    if which in ("map", "all"):
        kappas = [0.45, 0.55, 0.65, 0.8, 1.0, 1.25, 1.6]
        LDs = [3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 11.0]
        with Pool(2) as pool:
            df = pd.DataFrame(pool.map(job_map, list(itertools.product(kappas, LDs))))
        df.to_csv(os.path.join(OUT, "map_kappa_LD_v2.csv"), index=False)
        for c in ("status", "dose", "LWR", "EL", "missing"):
            print(df.pivot(index="kappa", columns="L_D", values=c).to_string() if c == "status"
                  else df.pivot(index="kappa", columns="L_D", values=c).round(2))
