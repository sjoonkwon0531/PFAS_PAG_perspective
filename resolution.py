"""
Resolution axis of the RLS trade-off (0.33 NA at 16 nm half-pitch; 0.55 NA High-NA at 12 and 10 nm).

(a) Mean-field exposure latitude EL(hp) for CAR at several acid diffusion lengths L_D.
(b) Stochastic dose-to-size and LWR for named scenarios and a tin-oxo MOR at hp = 16, 12, 10 nm.
Image modulation is held at 0.85 for every case (k1 = 0.39 at 0.33 NA/16 nm, 0.49 at 0.55 NA/12 nm,
0.41 at 0.55 NA/10 nm), so differences come from the resist, not the optics.
"""
import os, itertools
import numpy as np
import pandas as pd
from multiprocessing import Pool
from car_stochastic import (Params, MORParams, simulate, simulate_mor, metrics, mor_as_params,
                            dose_to_size_meanfield, mor_dose_to_size, latent_image_metrics)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
KP0 = 0.12725
ETA0 = 5.5
SEEDS = (1, 2, 3)
THK = {16: (30, 20), 12: (22, 15), 10: (18, 12)}      # (CAR, MOR) film thickness, nm


def el_job(args):
    hp, LD = args
    p = Params(hp=hp, L_D=LD, kp=KP0, thickness=THK[hp][0] if hp in THK else 25, Lx=1)
    r = latent_image_metrics(p)
    return dict(hp=hp, L_D=LD, **r)


def stoch_at_size(spec):
    kind, label, hp, over = spec
    if kind == "CAR":
        p = Params(hp=hp, kp=KP0, thickness=THK[hp][0], **over)
        d0 = dose_to_size_meanfield(p); sim = simulate; q = p
    else:
        p = MORParams(hp=hp, eta=ETA0, thickness=THK[hp][1], **over)
        d0 = mor_dose_to_size(p); sim = simulate_mor; q = mor_as_params(p)
    rows = []
    for f in (0.85, 1.0, 1.15):
        acc = []
        for s in SEEDS:
            p.seed = s
            acc.append(metrics(sim(p, d0 * f)["m"], q))
        rows.append(dict(dose=d0 * f, **{k: np.nanmean([a[k] for a in acc]) for k in acc[0]}))
    df = pd.DataFrame(rows).dropna().sort_values("dose")
    sgn = 1 if kind == "MOR" else -1
    cd, dose = df.CD.values, df.dose.values
    try:
        if kind == "CAR":
            ds = float(np.interp(hp, cd[::-1], dose[::-1]))
        else:
            ds = float(np.interp(hp, cd, dose))
        lwr = float(np.interp(ds, dose, df.LWR.values))
        ler = float(np.interp(ds, dose, df.LER.values))
        miss = float(np.interp(ds, dose, df.missing.values))
    except Exception:
        ds = lwr = ler = miss = np.nan
    return dict(kind=kind, label=label, hp=hp, dose_size=ds, LWR=lwr, LER=ler, missing=miss,
                LWR_over_CD=lwr / hp, **over)


SCEN = [
    ("CAR", "S0 Fluorinated baseline", dict(alpha_um=4.74, kappa=1.0, L_D=6.0)),
    ("CAR", "S2 PFAS-free, PEB-compensated", dict(alpha_um=4.36, kappa=0.999, L_D=10.95)),
    ("CAR", "S5 PFAS-free, co-designed", dict(alpha_um=5.27, kappa=1.0, L_D=4.0)),
    ("MOR", "MOR tin-oxo cage", dict()),
]

if __name__ == "__main__":
    with Pool(2) as pool:
        el = pd.DataFrame(pool.map(el_job, list(itertools.product(
            [8, 10, 12, 14, 16, 20, 24], [3.0, 4.0, 6.0, 8.0, 11.0]))))
    el.to_csv(os.path.join(OUT, "el_vs_hp.csv"), index=False)
    print(el.pivot(index="hp", columns="L_D", values="EL").round(1))
    specs = [(k, lab, hp, o) for hp in (16, 12, 10) for (k, lab, o) in SCEN]
    with Pool(2) as pool:
        st = pd.DataFrame(pool.map(stoch_at_size, specs))
    st.to_csv(os.path.join(OUT, "resolution_scenarios.csv"), index=False)
    print(st[["label", "hp", "dose_size", "LWR", "LWR_over_CD", "missing"]].round(2).to_string())
