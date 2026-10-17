"""
v2 resolution analysis (Fig. 5): shrinking-pitch stress test with a fixed sinusoidal aerial image.
NA labels are proxies only (review E4): optics are not modelled; pitch and film thickness change.

(a) Mean-field EL vs half-pitch for several L_D (results/el_vs_hp_v2.csv)
(b) Stochastic sizing of four resists at hp = 16, 12, 10 nm, six seeds (results/resolution_scenarios_v2.csv)
(c) Image-visibility sensitivity at hp = 12 nm, V = 0.70, 0.85, 1.00, four seeds (results/visibility_v2.csv)
(d) Time-step check of the key comparison: S0, S2, S5 at 16 and 12 nm with 960 steps (results/dtcheck_v2.csv)
"""
import os, sys, itertools
import numpy as np
import pandas as pd
from multiprocessing import Pool
from car_stochastic import Params, MORParams, latent_image_metrics
from sizing import size_at

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
KP0, Y_SN = 0.12725, 5.5
THK = {16: (30, 20), 12: (22, 15), 10: (18, 12)}
RES = {"S0 Fluorinated baseline": ("CAR", dict(alpha_um=4.74, kappa=1.0, L_D=6.0)),
       "S2 Idealized bake-equivalent": ("CAR", dict(alpha_um=4.36, kappa=0.999, L_D=10.95)),
       "S5 Co-designed PFAS-free": ("CAR", dict(alpha_um=5.27, kappa=1.0, L_D=4.0)),
       "MOR tin-oxo cage (illustrative)": ("MOR", dict())}


def mk(kind, hp, over, **extra):
    if kind == "CAR":
        return Params(hp=hp, kp=KP0, thickness=THK[hp][0], **over, **extra)
    return MORParams(hp=hp, eta=Y_SN, thickness=THK[hp][1], **over, **extra)


def el_job(args):
    hp, LD = args
    p = Params(hp=hp, L_D=LD, kp=KP0, thickness=THK[hp][0] if hp in THK else 25, Lx=1)
    return dict(hp=hp, L_D=LD, **latent_image_metrics(p))


def res_job(args):
    label, hp = args
    kind, over = RES[label]
    r = size_at(mk(kind, hp, over), kind=kind, seeds=(1, 2, 3, 4, 5, 6))
    return dict(label=label, kind=kind, hp=hp, LWR_over_CD=r["LWR"] / hp, LWR_over_CD_se=r["LWR_se"] / hp, **r)


def vis_job(args):
    label, V = args
    kind, over = RES[label]
    r = size_at(mk(kind, 12, over, modulation=V), kind=kind, seeds=(1, 2, 3, 4))
    return dict(label=label, visibility=V, **r)


def dt_job(args):
    label, hp, n = args
    kind, over = RES[label]
    r = size_at(mk(kind, hp, over, n_steps=n), kind=kind, seeds=(1, 2, 3))
    return dict(label=label, hp=hp, n_steps=n, **r)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 250)
    if which in ("el", "all"):
        with Pool(2) as pool:
            el = pd.DataFrame(pool.map(el_job, list(itertools.product([8, 10, 12, 14, 16, 20, 24], [3.0, 4.0, 6.0, 8.0, 11.0]))))
        el.to_csv(os.path.join(OUT, "el_vs_hp_v2.csv"), index=False)
        print(el.pivot(index="hp", columns="L_D", values="EL").round(1), flush=True)
    if which in ("res", "all"):
        with Pool(2) as pool:
            st = pd.DataFrame(pool.map(res_job, list(itertools.product(RES, (16, 12, 10)))))
        st.to_csv(os.path.join(OUT, "resolution_scenarios_v2.csv"), index=False)
        print(st[["label", "hp", "status", "dose", "LWR", "LWR_se", "LWR_over_CD", "missing", "missing_se"]].round(3).to_string(), flush=True)
    if which in ("vis", "all"):
        jobs = list(itertools.product(["S0 Fluorinated baseline", "S2 Idealized bake-equivalent", "S5 Co-designed PFAS-free"],
                                      (0.70, 0.85, 1.00)))
        with Pool(2) as pool:
            vs = pd.DataFrame(pool.map(vis_job, jobs))
        vs.to_csv(os.path.join(OUT, "visibility_v2.csv"), index=False)
        print(vs[["label", "visibility", "status", "dose", "LWR", "LWR_se", "missing"]].round(3).to_string(), flush=True)
    if which in ("dt", "all"):
        jobs = [(l, hp, n) for l in ("S0 Fluorinated baseline", "S2 Idealized bake-equivalent", "S5 Co-designed PFAS-free")
                for hp in (16, 12) for n in (960,)]
        with Pool(2) as pool:
            dt = pd.DataFrame(pool.map(dt_job, jobs))
        dt.to_csv(os.path.join(OUT, "dtcheck_v2.csv"), index=False)
        print(dt[["label", "hp", "n_steps", "status", "dose", "LWR", "LWR_se", "missing"]].round(3).to_string(), flush=True)
