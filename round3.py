"""
Round-3 review checks.

  python round3.py h55    additional direct re-evaluation of family-H candidates near LWR 5.5 nm (P0-2)
  python round3.py geo    model-form sensitivity of family B to the mixing rule for acid activity (P1-1):
                          geometric instead of linear (concentration-weighted) mixing, emulator fronts and
                          direct re-evaluation of the premium-setting candidates
  python round3.py summary
"""
import os, sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
import families as fam
import emulator_v2 as emu
from round2 import emu_designs, front, reeval_job, premiums, FAMS, OUT

T = pd.read_csv(os.path.join(OUT, "emulator_v2_table.csv"))
I = emu.build_interp(T, emu.ALPHAS, emu.KAPPAS, emu.LDS)


def cmd_h55(n=12, lwr_max=5.30):
    R = pd.read_csv(os.path.join(OUT, "round2_reeval.csv"))
    d_front = front(R[R.in_H])
    d55 = d_front[d_front.LWR <= 5.5].dose.min()
    E = emu_designs(I, "H")
    E = E[E.feasible & (E.LWR <= lwr_max) & (E.dose < d55)].sort_values("dose")
    # spread the picks: at most four per diffusion-length node, distinct compositions
    picks, seen = [], set()
    for _, r in E.iterrows():
        key = (round(r.F_eq, 3), round(r.n_I, 3), round(r.kappa_acid, 3), r.L_D)
        if key in seen or sum(1 for p in picks if p.L_D == r.L_D) >= 4:
            continue
        seen.add(key); picks.append(r)
        if len(picks) >= n:
            break
    jobs = [((r.F_eq, r.n_I, r.kappa_acid, r.s_PG, r.L_D, r.c_PAG), f"round3 H near 5.5 (pred {r.LWR:.2f})", 240) for r in picks]
    print("H candidates:", len(jobs), "current H dose at 5.5:", round(d55, 2), flush=True)
    with Pool(2) as pool:
        D = pd.DataFrame(pool.map(reeval_job, jobs))
    D["dose_pred"] = [r.dose for r in picks]; D["LWR_pred"] = [r.LWR for r in picks]
    D.to_csv(os.path.join(OUT, "round3_h55.csv"), index=False)
    print(D[["origin", "F_eq", "n_I", "kappa_acid", "s_PG", "L_D", "dose_pred", "LWR_pred", "dose", "LWR", "LWR_se", "feasible", "in_H"]]
          .round(3).to_string())


def mix_geo(xN, xT):
    """Geometric (log-linear) mixing of the acid activity of a nonaflate / aryl sulfonate blend."""
    F_eq, nI, _ = fam.mix(xN=xN, xT=xT, xS=max(0.0, 1 - xN - xT))
    return F_eq, nI, fam.KAPPA_ARYL ** (1 - xN)


def emu_geo(c=0.40, step=0.05):
    g = np.round(np.arange(0, 1 + 1e-9, step), 4)
    comps = sorted({tuple(round(v, 6) for v in mix_geo(xN, xT)) for xN in g for xT in g if xN + xT <= 1 + 1e-9})
    sPG = np.linspace(1, 3, 21); rows = []
    for L in emu.LDS:
        for (F_eq, nI, ka) in comps:
            over = [fam.params(F_eq, nI, ka, s, L, c) for s in sPG]
            a = np.array([o[0]["alpha_um"] for o in over]); k = np.array([o[0]["kappa"] for o in over])
            pts = np.c_[a, np.log(k)]
            rows.append(pd.DataFrame(dict(F_eq=F_eq, n_I=nI, kappa_acid=ka, s_PG=sPG, L_D=L, c_PAG=c, alpha_um=a, kappa=k,
                                          F_ug=[o[1] for o in over], dose=np.exp(I[(L, "dose")](pts)), LWR=I[(L, "LWR")](pts),
                                          EL=I[(L, "EL")](pts), missing=I[(L, "missing")](pts))))
    df = pd.concat(rows, ignore_index=True)
    df["feasible"] = np.isfinite(df.dose) & np.isfinite(df.LWR) & (df.EL >= 12) & (df.missing < 0.05)
    return df


def cmd_geo():
    G = front(emu_geo()); Fr = front(emu_designs(I, "free"))
    rows = []
    for t in (4.5, 4.75, 5.0, 5.5):
        db = Fr[Fr.LWR <= t].dose.min(); da = G[G.LWR <= t].dose.min()
        rows.append(dict(LWR_target=t, emu_premium_Bgeo=100 * (db / da - 1)))
    print(pd.DataFrame(rows).round(1).to_string())
    picks = []
    for t in (4.85, 5.0, 5.25, 5.5):
        g = G[G.LWR <= t]
        if len(g):
            picks.append(g.sort_values("dose").iloc[0])
    P = pd.DataFrame(picks).drop_duplicates(subset=["F_eq", "n_I", "kappa_acid", "s_PG", "L_D"])
    jobs = [((r.F_eq, r.n_I, r.kappa_acid, r.s_PG, r.L_D, r.c_PAG), f"round3 B geometric (pred {r.LWR:.2f})", 240) for _, r in P.iterrows()]
    with Pool(2) as pool:
        D = pd.DataFrame(pool.map(reeval_job, jobs))
    D["dose_pred"] = P.dose.values; D["LWR_pred"] = P.LWR.values
    D.to_csv(os.path.join(OUT, "round3_geo.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "round3_geo_emu.csv"), index=False)
    print(D[["origin", "F_eq", "n_I", "kappa_acid", "s_PG", "L_D", "dose_pred", "LWR_pred", "dose", "LWR", "feasible"]].round(3).to_string())


def cmd_summary():
    R = pd.read_csv(os.path.join(OUT, "round2_reeval.csv"))
    H = pd.read_csv(os.path.join(OUT, "round3_h55.csv")) if os.path.exists(os.path.join(OUT, "round3_h55.csv")) else None
    if H is not None:
        RR = pd.concat([R, H], ignore_index=True)
        feas = RR.feasible.values.astype(bool); mem = {f: RR[f"in_{f}"].values.astype(bool) for f in FAMS}
        p = premiums(RR.dose.values, RR.LWR.values, feas, mem, (4.5, 4.75, 5.0, 5.5))
        print("with round-3 H candidates:", {k: round(v, 1) for k, v in p.items() if k[0] == "H"})
        RR.to_csv(os.path.join(OUT, "round3_reeval_all.csv"), index=False)
    Gd = pd.read_csv(os.path.join(OUT, "round3_geo.csv")) if os.path.exists(os.path.join(OUT, "round3_geo.csv")) else None
    if Gd is not None:
        free = R[R.feasible & R.in_free]
        for t in (5.0, 5.5):
            db = free[free.LWR <= t].dose.min(); g = Gd[Gd.feasible & (Gd.LWR <= t)]
            print("B geometric, direct", t, round(100 * (db / g.dose.min() - 1), 1) if len(g) else None)


if __name__ == "__main__":
    {"h55": cmd_h55, "geo": cmd_geo, "summary": cmd_summary}[sys.argv[1]]()
