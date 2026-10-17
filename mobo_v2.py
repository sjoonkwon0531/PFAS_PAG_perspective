"""
v2 multi-objective Bayesian optimization (ParEGO) at 12 nm half-pitch with the conservative solver.

Changes from v1 (internal review E5):
  * equal evaluation budget for both chemistries (14 initial + 2 x 21 iterations = 56 evaluations)
  * two independent starts per chemistry (Latin hypercube seeds 7 and 11), pooled afterwards
  * guarded sizing (sizing.size_at) with explicit failure reasons
  * re-evaluation (mobo_v2.py reeval) selects candidates by three-objective non-dominated sorting
    (dose, LWR, film fluorine), keeps the first two fronts, and re-simulates them with six seeds
    never used during the search (101-106) and a three-dose ladder
  * premium uncertainty by parametric bootstrap over the re-evaluated means and standard errors
Usage:
  python mobo_v2.py search allowed 7 ; python mobo_v2.py search allowed 11
  python mobo_v2.py search free 7    ; python mobo_v2.py search free 11
  python mobo_v2.py reeval ; python mobo_v2.py premium
Design variables and mappings: Table 3 (kappa = kappa_acid(f_F) * s_PG is the relative effective
deprotection activity).
"""
import os, sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from car_stochastic import Params, latent_image_metrics
from sizing import size_at

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
KP0, HP, THK = 0.12725, 12, 22
LB = np.array([0.0, 0.0, 1.0, 3.0, 0.20])
UB = np.array([1.0, 3.0, 3.0, 10.0, 0.40])
NAMES = ["f_F", "n_I", "s_PG", "L_D", "c_PAG"]
F_FILM_PER_MMOLG = 117.48 / 0.27 * (THK / 30)      # ug film F per wafer-layer per (mmol/g) at f_F = 1
EL_MIN, MISS_MAX = 12.0, 0.05
KAPPA_ARYL = 0.3
N_INIT, N_ITER = 14, 21


def formulation(x):
    fF, nI, sPG, LD, c = x
    alpha = 4.43 + (4.36 + 0.38 * fF + 0.303 * nI - 4.43) * (c / 0.27)
    kappa = (KAPPA_ARYL + (1 - KAPPA_ARYL) * fF) * sPG
    return dict(alpha_um=alpha, kappa=kappa, L_D=LD, pag_mmol_g=c), F_FILM_PER_MMOLG * c * fF


def evaluate(x, seeds=(1, 2), factors=(0.9, 1.1)):
    over, F = formulation(x)
    p = Params(hp=HP, thickness=THK, kp=KP0, **over)
    r = size_at(p, factors=factors, seeds=seeds)
    lim = latent_image_metrics(p) if r["status"] == "ok" else dict(EL=np.nan, LILS=np.nan)
    rec = dict(zip(NAMES, x)); rec.update(over); rec.update(r); rec.update(lim); rec["F_ug"] = F
    rec["feasible"] = bool(r["status"] == "ok" and lim["EL"] >= EL_MIN and r["missing"] < MISS_MAX)
    return rec


def violation(df):
    v = np.where(df.status.values == "ok", 0.0, 3.0)
    el = np.nan_to_num(df.EL.values.astype(float), nan=0.0)
    ms = np.nan_to_num(df.missing.values.astype(float), nan=1.0)
    return v + np.maximum(0, EL_MIN - el) / EL_MIN + 5 * np.maximum(0, ms - MISS_MAX)


def parego_step(U, df, rng, dim_mask, n_cand=6000):
    Y = df[["LWR", "dose", "F_ug"]].values.astype(float)
    good = df.feasible.values & np.all(np.isfinite(Y), axis=1)
    if good.sum() == 0:
        s = violation(df)
    else:
        lo, hi = Y[good].min(0), Y[good].max(0)
        Yn = (Y - lo) / np.where(hi > lo, hi - lo, 1)
        Yn[~good] = 1.5
        w = rng.dirichlet(np.ones(3))
        s = np.max(w * Yn, axis=1) + 0.05 * np.sum(w * Yn, axis=1)
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(U.shape[1]) * 0.3, nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=0)
    gp.fit(U, s)
    C = rng.random((n_cand, U.shape[1]))
    best = U[np.argmin(s)]
    C[: n_cand // 3] = np.clip(best + 0.08 * rng.standard_normal((n_cand // 3, U.shape[1])), 0, 1)
    C[:, ~dim_mask] = 0.0
    mu, sd = gp.predict(C, return_std=True)
    imp = s.min() - mu
    z = imp / np.maximum(sd, 1e-9)
    return C[np.argmax(imp * norm.cdf(z) + sd * norm.pdf(z))]


def search(chem, start):
    lb, ub = LB.copy(), UB.copy()
    if chem == "free":
        ub[0] = 0.0
    dim_mask = ub > lb
    decode = lambda u: lb + u * (ub - lb)
    rng = np.random.default_rng(start)
    U = qmc.LatinHypercube(d=5, seed=start).random(N_INIT)
    U[:, ~dim_mask] = 0.0
    anchors = np.array([[1.0, 0.0, 1.0, 6.0, 0.27], [0.0, 0.0, 1.0, 6.0, 0.27]])
    A = np.clip((anchors - lb) / np.where(dim_mask, ub - lb, 1), 0, 1)
    U = np.vstack([A[1:] if chem == "free" else A, U])[:N_INIT]
    fn = os.path.join(OUT, f"mobo_v2_{chem}_s{start}.csv")
    with Pool(2) as pool:
        df = pd.DataFrame(pool.map(evaluate, [decode(u) for u in U])); df["iter"] = 0
    for it in range(1, N_ITER + 1):
        newU = np.array([parego_step(U, df, rng, dim_mask) for _ in range(2)])
        with Pool(2) as pool:
            d2 = pd.DataFrame(pool.map(evaluate, [decode(u) for u in newU])); d2["iter"] = it
        df = pd.concat([df, d2], ignore_index=True); U = np.vstack([U, newU])
        df.to_csv(fn, index=False)
        fb = df[df.feasible]
        print(f"{chem} s{start} iter {it:2d} n={len(df)} feasible={len(fb)} "
              f"minLWR={fb.LWR.min() if len(fb) else np.nan:.2f} minDose={fb.dose.min() if len(fb) else np.nan:.1f}",
              flush=True)
    df["start"] = start; df["chem"] = chem
    df.to_csv(fn, index=False)


def nondominated_sort(Y):
    n = len(Y); rank = np.full(n, -1); r = 0; remaining = np.arange(n)
    while len(remaining):
        sub = Y[remaining]
        nd = [i for i in range(len(sub)) if not np.any(np.all(sub <= sub[i], 1) & np.any(sub < sub[i], 1))]
        rank[remaining[nd]] = r; remaining = np.delete(remaining, nd); r += 1
    return rank


def reeval_job(args):
    chem, x = args
    rec = evaluate(np.asarray(x, float), seeds=(101, 102, 103, 104, 105, 106), factors=(0.85, 1.0, 1.15))
    rec["chem"] = chem
    return rec


def reeval(max_per_chem=12):
    jobs = [("baseline", [1.0, 0.0, 1.0, 6.0, 0.27])]
    for chem in ("allowed", "free"):
        parts = [pd.read_csv(os.path.join(OUT, f"mobo_v2_{chem}_s{s}.csv")) for s in (7, 11)
                 if os.path.exists(os.path.join(OUT, f"mobo_v2_{chem}_s{s}.csv"))]
        d = pd.concat(parts, ignore_index=True)
        f = d[d.feasible].copy()
        f["rank"] = nondominated_sort(f[["dose", "LWR", "F_ug"]].values.astype(float))
        sel = f[f["rank"] <= 1].sort_values(["rank", "LWR"]).head(max_per_chem)
        jobs += [(chem, r[NAMES].values.astype(float).tolist()) for _, r in sel.iterrows()]
    with Pool(2) as pool:
        res = pd.DataFrame(pool.map(reeval_job, jobs))
    res.to_csv(os.path.join(OUT, "mobo_v2_reeval.csv"), index=False)
    pd.set_option("display.width", 250)
    print(res[["chem"] + NAMES + ["kappa", "F_ug", "status", "dose", "dose_se", "LWR", "LWR_se", "EL", "missing", "feasible"]]
          .round(3).to_string())


def reeval_extra(max_per_chem=8):
    """Add designs on the two-objective (dose, LWR) search front that were not selected by the three-objective
    sort (the low-dose, high-fluorine end), so that the re-evaluated fronts cover loose LWR targets."""
    fn = os.path.join(OUT, "mobo_v2_reeval.csv")
    R0 = pd.read_csv(fn)
    done = R0[NAMES].values.astype(float)
    jobs = []
    for chem in ("allowed", "free"):
        d = pd.concat([pd.read_csv(os.path.join(OUT, f"mobo_v2_{chem}_s{s}.csv")) for s in (7, 11)], ignore_index=True)
        f = front2(d[d.feasible])
        new = [r for _, r in f.iterrows() if not np.any(np.all(np.isclose(done, r[NAMES].values.astype(float), atol=1e-6), 1))]
        jobs += [(chem, r[NAMES].values.astype(float).tolist()) for r in new[:max_per_chem]]
    print("extra designs:", len(jobs), flush=True)
    if jobs:
        with Pool(2) as pool:
            res = pd.DataFrame(pool.map(reeval_job, jobs))
        res["selection"] = "2-objective front"
        R0["selection"] = np.where(R0.chem == "baseline", "baseline", "3-objective rank <= 1")
        pd.concat([R0, res], ignore_index=True).to_csv(fn, index=False)
        print(res[["chem"] + NAMES + ["F_ug", "dose", "dose_se", "LWR", "LWR_se", "EL", "missing", "feasible"]].round(3).to_string())


def reeval_emu(targets_allowed=(4.6, 4.75, 4.85, 5.0, 5.3), targets_free=(4.6, 4.75, 5.0)):
    """Directly re-evaluate the lowest-dose emulator front designs (reference assumptions, c_PAG = 0.40) at several
    predicted LWR levels, so that candidates missed by the search are tested with the same protocol."""
    fn = os.path.join(OUT, "mobo_v2_reeval.csv")
    R0 = pd.read_csv(fn)
    F = pd.read_csv(os.path.join(OUT, "emulator_v2_fronts.csv"))
    F = F[F.case == "ref"]
    jobs = []
    for camp, chem, targets in (("fluorine-allowed", "allowed", targets_allowed), ("fluorine-free", "free", targets_free)):
        f = F[F.campaign == camp]
        for t in targets:
            g = f[f.LWR <= t]
            if len(g):
                r = g.sort_values("dose").iloc[0]
                x = [r.f_F, r.n_I, r.s_PG, r.L_D, 0.40]
                if not any(np.allclose(x, j[1]) for j in jobs):
                    jobs.append((chem, x))
    print("emulator-proposed designs:", len(jobs), flush=True)
    with Pool(2) as pool:
        res = pd.DataFrame(pool.map(reeval_job, jobs))
    res["selection"] = "emulator front"
    pd.concat([R0, res], ignore_index=True).to_csv(fn, index=False)
    print(res[["chem"] + NAMES + ["kappa", "alpha_um", "F_ug", "dose", "dose_se", "LWR", "LWR_se", "EL", "missing", "feasible"]].round(3).to_string())


def front2(df):
    df = df.sort_values(["dose", "LWR"]); best, keep = np.inf, []
    for i, r in df.iterrows():
        if r.LWR < best - 1e-12:
            keep.append(i); best = r.LWR
    return df.loc[keep]


def premium(targets=(4.4, 4.5, 4.75, 5.0, 5.5, 6.0), nboot=4000):
    R = pd.read_csv(os.path.join(OUT, "mobo_v2_reeval.csv"))
    R = R[R.feasible]
    # f_F = 0 lies inside the fluorine-allowed design space, so the attainable fluorine-allowed front is the
    # non-dominated set of both campaigns; the fluorine-free front uses the fluorine-free designs only.
    A, B = R[R.chem.isin(["allowed", "free"])], R[R.chem == "free"]
    rng = np.random.default_rng(0)

    def attain(df, t):
        m = df.LWR.values <= t
        return df.dose.values[m].min() if m.any() else np.nan

    rows = []
    for t in targets:
        da, db = attain(A, t), attain(B, t)
        boots = []
        for _ in range(nboot):
            # one draw per design, shared by both fronts (fluorine-free designs belong to both sets)
            a = A.assign(dose=rng.normal(A.dose, A.dose_se), LWR=rng.normal(A.LWR, A.LWR_se))
            b = a[a.chem == "free"]
            x, y = attain(a, t), attain(b, t)
            boots.append(100 * (y / x - 1) if np.isfinite(x) and np.isfinite(y) else np.nan)
        boots = np.array(boots)
        frac = np.isfinite(boots).mean()
        rows.append(dict(LWR_target=t, dose_allowed=da, dose_free=db,
                         premium=100 * (db / da - 1) if np.isfinite(da) and np.isfinite(db) else np.nan,
                         p05=np.nanpercentile(boots, 5) if frac > 0.5 else np.nan,
                         p95=np.nanpercentile(boots, 95) if frac > 0.5 else np.nan, frac_attained=frac))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(OUT, "mobo_v2_premium.csv"), index=False)
    print(P.round(2).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "search":
        search(sys.argv[2], int(sys.argv[3]))
    elif cmd == "reeval":
        reeval()
    elif cmd == "reeval_emu":
        reeval_emu()
    elif cmd == "reeval_extra":
        reeval_extra()
    elif cmd == "premium":
        premium()
