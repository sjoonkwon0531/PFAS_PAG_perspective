"""
Round-2 review analyses (P0 items).

  python round2.py emu        emulator fronts and premiums for the formulation families (free, B, H, U) at
                              c_PAG = 0.40; selects emulator front candidates for direct re-evaluation
  python round2.py search 7   constrained ParEGO campaign for family B (blend; same budget and settings as v2)
  python round2.py search 11
  python round2.py reeval     direct re-evaluation (seeds 101-106, three-dose ladder, dt = 0.25 s) of all
                              candidates: the 38 v3 designs, family-B search candidates and emulator candidates,
                              with per-seed records for the paired bootstrap and sizing diagnostics
  python round2.py dt         re-evaluation of the premium-setting front designs with dt = 0.0625 s (960 steps)
  python round2.py premium    family premiums with parametric (independent) and paired (field-level) bootstraps
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
import families as fam
import emulator_v2 as emu

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
KP0, HP = 0.12725, 12
EL_MIN, MISS_MAX = 12.0, 0.05
SEEDS_RE = (101, 102, 103, 104, 105, 106)
TARGETS = (4.5, 4.75, 5.0, 5.5)
FAMS = ("free", "B", "H", "U")


# ----------------------------------------------------------------------------- evaluation
def evaluate_comp(F_eq, nI, kacid, sPG, LD, c, seeds=(1, 2), factors=(0.9, 1.1), n_steps=240):
    over, F = fam.params(F_eq, nI, kacid, sPG, LD, c)
    p = Params(hp=HP, thickness=fam.THK, kp=KP0, n_steps=n_steps, **over)
    r = size_at(p, factors=factors, seeds=seeds)
    lim = latent_image_metrics(p) if r["status"] == "ok" else dict(EL=np.nan, LILS=np.nan)
    rec = dict(F_eq=F_eq, n_I=nI, kappa_acid=kacid, s_PG=sPG, L_D=LD, c_PAG=c, F_ug=F)
    rec.update(over); rec.update(r); rec.update(lim)
    rec["feasible"] = bool(r["status"] == "ok" and lim["EL"] >= EL_MIN and r["missing"] < MISS_MAX)
    for f in FAMS:
        rec[f"in_{f}"] = fam.member(f, F_eq, nI, kacid)
    return rec


def reeval_job(args):
    comp, origin, n_steps = args
    rec = evaluate_comp(*comp, seeds=SEEDS_RE, factors=(0.85, 1.0, 1.15), n_steps=n_steps)
    rec["origin"] = origin
    return rec


def front(df, key="dose"):
    d = df[df.feasible].assign(_d=lambda x: x[key].round(1)).sort_values(["_d", "LWR"])
    best, keep = np.inf, []
    for i, r in d.iterrows():
        if r.LWR < best - 1e-12:
            keep.append(i); best = r.LWR
    return d.loc[keep]


# ----------------------------------------------------------------------------- emulator
def emu_designs(I, fam_name, sPG_max=3.0, c=0.40):
    comps = fam.enumerate_family(fam_name)
    sPG = np.linspace(1, sPG_max, 21)
    rows = []
    for L in emu.LDS:
        for (F_eq, nI, ka) in comps:
            over = [fam.params(F_eq, nI, ka, s, L, c) for s in sPG]
            a = np.array([o[0]["alpha_um"] for o in over]); k = np.array([o[0]["kappa"] for o in over])
            pts = np.c_[a, np.log(k)]
            d = np.exp(I[(L, "dose")](pts)); lwr = I[(L, "LWR")](pts)
            el = I[(L, "EL")](pts); ms = I[(L, "missing")](pts)
            rows.append(pd.DataFrame(dict(F_eq=F_eq, n_I=nI, kappa_acid=ka, s_PG=sPG, L_D=L, c_PAG=c, alpha_um=a,
                                          kappa=k, F_ug=[o[1] for o in over], dose=d, LWR=lwr, EL=el, missing=ms)))
    df = pd.concat(rows, ignore_index=True)
    df["feasible"] = np.isfinite(df.dose) & np.isfinite(df.LWR) & (df.EL >= EL_MIN) & (df.missing < MISS_MAX)
    return df


def premium_table(fronts, targets=TARGETS):
    rows = []
    free = fronts["free"]
    for t in targets:
        db = free[free.LWR <= t].dose.min() if (free.LWR <= t).any() else np.nan
        r = dict(LWR_target=t, dose_free=db)
        for f in ("B", "H", "U"):
            A = fronts[f]
            da = A[A.LWR <= t].dose.min() if (A.LWR <= t).any() else np.nan
            r[f"dose_{f}"] = da
            r[f"premium_{f}"] = 100 * (db / da - 1) if np.isfinite(da) and np.isfinite(db) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def cmd_emu():
    T = pd.read_csv(os.path.join(OUT, "emulator_v2_table.csv"))
    I = emu.build_interp(T, emu.ALPHAS, emu.KAPPAS, emu.LDS)
    allf, fronts = [], {}
    for f in FAMS:
        df = emu_designs(I, f); df["family"] = f
        fr = front(df); fronts[f] = fr
        allf.append(fr)
    pd.concat(allf, ignore_index=True).to_csv(os.path.join(OUT, "round2_emu_fronts.csv"), index=False)
    P = premium_table(fronts, targets=(4.5, 4.6, 4.75, 5.0, 5.25, 5.5))
    P.to_csv(os.path.join(OUT, "round2_emu_premium.csv"), index=False)
    print(P.round(1).to_string())
    # candidates: lowest predicted dose at predicted LWR levels (families B and H; free as control)
    cand = []
    for f, levels in (("B", (4.6, 4.75, 4.85, 5.0, 5.25, 5.5, 5.8)), ("H", (4.6, 4.75, 4.85, 5.0, 5.25, 5.5, 5.8)),
                      ("free", (4.6, 4.75, 5.0))):
        fr = fronts[f]
        for t in levels:
            g = fr[fr.LWR <= t]
            if len(g):
                r = g.sort_values("dose").iloc[0]
                cand.append(dict(family=f, level=t, F_eq=r.F_eq, n_I=r.n_I, kappa_acid=r.kappa_acid, s_PG=r.s_PG,
                                 L_D=r.L_D, c_PAG=r.c_PAG, dose_pred=r.dose, LWR_pred=r.LWR))
    C = pd.DataFrame(cand).drop_duplicates(subset=["F_eq", "n_I", "kappa_acid", "s_PG", "L_D", "c_PAG"])
    C.to_csv(os.path.join(OUT, "round2_emu_candidates.csv"), index=False)
    print(C.round(3).to_string())


# ----------------------------------------------------------------------------- constrained search (family B)
LB = np.array([0.0, 0.0, 1.0, 3.0, 0.20]); UB = np.array([1.0, 1.0, 3.0, 10.0, 0.40])
VN = ["x_N", "u_T", "s_PG", "L_D", "c_PAG"]          # x_T = (1 - x_N) u_T, x_S = rest


def decode_B(x):
    xN, uT, sPG, LD, c = x
    xT = (1 - xN) * uT
    return fam.mix(xN=xN, xT=xT, xS=max(0.0, 1 - xN - xT)) + (sPG, LD, c)


def eval_B(x):
    comp = decode_B(x)
    rec = evaluate_comp(*comp)
    rec.update(dict(zip(VN, x)))
    return rec


def parego_step(U, df, rng, n_cand=6000):
    Y = df[["LWR", "dose", "F_ug"]].values.astype(float)
    good = df.feasible.values & np.all(np.isfinite(Y), axis=1)
    if good.sum() == 0:
        v = np.where(df.status.values == "ok", 0.0, 3.0)
        s = v + np.maximum(0, EL_MIN - np.nan_to_num(df.EL.values.astype(float))) / EL_MIN
    else:
        lo, hi = Y[good].min(0), Y[good].max(0)
        Yn = (Y - lo) / np.where(hi > lo, hi - lo, 1); Yn[~good] = 1.5
        w = rng.dirichlet(np.ones(3))
        s = np.max(w * Yn, axis=1) + 0.05 * np.sum(w * Yn, axis=1)
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(U.shape[1]) * 0.3, nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=0).fit(U, s)
    C = rng.random((n_cand, U.shape[1])); best = U[np.argmin(s)]
    C[: n_cand // 3] = np.clip(best + 0.08 * rng.standard_normal((n_cand // 3, U.shape[1])), 0, 1)
    mu, sd = gp.predict(C, return_std=True); imp = s.min() - mu; z = imp / np.maximum(sd, 1e-9)
    return C[np.argmax(imp * norm.cdf(z) + sd * norm.pdf(z))]


def cmd_search(start, n_init=14, n_iter=21):
    decode = lambda u: LB + u * (UB - LB)
    rng = np.random.default_rng(start)
    U = qmc.LatinHypercube(d=5, seed=start).random(n_init)
    anchor = np.clip((np.array([[1.0, 0.0, 1.0, 6.0, 0.27]]) - LB) / (UB - LB), 0, 1)
    U = np.vstack([anchor, U])[:n_init]
    fn = os.path.join(OUT, f"round2_searchB_s{start}.csv")
    with Pool(2) as pool:
        df = pd.DataFrame(pool.map(eval_B, [decode(u) for u in U])); df["iter"] = 0
    for it in range(1, n_iter + 1):
        newU = np.array([parego_step(U, df, rng) for _ in range(2)])
        with Pool(2) as pool:
            d2 = pd.DataFrame(pool.map(eval_B, [decode(u) for u in newU])); d2["iter"] = it
        df = pd.concat([df, d2], ignore_index=True); U = np.vstack([U, newU]); df.to_csv(fn, index=False)
        fb = df[df.feasible]
        print(f"B s{start} iter {it:2d} n={len(df)} feasible={len(fb)} minLWR={fb.LWR.min() if len(fb) else np.nan:.2f} "
              f"minDose={fb.dose.min() if len(fb) else np.nan:.1f}", flush=True)
    df["start"] = start; df.to_csv(fn, index=False)


def nondominated_sort(Y):
    n = len(Y); rank = np.full(n, -1); r = 0; rem = np.arange(n)
    while len(rem):
        sub = Y[rem]
        nd = [i for i in range(len(sub)) if not np.any(np.all(sub <= sub[i], 1) & np.any(sub < sub[i], 1))]
        rank[rem[nd]] = r; rem = np.delete(rem, nd); r += 1
    return rank


# ----------------------------------------------------------------------------- re-evaluation
def comp_key(c):
    return tuple(np.round(c, 5))


def cmd_reeval():
    jobs, seen = [], set()

    def add(comp, origin):
        k = comp_key(comp)
        if k not in seen:
            seen.add(k); jobs.append((tuple(float(v) for v in comp), origin, 240))

    # baseline and the 38 v3 designs (U mapping), same seeds -> identical means, now with per-seed records
    R3 = pd.read_csv(os.path.join(OUT, "mobo_v2_reeval.csv"))
    for _, r in R3.iterrows():
        F_eq, nI, ka = r.f_F, r.n_I, fam.KAPPA_ARYL + (1 - fam.KAPPA_ARYL) * r.f_F
        add((F_eq, nI, ka, r.s_PG, r.L_D, r.c_PAG), "v3:" + (r.chem if r.chem == "baseline" else r.get("selection", "")))
    # family-B search candidates: first two three-objective fronts (up to 12) plus the two-objective front
    parts = [pd.read_csv(os.path.join(OUT, f"round2_searchB_s{s}.csv")) for s in (7, 11)
             if os.path.exists(os.path.join(OUT, f"round2_searchB_s{s}.csv"))]
    if parts:
        d = pd.concat(parts, ignore_index=True); f = d[d.feasible].copy()
        f["rank"] = nondominated_sort(f[["dose", "LWR", "F_ug"]].values.astype(float))
        for _, r in f[f["rank"] <= 1].sort_values(["rank", "LWR"]).head(12).iterrows():
            add((r.F_eq, r.n_I, r.kappa_acid, r.s_PG, r.L_D, r.c_PAG), "B search: 3-objective")
        for _, r in front(f).iterrows():
            add((r.F_eq, r.n_I, r.kappa_acid, r.s_PG, r.L_D, r.c_PAG), "B search: 2-objective")
    C = pd.read_csv(os.path.join(OUT, "round2_emu_candidates.csv"))
    for _, r in C.iterrows():
        add((r.F_eq, r.n_I, r.kappa_acid, r.s_PG, r.L_D, r.c_PAG), f"emulator {r.family}")
    print("re-evaluation jobs:", len(jobs), flush=True)
    with Pool(2) as pool:
        R = pd.DataFrame(pool.map(reeval_job, jobs))
    R.to_csv(os.path.join(OUT, "round2_reeval.csv"), index=False)
    print(R[["origin", "F_eq", "n_I", "kappa_acid", "s_PG", "L_D", "c_PAG", "F_ug", "dose", "LWR", "EL", "missing",
             "feasible", "in_B", "in_H", "in_U"]].round(3).to_string())


def fronts_from(R):
    R = R[R.feasible]
    return {f: front(R[R[f"in_{f}"]]) for f in FAMS}


def cmd_dt():
    R = pd.read_csv(os.path.join(OUT, "round2_reeval.csv"))
    Fr = fronts_from(R)
    pick = set()
    for f in FAMS:
        fr = Fr[f]
        for t in TARGETS + (4.6, 4.85):
            g = fr[fr.LWR <= t]
            if len(g):
                pick.add(g.sort_values("dose").index[0])
    sel = R.loc[sorted(pick)]
    jobs = [((r.F_eq, r.n_I, r.kappa_acid, r.s_PG, r.L_D, r.c_PAG), r.origin, 960) for _, r in sel.iterrows()]
    print("dt jobs:", len(jobs), flush=True)
    with Pool(2) as pool:
        D = pd.DataFrame(pool.map(reeval_job, jobs))
    D["row_240"] = list(sel.index)
    D.to_csv(os.path.join(OUT, "round2_dt0625.csv"), index=False)
    print(D[["origin", "F_eq", "n_I", "s_PG", "L_D", "dose", "LWR", "feasible"]].round(3).to_string())


# ----------------------------------------------------------------------------- premiums and bootstraps
def per_seed_table(R):
    """(n_design, 6) arrays of per-seed dose and LWR (NaN where a seed was not sized)."""
    D = np.full((len(R), len(SEEDS_RE)), np.nan); L = D.copy()
    for i, (_, r) in enumerate(R.iterrows()):
        if not isinstance(r.get("per_seed"), str) or not r.per_seed:
            continue
        ss = [int(s) for s in str(r.seeds_sized).split(";")]
        for s, pair in zip(ss, r.per_seed.split(";")):
            d, l = (float(v) for v in pair.split(","))
            j = SEEDS_RE.index(s); D[i, j] = d; L[i, j] = l
    return D, L


def premiums(dose, lwr, feas, member, targets):
    out = {}
    fr = {f: (lambda m: (dose[m], lwr[m]))(feas & member[f]) for f in FAMS}
    for t in targets:
        db = fr["free"][0][fr["free"][1] <= t]; db = db.min() if db.size else np.nan
        for f in ("B", "H", "U"):
            da = fr[f][0][fr[f][1] <= t]; da = da.min() if da.size else np.nan
            out[(f, t)] = 100 * (db / da - 1) if np.isfinite(da) and np.isfinite(db) else np.nan
    return out


def cmd_premium(fn="round2_reeval.csv", tag="", nboot=4000):
    R = pd.read_csv(os.path.join(OUT, fn)).reset_index(drop=True)
    feas = R.feasible.values.astype(bool)
    member = {f: R[f"in_{f}"].values.astype(bool) for f in FAMS}
    point = premiums(R.dose.values, R.LWR.values, feas, member, TARGETS)
    rng = np.random.default_rng(0)
    # (i) parametric, one draw per design shared by all fronts (independent across designs)
    bi = {k: [] for k in point}
    for _ in range(nboot):
        d = rng.normal(R.dose, R.dose_se); l = rng.normal(R.LWR, R.LWR_se)
        for k, v in premiums(d, l, feas, member, TARGETS).items():
            bi[k].append(v)
    # (ii) paired field-level bootstrap: resample the six random fields jointly for all designs
    Dp, Lp = per_seed_table(R)
    bp = {k: [] for k in point}
    for _ in range(nboot):
        j = rng.integers(0, len(SEEDS_RE), len(SEEDS_RE))
        with np.errstate(all="ignore"):
            d = np.nanmean(Dp[:, j], axis=1); l = np.nanmean(Lp[:, j], axis=1)
        d = np.where(np.isfinite(d), d, R.dose.values); l = np.where(np.isfinite(l), l, R.LWR.values)
        for k, v in premiums(d, l, feas, member, TARGETS).items():
            bp[k].append(v)
    rows = []
    for (f, t), v in point.items():
        a, b = np.array(bi[(f, t)]), np.array(bp[(f, t)])
        q = lambda x, p: np.nanpercentile(x, p) if np.isfinite(x).mean() > 0.5 else np.nan
        rows.append(dict(family=f, LWR_target=t, premium=v, ind_p05=q(a, 5), ind_p95=q(a, 95),
                         paired_p05=q(b, 5), paired_p95=q(b, 95), frac_attained_paired=np.isfinite(b).mean()))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(OUT, f"round2_premium{tag}.csv"), index=False)
    print(P.round(1).to_string())
    Fr = fronts_from(R)
    pd.concat([fr.assign(front=f) for f, fr in Fr.items()]).to_csv(os.path.join(OUT, f"round2_fronts{tag}.csv"), index=False)


def cmd_emu_sweep():
    """Emulator premium versus the largest fluorine-free activity kappa_aryl * s_PG,max (s_PG,max = 3) per family."""
    T = pd.read_csv(os.path.join(OUT, "emulator_v2_table.csv"))
    I = emu.build_interp(T, emu.ALPHAS, emu.KAPPAS, emu.LDS)
    rows = []
    k0 = fam.KAPPA_ARYL
    for km in np.round(np.arange(0.5, 2.01, 0.1), 2):
        fam.KAPPA_ARYL = km / 3.0
        fronts = {f: front(emu_designs(I, f).assign(family=f)) for f in FAMS}
        P = premium_table(fronts, targets=(4.5, 4.75, 5.0, 5.5))
        P["kappa_max_free"] = km
        P["min_LWR_free"] = fronts["free"].LWR.min() if len(fronts["free"]) else np.nan
        rows.append(P)
    fam.KAPPA_ARYL = k0
    W = pd.concat(rows, ignore_index=True)
    W.to_csv(os.path.join(OUT, "round2_emu_sweep.csv"), index=False)
    print(W.pivot_table(index="kappa_max_free", columns="LWR_target", values=["premium_B", "premium_U"]).round(1).to_string())


def cmd_dt_premium():
    """Premium on the subset of premium-setting designs at both time steps."""
    D = pd.read_csv(os.path.join(OUT, "round2_dt0625.csv"))
    R = pd.read_csv(os.path.join(OUT, "round2_reeval.csv"))
    S = R.loc[D.row_240.values].reset_index(drop=True)
    rows = []
    for lab, X in (("dt = 0.25 s", S), ("dt = 0.0625 s", D)):
        feas = X.feasible.values.astype(bool); member = {f: X[f"in_{f}"].values.astype(bool) for f in FAMS}
        for (f, t), v in premiums(X.dose.values, X.LWR.values, feas, member, TARGETS).items():
            rows.append(dict(step=lab, family=f, LWR_target=t, premium=v))
    P = pd.DataFrame(rows).pivot_table(index=["family", "LWR_target"], columns="step", values="premium").reset_index()
    P.to_csv(os.path.join(OUT, "round2_dt_premium.csv"), index=False)
    print(P.round(1).to_string())
    C = pd.DataFrame(dict(origin=D.origin, dose_240=S.dose, dose_960=D.dose, LWR_240=S.LWR, LWR_960=D.LWR,
                          dLWR_pct=100 * (D.LWR / S.LWR - 1), ddose_pct=100 * (D.dose / S.dose - 1)))
    C.to_csv(os.path.join(OUT, "round2_dt_designs.csv"), index=False)
    print(C.round(2).to_string())


if __name__ == "__main__":
    c = sys.argv[1]
    {"emu": cmd_emu, "emu_sweep": cmd_emu_sweep, "reeval": cmd_reeval, "dt": cmd_dt, "dt_premium": cmd_dt_premium}.get(c, lambda: None)()
    if c == "search":
        cmd_search(int(sys.argv[2]))
    if c == "premium":
        cmd_premium()
