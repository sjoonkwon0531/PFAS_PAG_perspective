"""
v2 emulator for the sensitivity of the fluorine-free exposure-dose premium (Fig. 6d, Table S1, Fig. S1).

Changes from v1 (internal review S2, S3, E5):
  * kappa grid extended down to 0.37, just above the c_PAG = 0.40 saturation-limit floor (0.369)
  * guarded sizing with seed-based standard errors (three seeds, three doses)
  * hold-out validation: 16 random (alpha, kappa, L_D) points simulated directly and compared with
    the interpolated prediction
  * nuisance checks: reduced tables (alpha in {4.3, 5.7}, eight kappa values, L_D = 6.5 nm) with the
    acid yield phi = 1.5 and 3, electron blur sigma_e = 3 nm, and development threshold m_th = 0.6
Usage: python emulator_v2.py table | holdout | nuisance | analyze
"""
import os, sys, itertools
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.interpolate import RegularGridInterpolator
from car_stochastic import Params, latent_image_metrics, amplification_number
from sizing import size_at

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
KP0, HP, THK, CPAG = 0.12725, 12, 22, 0.40
ALPHAS = [4.3, 5.0, 5.7, 6.4]
KAPPAS = [0.37, 0.42, 0.48, 0.55, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0, 3.0, 5.0]
LDS = [5.0, 6.5, 8.0]
EL_MIN, MISS_MAX = 12.0, 0.05
NUIS = {"ref": {}, "phi1.5": dict(phi=1.5), "phi3": dict(phi=3.0), "sigma3": dict(sigma_e=3.0), "mth0.6": dict(m_th=0.6)}
NUIS_ALPHAS, NUIS_KAPPAS, NUIS_LD = [4.3, 5.7], [0.42, 0.55, 0.65, 0.8, 1.0, 1.25, 1.6, 2.5], 6.5


def node(args):
    a, k, L, extra = args
    p = Params(hp=HP, thickness=THK, kp=KP0, alpha_um=a, kappa=k, L_D=L, pag_mmol_g=CPAG, **extra)
    r = size_at(p, seeds=(21, 22, 23))
    lim = latent_image_metrics(p) if r["status"] == "ok" else dict(EL=np.nan, LILS=np.nan)
    return dict(alpha_um=a, kappa=k, L_D=L, **r, **lim)


def build_interp(T, alphas, kappas, lds):
    I = {}
    for L in lds:
        sub = T[np.isclose(T.L_D, L)]
        for q, tr in (("dose", np.log), ("LWR", None), ("EL", None), ("missing", None)):
            Z = sub.pivot_table(index="alpha_um", columns="kappa", values=q).reindex(index=alphas, columns=kappas).values
            Z = tr(Z) if tr is not None else Z
            I[(L, q)] = RegularGridInterpolator((np.array(alphas), np.log(kappas)), Z, bounds_error=False, fill_value=np.nan)
    return I


def predict(I, a, k, L, lds):
    """Interpolate linearly in L_D between the two bracketing nodes."""
    lds = np.array(lds)
    j = np.clip(np.searchsorted(lds, L) - 1, 0, len(lds) - 2)
    w = (L - lds[j]) / (lds[j + 1] - lds[j])
    out = {}
    for q in ("dose", "LWR", "EL", "missing"):
        v0 = I[(lds[j], q)]([[a, np.log(k)]])[0]; v1 = I[(lds[j + 1], q)]([[a, np.log(k)]])[0]
        v = (1 - w) * v0 + w * v1
        out[q] = np.exp(v) if q == "dose" else v
    return out


def enumerate_designs(I, lds, ka, smax, free, alphas):
    fF = np.array([0.0]) if free else np.linspace(0, 1, 21)
    nI = np.linspace(0, 3, 13); sPG = np.linspace(1, smax, 21)
    rows = []
    for L in lds:
        F, N, S = [g.ravel() for g in np.meshgrid(fF, nI, sPG, indexing="ij")]
        alpha = 4.43 + (4.36 + 0.38 * F + 0.303 * N - 4.43) * (CPAG / 0.27)
        kap = (ka + (1 - ka) * F) * S
        pts = np.c_[alpha, np.log(kap)]
        d = np.exp(I[(L, "dose")](pts)); lwr = I[(L, "LWR")](pts); el = I[(L, "EL")](pts); ms = I[(L, "missing")](pts)
        rows.append(pd.DataFrame(dict(f_F=F, n_I=N, s_PG=S, L_D=L, alpha=alpha, kappa=kap, dose=d, LWR=lwr, EL=el, missing=ms)))
    df = pd.concat(rows, ignore_index=True)
    df["feasible"] = np.isfinite(df.dose) & np.isfinite(df.LWR) & (df.EL >= EL_MIN) & (df.missing < MISS_MAX)
    return df


def front(df):
    f = df[df.feasible].sort_values(["dose", "LWR"]); best, keep = np.inf, []
    for i, r in f.iterrows():
        if r.LWR < best - 1e-9:
            keep.append(i); best = r.LWR
    return f.loc[keep]


def premium_curve(I, lds, alphas, targets=(4.4, 4.5, 4.75, 5.0), kmax_grid=None, smax=3.0):
    kmax_grid = np.round(np.arange(0.5, 2.01, 0.1), 2) if kmax_grid is None else kmax_grid
    out = []
    for km in kmax_grid:
        ka = km / smax
        A = front(enumerate_designs(I, lds, ka, smax, False, alphas))
        B = front(enumerate_designs(I, lds, ka, smax, True, alphas))
        r = dict(kappa_max_free=km, min_LWR_free=B.LWR.min() if len(B) else np.nan,
                 min_dose_free=B.dose.min() if len(B) else np.nan, min_LWR_allowed=A.LWR.min())
        for t in targets:
            da = A[A.LWR <= t].dose.min() if (A.LWR <= t).any() else np.nan
            db = B[B.LWR <= t].dose.min() if len(B) and (B.LWR <= t).any() else np.nan
            r[f"premium_{t}"] = 100 * (db / da - 1) if np.isfinite(da) and np.isfinite(db) else np.nan
        out.append(r)
    return pd.DataFrame(out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    pd.set_option("display.width", 250)
    if cmd == "table":
        jobs = [(a, k, L, {}) for a, k, L in itertools.product(ALPHAS, KAPPAS, LDS)]
        with Pool(2) as pool:
            T = pd.DataFrame(pool.map(node, jobs, chunksize=3))
        T.to_csv(os.path.join(OUT, "emulator_v2_table.csv"), index=False)
        print(T.status.value_counts())
        print(T.pivot_table(index="kappa", columns="L_D", values="LWR").round(2))
    elif cmd == "holdout":
        rng = np.random.default_rng(42)
        pts = [(float(rng.uniform(4.3, 6.4)), float(np.exp(rng.uniform(np.log(0.6), np.log(3.0)))),
                float(rng.uniform(5.0, 8.0)), {}) for _ in range(16)]
        with Pool(2) as pool:
            H = pd.DataFrame(pool.map(node, pts))
        H.to_csv(os.path.join(OUT, "emulator_v2_holdout_sim.csv"), index=False)
        print(H[["alpha_um", "kappa", "L_D", "status", "dose", "LWR", "LWR_se"]].round(3).to_string())
    elif cmd == "nuisance":
        for key, extra in NUIS.items():
            if key == "ref":
                continue
            jobs = [(a, k, NUIS_LD, extra) for a, k in itertools.product(NUIS_ALPHAS, NUIS_KAPPAS)]
            with Pool(2) as pool:
                Tn = pd.DataFrame(pool.map(node, jobs))
            Tn["variant"] = key
            Tn.to_csv(os.path.join(OUT, f"emulator_v2_nuis_{key}.csv"), index=False)
            print(key, Tn.status.value_counts().to_dict(), flush=True)
        jobs = [(a, k, NUIS_LD, {}) for a, k in itertools.product(NUIS_ALPHAS, NUIS_KAPPAS)]
        with Pool(2) as pool:
            Tn = pd.DataFrame(pool.map(node, jobs))
        Tn["variant"] = "ref"; Tn.to_csv(os.path.join(OUT, "emulator_v2_nuis_ref.csv"), index=False)
    elif cmd == "nuis_rel":
        # premium at targets defined relative to the best fluorine-allowed LWR of each variant (+0.25 and +0.5 nm),
        # so that variants with different roughness levels can be compared
        rows = []
        for key in NUIS:
            Tn = pd.read_csv(os.path.join(OUT, f"emulator_v2_nuis_{key}.csv"))
            In = build_interp(Tn, NUIS_ALPHAS, NUIS_KAPPAS, [NUIS_LD])
            for km in np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5]):
                ka = km / 3.0
                A = front(enumerate_designs(In, [NUIS_LD], ka, 3.0, False, NUIS_ALPHAS))
                B = front(enumerate_designs(In, [NUIS_LD], ka, 3.0, True, NUIS_ALPHAS))
                r = dict(variant=key, kappa_max_free=km, best_LWR_allowed=A.LWR.min(),
                         min_LWR_free=B.LWR.min() if len(B) else np.nan)
                for d in (0.25, 0.5):
                    t = A.LWR.min() + d
                    da = A[A.LWR <= t].dose.min()
                    db = B[B.LWR <= t].dose.min() if len(B) and (B.LWR <= t).any() else np.nan
                    r[f"premium_rel{d}"] = 100 * (db / da - 1) if np.isfinite(db) else np.nan
                rows.append(r)
        N = pd.DataFrame(rows); N.to_csv(os.path.join(OUT, "emulator_v2_nuisance_premium_rel.csv"), index=False)
        print(N.round(2).to_string())
    elif cmd == "analyze":
        T = pd.read_csv(os.path.join(OUT, "emulator_v2_table.csv"))
        I = build_interp(T, ALPHAS, KAPPAS, LDS)
        # hold-out validation
        H = pd.read_csv(os.path.join(OUT, "emulator_v2_holdout_sim.csv"))
        H = H[H.status == "ok"].copy()
        pr = [predict(I, r.alpha_um, r.kappa, r.L_D, LDS) for _, r in H.iterrows()]
        H["dose_pred"] = [x["dose"] for x in pr]; H["LWR_pred"] = [x["LWR"] for x in pr]
        H["dose_err_pct"] = 100 * (H.dose_pred / H.dose - 1); H["LWR_err"] = H.LWR_pred - H.LWR
        H.to_csv(os.path.join(OUT, "emulator_v2_holdout.csv"), index=False)
        print("holdout n=%d  dose RMSE %.2f%%  LWR RMSE %.3f nm  (mean LWR_se %.3f)" % (
            len(H), np.sqrt(np.mean(H.dose_err_pct ** 2)), np.sqrt(np.mean(H.LWR_err ** 2)), H.LWR_se.mean()))
        # premium curve and named cases
        W = premium_curve(I, LDS, ALPHAS)
        W.to_csv(os.path.join(OUT, "emulator_v2_kappa_max_sweep.csv"), index=False)
        print(W.round(1).to_string())
        cases = [("ref", 0.3, 3.0), ("ka02", 0.2, 3.0), ("ka05", 0.5, 3.0), ("spg2", 0.3, 2.0), ("spg5", 0.3, 5.0)]
        rows, fr = [], []
        for key, ka, smax in cases:
            A = front(enumerate_designs(I, LDS, ka, smax, False, ALPHAS)); B = front(enumerate_designs(I, LDS, ka, smax, True, ALPHAS))
            for nm, f in (("fluorine-allowed", A), ("fluorine-free", B)):
                g = f.copy(); g["case"] = key; g["campaign"] = nm; fr.append(g)
            r = dict(case=key, kappa_aryl=ka, sPG_max=smax, product=ka * smax,
                     min_dose_free=B.dose.min() if len(B) else np.nan, min_LWR_free=B.LWR.min() if len(B) else np.nan)
            for t in (4.4, 4.5, 4.75, 5.0):
                da = A[A.LWR <= t].dose.min() if (A.LWR <= t).any() else np.nan
                db = B[B.LWR <= t].dose.min() if len(B) and (B.LWR <= t).any() else np.nan
                r[f"premium_{t}"] = 100 * (db / da - 1) if np.isfinite(da) and np.isfinite(db) else np.nan
            rows.append(r)
        S = pd.DataFrame(rows); S.to_csv(os.path.join(OUT, "emulator_v2_cases.csv"), index=False)
        pd.concat(fr, ignore_index=True).to_csv(os.path.join(OUT, "emulator_v2_fronts.csv"), index=False)
        print(S.round(1).to_string())
        # nuisance variants (single L_D node, reduced alpha grid)
        NR = []
        for key in NUIS:
            fn = os.path.join(OUT, f"emulator_v2_nuis_{key}.csv")
            if not os.path.exists(fn):
                continue
            Tn = pd.read_csv(fn)
            In = build_interp(Tn, NUIS_ALPHAS, NUIS_KAPPAS, [NUIS_LD])
            Wn = premium_curve(In, [NUIS_LD], NUIS_ALPHAS, kmax_grid=np.array([0.6, 0.8, 1.0, 1.2, 1.5]))
            Wn["variant"] = key; NR.append(Wn)
        if NR:
            N = pd.concat(NR, ignore_index=True); N.to_csv(os.path.join(OUT, "emulator_v2_nuisance_premium.csv"), index=False)
            print(N.round(1).to_string())
