"""
Shared, guarded dose-to-size evaluation (v2), replacing five separate implementations.

Review items addressed: E3 (uncertainty by independent field), E6 (adaptive bracketing, monotonicity,
explicit failure reasons), E2 (single production time step).

size_at(p, kind) returns a dict with
  status        'ok' | 'analytic_infeasible' (Gamma <= ln 2) | 'unsized_in_range' | 'not_bracketed'
  dose, LWR, LER, missing      values interpolated on the ensemble-mean CD curve at CD = hp
  dose_se, LWR_se, LER_se, missing_se   standard errors across independent seeds (per-seed interpolation)
  n_seeds_sized                 seeds whose own CD curve bracketed the target
  doses_used                    the dose ladder actually evaluated
LWR is the valid-row LWR (rows without an edge pair are excluded and counted in 'missing', the
missing-edge-pair segment fraction).
"""
import numpy as np
from car_stochastic import (Params, MORParams, simulate, simulate_mor, metrics, mor_as_params,
                            dose_to_size_meanfield, mor_dose_to_size, amplification_number)

N_STEPS = 240          # production time step dt = 0.25 s (see SI, convergence study)


def _target_slope(kind):
    return -1 if kind == "CAR" else +1        # CAR: CD falls with dose; MOR (negative tone): rises


def size_at(p, kind="CAR", factors=(0.85, 1.0, 1.15), seeds=(1, 2, 3), max_extend=3):
    sim = simulate if kind == "CAR" else simulate_mor
    q = p if kind == "CAR" else mor_as_params(p)
    d0 = dose_to_size_meanfield(p) if kind == "CAR" else mor_dose_to_size(p)
    out = dict(status="ok", dose=np.nan, LWR=np.nan, LER=np.nan, missing=np.nan,
               dose_se=np.nan, LWR_se=np.nan, LER_se=np.nan, missing_se=np.nan,
               n_seeds_sized=0, doses_used="", dose_mf=d0)
    if kind == "CAR":
        out["Gamma"] = amplification_number(p)
    if not np.isfinite(d0):
        out["status"] = ("analytic_infeasible" if kind == "CAR" and out["Gamma"] <= np.log(2)
                         else "unsized_in_range")
        return out
    table = {}                                  # dose -> list of per-seed metric dicts

    def run(d):
        res = []
        for s in seeds:
            p.seed = s
            res.append(metrics(sim(p, d)["m"], q))
        table[d] = res

    for f in factors:
        run(d0 * f)
    sgn = _target_slope(kind)
    for _ in range(max_extend):
        ds = np.array(sorted(table))
        cd = np.array([np.nanmean([r["CD"] for r in table[d]]) for d in ds])
        ok = np.isfinite(cd)
        if ok.sum() >= 2 and np.nanmin(cd) <= p.hp <= np.nanmax(cd):
            break
        # extend the ladder toward the target
        if np.all(~ok) or (sgn < 0 and np.nanmin(cd) > p.hp) or (sgn > 0 and np.nanmax(cd) < p.hp):
            run(ds.max() * 1.2)
        else:
            run(ds.min() / 1.2)
    ds = np.array(sorted(table))
    out["doses_used"] = ";".join(f"{d:.2f}" for d in ds)
    keys = ("CD", "LWR", "LER", "missing")
    M = {k: np.array([[r[k] for r in table[d]] for d in ds]) for k in keys}   # (n_dose, n_seed)
    cd = np.nanmean(M["CD"], axis=1)
    ok = np.isfinite(cd)
    if ok.sum() < 2 or not (np.nanmin(cd) <= p.hp <= np.nanmax(cd)):
        out["status"] = "not_bracketed"
        return out
    # Production rule: the ensemble-mean CD values are sorted by CD and the dose at CD = hp is obtained
    # by linear interpolation in the sorted (CD, dose) pairs. For a monotone ladder this equals local linear
    # interpolation inside the bracketing dose segment. Diagnostics (round-2 review E-R2-3): number of
    # crossings of CD = hp along the dose-ordered ladder, monotonicity flag, and the bracketing-segment
    # estimate (crossing closest to the mean-field dose) as a cross-check.
    x, y = ds[ok], cd[ok]
    order = np.argsort(y)
    dsz = float(np.interp(p.hp, y[order], x[order]))
    out["dose"] = dsz
    r = y - p.hp
    segs = [i for i in range(len(x) - 1) if r[i] == 0 or r[i] * r[i + 1] < 0]
    out["n_crossings"] = len(segs)
    out["monotone"] = bool(np.all(np.diff(y) < 0)) if kind == "CAR" else bool(np.all(np.diff(y) > 0))
    if segs:
        i = min(segs, key=lambda k: abs(0.5 * (x[k] + x[k + 1]) - d0))
        out["dose_bracket"] = float(x[i] + (p.hp - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])) if y[i + 1] != y[i] else float(x[i])
    else:
        out["dose_bracket"] = np.nan
    for k in ("LWR", "LER", "missing"):
        out[k] = float(np.interp(dsz, ds[ok], np.nanmean(M[k], axis=1)[ok]))
    # per-seed estimates for standard errors (independent fields)
    per = {k: [] for k in ("dose", "LWR", "LER", "missing")}
    for j in range(len(seeds)):
        c = M["CD"][:, j]; m = np.isfinite(c)
        if m.sum() < 2 or not (c[m].min() <= p.hp <= c[m].max()):
            continue
        o = np.argsort(c[m]); dj = float(np.interp(p.hp, c[m][o], ds[m][o]))
        per["dose"].append(dj)
        for k in ("LWR", "LER", "missing"):
            per[k].append(float(np.interp(dj, ds[m], M[k][m, j])))
    n = len(per["dose"])
    out["n_seeds_sized"] = n
    out["per_seed"] = ";".join(f"{a:.4f},{b:.4f}" for a, b in zip(per["dose"], per["LWR"]))   # dose,LWR per sized seed
    out["seeds_sized"] = ";".join(str(seeds[j]) for j in range(len(seeds))
                                  if np.isfinite(M["CD"][:, j]).sum() >= 2 and
                                  np.nanmin(M["CD"][:, j]) <= p.hp <= np.nanmax(M["CD"][:, j]))
    if n >= 2:
        for k in per:
            out[f"{k}_se"] = float(np.std(per[k], ddof=1) / np.sqrt(n))
    return out
