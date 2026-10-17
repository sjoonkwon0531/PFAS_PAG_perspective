"""
Summarize the sensitivity study of the fluorine-free dose premium.
For each case, the re-evaluated Pareto designs (three doses, three seeds) of the fluorine-allowed
and fluorine-free campaigns are combined into attainment fronts. The reference case pools the
main campaign (mobo_pareto_reeval.csv) with the independent reduced-budget campaign of the
sensitivity study, because both use identical assumptions.
Output: results/sensitivity_summary.csv, results/sensitivity_fronts.csv
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
CASES = [("ref", "Reference (κ_aryl = 0.3, s_PG ≤ 3, linear)", ["mobo_pareto_reeval.csv", "mobo_pareto_reeval_sens_ref.csv"]),
         ("ka02", "Weaker non-fluorinated acid (κ_aryl = 0.2)", ["mobo_pareto_reeval_sens_ka02.csv"]),
         ("ka05", "Stronger non-fluorinated acid (κ_aryl = 0.5)", ["mobo_pareto_reeval_sens_ka05.csv"]),
         ("spg5", "More labile protecting groups (s_PG ≤ 5)", ["mobo_pareto_reeval_sens_spg5.csv"]),
         ("logk", "Log-linear acid-strength mapping", ["mobo_pareto_reeval_sens_logk.csv"])]
TARGETS = [4.2, 4.3, 4.4, 4.5, 4.75, 5.0]


def pareto(Y):
    keep = np.ones(len(Y), bool)
    for i in range(len(Y)):
        if np.any(np.all(Y <= Y[i], 1) & np.any(Y < Y[i], 1)):
            keep[i] = False
    return keep


def front(df):
    df = df[df.feasible]
    if len(df) == 0:
        return df
    return df[pareto(df[["dose", "LWR"]].values)].sort_values("dose")


rows, fronts = [], []
for key, label, files in CASES:
    parts = [pd.read_csv(os.path.join(R, f)) for f in files if os.path.exists(os.path.join(R, f))]
    if not parts:
        continue
    d = pd.concat(parts, ignore_index=True)
    base = d[d.campaign == "baseline"]
    A, B = front(d[d.campaign == "fluorine-allowed"]), front(d[d.campaign == "fluorine-free"])
    for nm, f in (("fluorine-allowed", A), ("fluorine-free", B)):
        g = f.copy(); g["case"] = key; g["campaign"] = nm; fronts.append(g)
    row = dict(case=key, label=label, n_allowed=len(A), n_free=len(B),
               base_dose=base.dose.mean(), base_LWR=base.LWR.mean(),
               min_LWR_allowed=A.LWR.min() if len(A) else np.nan,
               min_LWR_free=B.LWR.min() if len(B) else np.nan)
    for t in TARGETS:
        da = A[A.LWR <= t].dose.min() if len(A) and (A.LWR <= t).any() else np.nan
        db = B[B.LWR <= t].dose.min() if len(B) and (B.LWR <= t).any() else np.nan
        row[f"dA_{t}"] = da; row[f"dB_{t}"] = db
        row[f"premium_{t}"] = 100 * (db / da - 1) if np.isfinite(da) and np.isfinite(db) else np.nan
    # best fluorine-free LWR achievable at or below the baseline dose
    bd = row["base_dose"]
    row["free_LWR_at_base_dose"] = B[B.dose <= bd].LWR.min() if len(B) and (B.dose <= bd).any() else np.nan
    rows.append(row)

S = pd.DataFrame(rows)
S.to_csv(os.path.join(R, "sensitivity_summary.csv"), index=False)
pd.concat(fronts, ignore_index=True).to_csv(os.path.join(R, "sensitivity_fronts.csv"), index=False)
pd.set_option("display.width", 250)
print(S.round(2).to_string())
