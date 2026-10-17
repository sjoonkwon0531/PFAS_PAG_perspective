import pandas as pd, os
from multiprocessing import Pool
from round2 import reeval_job, OUT
H = pd.read_csv(os.path.join(OUT, "round3_h55.csv")); H = H[H.LWR <= 5.5]
jobs = [((r.F_eq, r.n_I, r.kappa_acid, r.s_PG, r.L_D, r.c_PAG), "round3 H near 5.5", 960) for _, r in H.iterrows()]
with Pool(2) as p:
    D = pd.DataFrame(p.map(reeval_job, jobs))
D["dose_240"] = H.dose.values; D["LWR_240"] = H.LWR.values
D.to_csv(os.path.join(OUT, "round3_h55_dt.csv"), index=False)
print(D[["dose_240", "dose", "LWR_240", "LWR", "LWR_se"]].round(3))
