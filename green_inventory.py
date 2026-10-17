"""
Fluorine (PFAS) and tin mass balance per 300 mm wafer exposure layer.

Why a mass balance: the sustainability case for PFAS-free resists rests on how much
persistent fluorochemical leaves the fab and through which stream. The film itself is
fixed by geometry, so the retained inventory is exact for a given formulation; the
dispensed inventory scales with the spin-coat utilization eta_u, which we treat as a
swept parameter rather than a single literature value.

 (18) m_film   = rho_film * h * A_wafer
 (19) m_disp   = m_film / eta_u
 (20) m_F,X    = m_X * w_PAG * w_F,PAG            (X = film, dispensed)
 (21) streams:  spin-off/edge-bead = m_disp - m_film ;
                developer effluent  = f_open * m_film  (positive tone, open-area fraction) ;
                plasma strip/etch   = (1 - f_open) * m_film
"""
import os
import numpy as np
import pandas as pd
import periodictable as pt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from euv_absorption import ANIONS, TPS, parse, molar_mass, N_A

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
A_WAFER = np.pi * 15.0 ** 2           # cm^2, 300 mm wafer
F_MASS = pt.F.mass


def pag_props(anion, mmol_g=0.27):
    an = ANIONS[anion][0]
    comp = parse(TPS)
    for e, n in parse(an).items():
        comp[e] = comp.get(e, 0) + n
    M = molar_mass(comp)
    wF = comp.get("F", 0) * F_MASS / M
    return M, mmol_g * 1e-3 * M, wF


def car_inventory(anion, h_nm=30, rho=1.2, eta_u=0.05, f_open=0.5, mmol_g=0.27):
    m_film = rho * h_nm * 1e-7 * A_WAFER                    # g
    M, w_pag, wF = pag_props(anion, mmol_g)
    mF_film = m_film * w_pag * wF
    mF_disp = mF_film / eta_u
    return dict(anion=anion, h_nm=h_nm, eta_u=eta_u, m_film_mg=1e3 * m_film,
                F_film_ug=1e6 * mF_film, F_dispensed_ug=1e6 * mF_disp,
                F_spin_waste_ug=1e6 * (mF_disp - mF_film),
                F_developer_ug=1e6 * f_open * mF_film,
                F_strip_ug=1e6 * (1 - f_open) * mF_film)


def mor_inventory(h_nm=20, rho=1.9, eta_u=0.05):
    c = parse("C48H116O22Sn12")
    wSn = 12 * pt.Sn.mass / molar_mass(c)
    m_film = rho * h_nm * 1e-7 * A_WAFER
    return dict(h_nm=h_nm, eta_u=eta_u, m_film_mg=1e3 * m_film, Sn_film_ug=1e6 * m_film * wSn,
                Sn_dispensed_ug=1e6 * m_film * wSn / eta_u)


if __name__ == "__main__":
    rows = [car_inventory(a) for a in ANIONS]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fluorine_inventory.csv"), index=False)
    print(df.round(2).to_string(index=False))
    etas = np.logspace(np.log10(0.01), np.log10(0.5), 40)
    sweep = pd.DataFrame([car_inventory("Nonaflate", eta_u=e) for e in etas] +
                         [car_inventory("PFOS", eta_u=e) for e in etas])
    sweep.to_csv(os.path.join(OUT, "fluorine_vs_utilization.csv"), index=False)
    mor = pd.DataFrame([mor_inventory(eta_u=e) for e in etas])
    mor.to_csv(os.path.join(OUT, "tin_vs_utilization.csv"), index=False)
    print(pd.DataFrame([mor_inventory()]).round(2).to_string(index=False))
