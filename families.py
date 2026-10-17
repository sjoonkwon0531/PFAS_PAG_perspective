"""
Round-2 review (P0): chemically defined formulation families for the 12 nm design analysis.

All families share the linear mappings of Table 4, written for a PAG mixture of anion endmembers:
    alpha = 4.43 + (4.36 + 0.38 F_eq + 0.303 n_I - 4.43) * c_PAG / 0.27        [um^-1]
    kappa = kappa_acid * s_PG,   kappa_acid = sum_j x_j kappa_j
    film F per wafer-layer = F_FILM_PER_MMOLG * c_PAG * F_eq
where x_j are mole fractions of the anion endmembers, F_eq is the mean number of F atoms per anion divided
by 9 (nonaflate = 1) and n_I the mean number of iodine atoms per anion.

Endmembers
    N  nonaflate (perfluorobutanesulfonate)        F 9, I 0, kappa 1
    S  non-iodinated aryl sulfonate (tosylate)     F 0, I 0, kappa KAPPA_ARYL
    T  triiodinated aryl sulfonate                 F 0, I 3, kappa KAPPA_ARYL
    H  hybrid triiodobenzoyloxy-tetrafluoroalkanesulfonate (class described in US 10,831,100 B2)
                                                    F 4, I 3, kappa KAPPA_H (assumed 1)
Families
    free   S + T                       (f_F = 0)
    B      N + S + T  (blend)          F_eq = x_N, n_I = 3 x_T, hence F_eq + n_I/3 <= 1
    H      N + S + T + H (blend incl. hybrid anion)
    U      independent f_F and n_I (unconstrained benchmark used in v3, not an upper bound; F_eq = f_F, kappa_acid from f_F)
"""
import numpy as np

KAPPA_ARYL, KAPPA_H = 0.3, 1.0
THK = 22
F_FILM_PER_MMOLG = 117.48 / 0.27 * (THK / 30)


def mix(xN=0.0, xS=0.0, xT=0.0, xH=0.0):
    tot = xN + xS + xT + xH
    xN, xS, xT, xH = (v / tot for v in (xN, xS, xT, xH))
    F_eq = xN + 4.0 / 9.0 * xH
    nI = 3.0 * (xT + xH)
    kacid = xN * 1.0 + xH * KAPPA_H + (xS + xT) * KAPPA_ARYL
    return F_eq, nI, kacid


def params(F_eq, nI, kacid, sPG, LD, c):
    alpha = 4.43 + (4.36 + 0.38 * F_eq + 0.303 * nI - 4.43) * (c / 0.27)
    return dict(alpha_um=alpha, kappa=kacid * sPG, L_D=LD, pag_mmol_g=c), F_FILM_PER_MMOLG * c * F_eq


def from_U(fF, nI, sPG, LD, c):
    """v3 independent mapping (Table 4)."""
    return params(fF, nI, KAPPA_ARYL + (1 - KAPPA_ARYL) * fF, sPG, LD, c)


def member(fam, F_eq, nI, kacid, tol=1e-6):
    """Is a design with (F_eq, n_I, kappa_acid) attainable in family fam?"""
    a = (kacid - KAPPA_ARYL) / (1 - KAPPA_ARYL)                    # strong-acid fraction (x_N + x_H)
    if fam == "free":
        return F_eq <= tol and a <= tol and nI <= 3 + tol
    if fam == "B":
        return abs(a - F_eq) <= 1e-4 and F_eq + nI / 3 <= 1 + 1e-4
    if fam == "H":
        # x_N + x_H = a, x_N + 4/9 x_H = F_eq  ->  x_H = (a - F_eq) * 9/5, x_N = a - x_H ; x_T = nI/3 - x_H
        xH = (a - F_eq) * 9.0 / 5.0; xN = a - xH; xT = nI / 3.0 - xH; xS = 1 - xN - xH - xT
        return min(xH, xN, xT, xS) >= -1e-4
    if fam == "U":
        return abs(a - F_eq) <= 1e-4
    raise ValueError(fam)


def enumerate_family(fam, step=0.05):
    """Composition grid (F_eq, n_I, kappa_acid) for a family."""
    g = np.round(np.arange(0, 1 + 1e-9, step), 4)
    out = set()
    if fam == "free":
        for xT in g:
            out.add(mix(xS=1 - xT, xT=xT))
    elif fam == "B":
        for xN in g:
            for xT in g:
                if xN + xT <= 1 + 1e-9:
                    out.add(mix(xN=xN, xT=xT, xS=max(0.0, 1 - xN - xT)))
    elif fam == "H":
        for xN in g:
            for xH in g:
                for xT in g:
                    if xN + xH + xT <= 1 + 1e-9:
                        out.add(mix(xN=xN, xH=xH, xT=xT, xS=max(0.0, 1 - xN - xH - xT)))
    elif fam == "U":
        for fF in g:
            for nI in 3 * g:
                out.add((fF, nI, KAPPA_ARYL + (1 - KAPPA_ARYL) * fF))
    return sorted({tuple(round(v, 6) for v in t) for t in out})
