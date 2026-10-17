"""
EUV (13.5 nm) photoabsorption of resist components and formulations.

Source data: Henke, Gullikson and Davis atomic scattering factors (CXRO tables,
as distributed with the `periodictable` package, files xsf/*.nff).
Atomic cross section: sigma_a = 2 r_e lambda f2          (Eq. 1)
Linear absorption:    alpha   = sum_i N_i sigma_a,i       (Eq. 2)
Mass absorption:      mu/rho  = N_A sum_i x_i sigma_a,i / M  (Eq. 3)
Independent-atom approximation (no chemical-state correction).
"""
import os, re
import numpy as np
import pandas as pd
import periodictable as pt

LAM_NM = 13.5
E_EV = 1239.84198 / LAM_NM            # 91.84 eV
R_E_CM = 2.8179403262e-13             # classical electron radius, cm
N_A = 6.02214076e23
XSF = os.path.join(os.path.dirname(pt.__file__), "xsf")


def f2(sym):
    d = np.loadtxt(os.path.join(XSF, sym.lower() + ".nff"), skiprows=1)
    return float(np.interp(E_EV, d[:, 0], d[:, 2]))


def sigma_cm2(sym):
    return 2 * R_E_CM * (LAM_NM * 1e-7) * f2(sym)


def parse(formula):
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula):
        out[el] = out.get(el, 0) + (float(n) if n else 1.0)
    return out


def molar_mass(comp):
    return sum(pt.elements.symbol(e).mass * n for e, n in comp.items())


def mass_abs(comp):
    """mu/rho in cm^2/g"""
    return N_A * sum(sigma_cm2(e) * n for e, n in comp.items()) / molar_mass(comp)


def blend(parts):
    """parts: list of (formula, mass fraction). Returns per-gram atom inventory."""
    inv = {}
    for formula, w in parts:
        c = parse(formula)
        M = molar_mass(c)
        for e, n in c.items():
            inv[e] = inv.get(e, 0) + w * n / M      # mol atoms per g
    return inv


def mass_abs_inventory(inv):
    return N_A * sum(sigma_cm2(e) * n for e, n in inv.items())


# ---- Components -----------------------------------------------------------
TPS = "C18H15S"   # triphenylsulfonium cation
ANIONS = {
    # label: (anion formula, fluorinated?, note)
    "PFOS":      ("C8F17SO3", True,  "perfluorooctanesulfonate (legacy, phased out)"),
    "Nonaflate": ("C4F9SO3",  True,  "perfluorobutanesulfonate"),
    "Triflate":  ("CF3SO3",   True,  "trifluoromethanesulfonate"),
    "Tosylate":  ("C7H7SO3",  False, "p-toluenesulfonate"),
    "Camphorsulfonate": ("C10H15SO4", False, "10-camphorsulfonate"),
    "4-Iodobenzenesulfonate": ("C6H4ISO3", False, "single iodine, F-free"),
    "2,4,6-Triiodobenzenesulfonate": ("C6H2I3SO3", False, "three iodines, F-free"),
}
# Polymer: 4-hydroxystyrene / tert-butyl acrylate 65:35 (ESCAP-type), per mol repeat
HS, TBA = "C8H8O", "C7H12O2"

COMPONENTS = {
    "H": "H", "C": "C", "O": "O", "F": "F", "S": "S", "I": "I", "Sn": "Sn",
    "Poly(4-hydroxystyrene)": HS,
    "PTFE repeat (C2F4)": "C2F4",
    "SnO2": "SnO2", "HfO2": "HfO2", "ZrO2": "ZrO2", "ZnO": "ZnO",
}


def polymer_formula():
    c1, c2 = parse(HS), parse(TBA)
    tot = {e: 0.65 * c1.get(e, 0) + 0.35 * c2.get(e, 0) for e in set(c1) | set(c2)}
    return "".join(f"{e}{tot[e]:.4f}" for e in sorted(tot))


def car_formulation(anion_label, pag_mmol_per_g=0.27, density=1.20):
    an, fl, note = ANIONS[anion_label]
    pag = parse(TPS)
    for e, n in parse(an).items():
        pag[e] = pag.get(e, 0) + n
    pag_formula = "".join(f"{e}{n:g}" for e, n in pag.items())
    Mpag = molar_mass(pag)
    w_pag = pag_mmol_per_g * 1e-3 * Mpag
    inv = blend([(pag_formula, w_pag), (polymer_formula(), 1 - w_pag)])
    mu = mass_abs_inventory(inv)
    # fraction of absorbed photons absorbed by PAG atoms
    inv_pag = blend([(pag_formula, w_pag)])
    frac_pag = mass_abs_inventory(inv_pag) / mu
    nF = parse(an).get("F", 0)
    return dict(anion=anion_label, fluorinated=fl, note=note, PAG_formula=pag_formula,
                PAG_MW=round(Mpag, 2), PAG_wt_pct=round(100 * w_pag, 2),
                F_wt_pct=round(100 * w_pag * nF * pt.F.mass / Mpag, 2),
                mu_rho_cm2_g=mu, density_g_cm3=density,
                alpha_um_inv=mu * density * 1e-4,
                absorbed_fraction_30nm=1 - np.exp(-mu * density * 30e-7),
                frac_absorbed_by_PAG=frac_pag)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    atoms = pd.DataFrame([dict(element=e, f2=f2(e), sigma_cm2=sigma_cm2(e),
                               sigma_per_mass_cm2_g=sigma_cm2(e) * N_A / pt.elements.symbol(e).mass)
                          for e in ["H", "C", "N", "O", "F", "S", "Cl", "Br", "I", "Si", "Zn",
                                    "Zr", "Sn", "Sb", "Te", "Hf", "Bi"]])
    atoms.to_csv(os.path.join(here, "henke_f2_13p5nm.csv"), index=False)
    rows = [car_formulation(a) for a in ANIONS]
    car = pd.DataFrame(rows)
    car.to_csv(os.path.join(here, "car_formulations_13p5nm.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"E = {E_EV:.3f} eV")
    print(atoms.round(4).to_string(index=False))
    print(car.drop(columns=["note", "PAG_formula"]).round(4).to_string(index=False))
    for k, v in COMPONENTS.items():
        print(f"{k:28s} mu/rho = {mass_abs(parse(v)):9.0f} cm2/g")
