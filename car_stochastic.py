"""
PFAS-free EUV resist simulator (v0.1)
=====================================
Voxel-level stochastic exposure + continuum post-exposure-bake (PEB) chemistry
for a chemically amplified resist (CAR), dense line/space pattern.

Why this model: stochastic (shot-noise) effects dominate patterning error at EUV
resolution, so a mean-field resist model cannot show how changing the photoacid
generator (PAG) chemistry trades off resolution, line-edge roughness and dose
(the RLS trade-off). We therefore sample photons and PAG molecules discretely
and then solve the PEB reaction-diffusion chemistry on the noisy field.

Governing equations (numbered as in the manuscript draft v2):
 (4)  I(y) = 1 + V cos(2 pi y / p)                           aerial image, visibility V
 (5)  <n_ph(r)> = Phi * I(y)/<I> * [exp(-alpha z) - exp(-alpha (z+dz))] * dx dy
      n_ph ~ Poisson(<n_ph>)                              photon shot noise
 (6)  eps(r) = (G_sigma_e * n_ph)(r)                      secondary-electron blur
 (7)  A0(r) ~ Binomial(N_PAG(r), 1 - exp(-c * eps(r))),   N_PAG ~ Poisson(rho_PAG dV)
      c = phi / (rho_PAG dV)  so that acid yield per absorbed photon -> phi at low dose
 (8)  dA/dt = D_A lap A - k_q A Q
 (9)  dQ/dt = D_Q lap Q - k_q A Q            (k_q -> infinity: instantaneous neutralization)
 (10) dP/dt = -k_p A P                       (P: protected fraction, P(0)=1)
 (12) development: resist dissolves where m = 1 - P > m_th
(11) Diffusion length L_D = sqrt(2 D_A t_PEB).
Numerics: operator splitting per time step (diffuse A and Q by exact Fourier/DCT propagator,
then deprotect with the diffused acid, then neutralize A and Q instantaneously).
Boundary conditions: periodic in x (along line) and y (across pitch), no-flux in z.
"""
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy import fft
from scipy.ndimage import gaussian_filter

E_PH_J = 6.62607015e-34 * 2.99792458e8 / 13.5e-9      # 1.4714e-17 J
N_A = 6.02214076e23


@dataclass
class Params:
    # geometry (nm)
    hp: float = 16.0
    n_pitch: int = 2
    Lx: int = 200
    thickness: int = 30
    dx: float = 1.0
    # optics
    modulation: float = 0.85          # aerial image visibility V for dense L/S (Eq. 4)
    # resist absorption (from data/euv_absorption.py)
    alpha_um: float = 4.74             # 1/um
    # acid generation
    pag_mmol_g: float = 0.27
    density: float = 1.20              # g/cm3
    phi: float = 2.0                   # acids per absorbed photon (low-dose limit)
    sigma_e: float = 2.0               # nm, secondary electron blur
    # quencher
    quencher_ratio: float = 0.15       # mol quencher / mol PAG
    # PEB
    t_peb: float = 60.0                # s
    L_D: float = 6.0                   # nm, acid diffusion length
    L_Q: float = 1.5                   # nm, quencher diffusion length (bulky base)
    kp: float = 0.12725                # nm^3 / s, calibrated (mean-field dose-to-size 35 mJ/cm2, baseline)
    kappa: float = 1.0                 # acid-strength multiplier on kp
    n_steps: int = 240                 # dt = 0.25 s (v2 production; see convergence study)
    m_th: float = 0.5
    seed: int = 0

    @property
    def rho_pag(self):                 # molecules per nm^3
        return self.pag_mmol_g * 1e-3 * self.density * N_A * 1e-21


def photons_per_nm2(dose_mJ_cm2):
    return dose_mJ_cm2 * 1e-3 / E_PH_J * 1e-14


def _diffusion_operator(shape, dx, D, dt):
    """Exact propagator of the lattice (semi-discrete) diffusion equation dc/dt = D L_h c,
    where L_h is the standard second-order discrete Laplacian (periodic in x and y, no-flux in z).
    Using the discrete eigenvalues (2 - 2 cos(k dx)) / dx^2 instead of the continuous k^2 makes the
    propagator a Markov semigroup on the grid: it conserves mass exactly and keeps every voxel
    non-negative, so no clipping (and no artificial mass creation) is needed.  (v2 solver)"""
    nx, ny, nz = shape
    lx = (2 - 2 * np.cos(2 * np.pi * fft.fftfreq(nx))) / dx ** 2
    ly = (2 - 2 * np.cos(2 * np.pi * fft.fftfreq(ny))) / dx ** 2
    lz = (2 - 2 * np.cos(np.pi * np.arange(nz) / nz)) / dx ** 2      # DCT-II = Neumann (cell-centred)
    lam = lx[:, None, None] + ly[None, :, None] + lz[None, None, :]
    return np.exp(-D * lam * dt)


def _diffuse(field, op):
    """Apply the lattice propagator: DCT-II in z (no-flux), real FFT in x, complex FFT in y."""
    nx = field.shape[0]
    f = fft.dct(field, type=2, axis=2, norm="ortho")
    f = fft.rfftn(f, axes=(1, 0))                       # real transform along x (last listed axis)
    f *= op[: nx // 2 + 1]
    f = fft.irfftn(f, s=(field.shape[1], nx), axes=(1, 0))
    return fft.idct(f, type=2, axis=2, norm="ortho")


def simulate(p: Params, dose, stochastic=True, return_fields=False):
    rng = np.random.default_rng(p.seed)
    pitch = 2 * p.hp
    ny = int(round(p.n_pitch * pitch / p.dx))
    nx = int(round(p.Lx / p.dx)) if stochastic else 1
    nz = int(round(p.thickness / p.dx))
    y = (np.arange(ny) + 0.5) * p.dx
    # line centred at y = pitch/2 (dark), space centred at y = 0 (bright)
    I = 1.0 + p.modulation * np.cos(2 * np.pi * y / pitch)
    z = np.arange(nz) * p.dx
    a = p.alpha_um * 1e-3                                   # 1/nm
    absz = np.exp(-a * z) - np.exp(-a * (z + p.dx))         # fraction absorbed per layer
    mean_ph = photons_per_nm2(dose) * p.dx ** 2 * I[None, :, None] * absz[None, None, :]
    mean_ph = np.broadcast_to(mean_ph, (nx, ny, nz))
    dV = p.dx ** 3
    c = p.phi / (p.rho_pag * dV)
    if stochastic:
        nph = rng.poisson(mean_ph).astype(float)
    else:
        nph = mean_ph.copy()
    sig = p.sigma_e / p.dx
    eps = gaussian_filter(nph, sigma=(sig, sig, sig), mode=("wrap", "wrap", "reflect"))
    prob = 1 - np.exp(-c * eps)
    if stochastic:
        npag = rng.poisson(p.rho_pag * dV, size=nph.shape)
        A = rng.binomial(npag, prob).astype(float)
        Q = rng.poisson(p.quencher_ratio * p.rho_pag * dV, size=nph.shape).astype(float)
    else:
        A = p.rho_pag * dV * prob
        Q = np.full(nph.shape, p.quencher_ratio * p.rho_pag * dV)
    A /= dV; Q /= dV                                        # concentrations, 1/nm^3
    acid_initial = A.copy()
    dt = p.t_peb / p.n_steps
    DA = p.L_D ** 2 / (2 * p.t_peb)
    DQ = p.L_Q ** 2 / (2 * p.t_peb)
    opA = _diffusion_operator(A.shape, p.dx, DA, dt)
    opQ = _diffusion_operator(A.shape, p.dx, DQ, dt) if DQ > 0 else None
    P = np.ones_like(A)
    keff = p.kp * p.kappa
    # initial neutralization
    n = np.minimum(A, Q); A -= n; Q -= n
    for _ in range(p.n_steps):
        A = _diffuse(A, opA); np.maximum(A, 0, out=A)      # removes only round-off (|negatives| ~ 1e-16)
        if opQ is not None:
            Q = _diffuse(Q, opQ); np.maximum(Q, 0, out=Q)
        P *= np.exp(-keff * A * dt)                         # Eq. 10 (acid present during the step)
        n = np.minimum(A, Q); A -= n; Q -= n               # Eqs. 8-9, k_q -> inf
    m = 1 - P
    out = dict(m=m, dose=dose)
    if return_fields:
        out.update(acid0=acid_initial, nph=nph)
    return out


def edges(m, p: Params, z_index=None):
    """Return list of edge-position arrays (one per edge) along x, via threshold crossings."""
    if z_index is None:
        z_index = m.shape[2] // 2
    sl = m[:, :, z_index]
    nx, ny = sl.shape
    pitch = 2 * p.hp
    res = []
    lines = []
    for ip in range(p.n_pitch):
        yc = (ip + 0.5) * pitch                                  # line centre
        left = np.full(nx, np.nan); right = np.full(nx, np.nan)
        for ix in range(nx):
            row = sl[ix]
            jc = int(yc / p.dx)
            # walk outwards from line centre until m exceeds threshold
            if row[jc] >= p.m_th:
                continue                                         # line missing (bridge break)
            j = jc
            while j > jc - int(pitch / 2 / p.dx) and row[j] < p.m_th:
                j -= 1
            if row[j] >= p.m_th:
                f = (p.m_th - row[j]) / (row[j + 1] - row[j])
                left[ix] = (j + 0.5 + f) * p.dx
            j = jc
            while j < jc + int(pitch / 2 / p.dx) - 1 and row[j] < p.m_th:
                j += 1
            if row[j] >= p.m_th:
                f = (p.m_th - row[j - 1]) / (row[j] - row[j - 1])
                right[ix] = (j - 1 + 0.5 + f) * p.dx
        lines.append((left, right))
    return lines


def metrics(m, p: Params):
    """CD, LER (3 sigma), LWR (3 sigma), defect fraction, averaged over depth slices."""
    nz = m.shape[2]
    zs = [nz // 4, nz // 2, 3 * nz // 4]
    cds, lers, lwrs, miss = [], [], [], []
    for zi in zs:
        for left, right in edges(m, p, zi):
            ok = np.isfinite(left) & np.isfinite(right)
            miss.append(1 - ok.mean())
            if ok.sum() < 10:
                continue
            w = right[ok] - left[ok]
            cds.append(w.mean())
            lwrs.append(3 * w.std())
            lers.append(1.5 * (left[ok].std() + right[ok].std()))
    if not cds:
        return dict(CD=np.nan, LER=np.nan, LWR=np.nan, missing=float(np.mean(miss)))
    return dict(CD=float(np.mean(cds)), LER=float(np.mean(lers)), LWR=float(np.mean(lwrs)),
                missing=float(np.mean(miss)))


MF_STEPS = 60   # mean-field (noise-free) solves: dose-to-size changes by < 0.2% between 60 and 960 steps


def _mf(p):
    from copy import copy
    q = copy(p); q.n_steps = min(p.n_steps, MF_STEPS)
    return q


def mean_field_cd(p: Params, dose):
    m = simulate(_mf(p), dose, stochastic=False)["m"]
    sl = m[0, :, m.shape[2] // 2]
    pitch = 2 * p.hp
    ny = len(sl)
    yc = 0.5 * pitch
    jc = int(yc / p.dx)
    half = int(pitch / 2 / p.dx)
    seg = sl[jc - half: jc + half]
    ys = (np.arange(jc - half, jc + half) + 0.5) * p.dx
    below = seg < p.m_th
    if not below.any():
        return 0.0
    if below.all():
        return pitch
    # linear-interpolated width of the region below threshold
    idx = np.where(below)[0]
    i0, i1 = idx[0], idx[-1]
    f0 = (p.m_th - seg[i0 - 1]) / (seg[i0] - seg[i0 - 1])
    f1 = (p.m_th - seg[i1]) / (seg[i1 + 1] - seg[i1])
    return (ys[i1] + f1 * p.dx) - (ys[i0 - 1] + f0 * p.dx)


def dose_to_size_meanfield(p: Params, lo=5, hi=300, target=None):
    target = p.hp if target is None else target
    if mean_field_cd(p, hi) > target:
        return np.nan                                   # cannot be sized at any dose
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        cd = mean_field_cd(p, mid)
        if cd > target:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def amplification_number(p: Params):
    """Gamma = kappa k_p rho_PAG (1 - q) t_PEB  (Eq. 15). Patterning requires
    Gamma > ln[1/(1 - m_th)] even at full PAG conversion."""
    return p.kappa * p.kp * p.rho_pag * (1 - p.quencher_ratio) * p.t_peb


def latent_image_metrics(p: Params):
    """Mean-field exposure latitude (dose range for CD within +/-10 %, % of dose-to-size)
    and normalized latent-image slope hp*|dm/dy| at the line edge."""
    ds = dose_to_size_meanfield(p)
    if not np.isfinite(ds):
        return dict(EL=np.nan, LILS=np.nan)
    d_lo = dose_to_size_meanfield(p, target=1.1 * p.hp)
    d_hi = dose_to_size_meanfield(p, target=0.9 * p.hp)
    EL = 100 * (d_hi - d_lo) / ds
    mm = simulate(_mf(p), ds, stochastic=False)["m"]
    m = mm[0, :, mm.shape[2] // 2]
    y = (np.arange(len(m)) + 0.5) * p.dx
    g = np.gradient(m, y)
    yl = p.hp - p.hp / 2                                 # left edge of the line centred at y = hp
    LILS = p.hp * abs(np.interp(yl, y, g))
    return dict(EL=float(EL), LILS=float(LILS))


# ---------------------------------------------------------------------------
# Metal oxide resist (MOR), negative tone, non-amplified
# ---------------------------------------------------------------------------
@dataclass
class MORParams(Params):
    """Tin-oxo-cage-type negative-tone resist.
    (13)  conv(r) ~ Binomial(N_Sn(r), 1 - exp(-c_Sn * eps(r))) / N_Sn(r),  c_Sn = Y_Sn / (rho_Sn dV);
          short-range condensation m_c = G_Lc * conv ;  line retained where m_c > c_th
    """
    alpha_um: float = 13.7             # data/euv_absorption.py: tin-oxo cage, density 1.9 g/cm3 (assumed)
    rho_sn: float = 5.56               # Sn atoms per nm^3 at 1.9 g/cm3
    eta: float = 5.5                   # Y_Sn: converted Sn sites per absorbed photon (calibrated, ~45 mJ/cm2)
    L_c: float = 1.0                   # nm, condensation (crosslink) range
    c_th: float = 0.30
    thickness: int = 20


def simulate_mor(p: MORParams, dose, stochastic=True):
    rng = np.random.default_rng(p.seed)
    pitch = 2 * p.hp
    ny = int(round(p.n_pitch * pitch / p.dx))
    nx = int(round(p.Lx / p.dx)) if stochastic else 1
    nz = int(round(p.thickness / p.dx))
    y = (np.arange(ny) + 0.5) * p.dx
    I = 1.0 - p.modulation * np.cos(2 * np.pi * y / pitch)       # bright at the line centre
    z = np.arange(nz) * p.dx
    a = p.alpha_um * 1e-3
    absz = np.exp(-a * z) - np.exp(-a * (z + p.dx))
    mean_ph = np.broadcast_to(photons_per_nm2(dose) * p.dx ** 2 * I[None, :, None] * absz[None, None, :],
                              (nx, ny, nz))
    nph = rng.poisson(mean_ph).astype(float) if stochastic else mean_ph.copy()
    sig = p.sigma_e / p.dx
    eps = gaussian_filter(nph, sigma=(sig, sig, sig), mode=("wrap", "wrap", "reflect"))
    dV = p.dx ** 3
    c = p.eta / (p.rho_sn * dV)
    prob = 1 - np.exp(-c * eps)
    if stochastic:
        nsn = rng.poisson(p.rho_sn * dV, size=nph.shape)
        conv = np.where(nsn > 0, rng.binomial(nsn, prob) / np.maximum(nsn, 1), 0.0)
    else:
        conv = prob
    if p.L_c > 0:
        s = p.L_c / p.dx
        conv = gaussian_filter(conv, sigma=(s, s, s), mode=("wrap", "wrap", "reflect"))
    # map to the CAR convention used by edges(): line where m < m_th
    m = 1.0 - conv
    return dict(m=m, dose=dose)


def mor_as_params(p: MORParams):
    """Copy with m_th mapped so that edges()/metrics() can be reused."""
    from copy import copy
    q = copy(p); q.m_th = 1.0 - p.c_th
    return q


def mor_cd_meanfield(p: MORParams, dose):
    q = mor_as_params(p)
    m = simulate_mor(p, dose, stochastic=False)["m"]
    sl = m[0, :, m.shape[2] // 2]
    pitch = 2 * p.hp
    jc = int(0.5 * pitch / p.dx); half = int(pitch / 2 / p.dx)
    seg = sl[jc - half: jc + half]
    ys = (np.arange(jc - half, jc + half) + 0.5) * p.dx
    below = seg < q.m_th
    if not below.any():
        return 0.0
    if below.all():
        return pitch
    idx = np.where(below)[0]; i0, i1 = idx[0], idx[-1]
    f0 = (q.m_th - seg[i0 - 1]) / (seg[i0] - seg[i0 - 1])
    f1 = (q.m_th - seg[i1]) / (seg[i1 + 1] - seg[i1])
    return (ys[i1] + f1 * p.dx) - (ys[i0 - 1] + f0 * p.dx)


def mor_dose_to_size(p: MORParams, lo=2, hi=500, target=None):
    target = p.hp if target is None else target
    # negative tone: CD increases with dose
    if mor_cd_meanfield(p, hi) < target:
        return np.nan
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if mor_cd_meanfield(p, mid) < target:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


if __name__ == "__main__":
    import time
    p = Params()
    print("rho_PAG = %.3f /nm3, photons/nm2 at 30 mJ/cm2 = %.2f" % (p.rho_pag, photons_per_nm2(30)))
    d = dose_to_size_meanfield(p)
    print("mean-field dose-to-size: %.1f mJ/cm2" % d)
    t = time.time()
    r = simulate(p, d)
    print(metrics(r["m"], p), "%.1fs" % (time.time() - t))
