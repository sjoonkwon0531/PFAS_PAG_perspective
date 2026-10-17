"""
Solver verification tests (v2), requested in the internal review (E1, E2).
Run: python test_solver.py   -> writes results/solver_tests.csv and prints PASS/FAIL.

T1  diffusion only, Dirac voxel: mass conserved, min >= -1e-12, variance = 2 D t per axis (lattice)
T2  diffusion only, uniform field: unchanged
T3  diffusion only, real Poisson quencher field: mass conserved, no negatives
T4  diffusion + equal-amount neutralization: sum(A - Q) invariant
T5  uniform saturation: numerical deprotection equals 1 - exp(-Gamma) (Eq. 14)
T6  clipping audit on the baseline realization (v1 vs v2 propagator)
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import fft
from car_stochastic import Params, _diffusion_operator, _diffuse, simulate, amplification_number

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []


def rec(name, ok, **kw):
    rows.append(dict(test=name, passed=bool(ok), **kw))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  " + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()))


shape, D, dt, nstep = (64, 48, 30), 0.3, 1.0, 60
op = _diffusion_operator(shape, 1.0, D, dt)
# T1
c = np.zeros(shape); c[32, 24, 15] = 1.0
for _ in range(nstep):
    c = _diffuse(c, op)
x = np.arange(shape[0]) - 32
var_x = float((c.sum(axis=(1, 2)) * x ** 2).sum())
rec("T1 Dirac mass", abs(c.sum() - 1) < 1e-12, mass_err=abs(c.sum() - 1), min=float(c.min()))
rec("T1 Dirac variance", abs(var_x / (2 * D * dt * nstep) - 1) < 1e-6, var_ratio=var_x / (2 * D * dt * nstep))
# T2
u = np.full(shape, 3.7)
for _ in range(10):
    u = _diffuse(u, op)
rec("T2 uniform", np.max(np.abs(u - 3.7)) < 1e-12, max_dev=float(np.max(np.abs(u - 3.7))))
# T3
rng = np.random.default_rng(1)
q = rng.poisson(0.03, size=shape).astype(float); m0 = q.sum()
opq = _diffusion_operator(shape, 1.0, 1.5 ** 2 / 120, 1.0)
negmass = 0.0
for _ in range(60):
    q = _diffuse(q, opq); negmass += -q[q < 0].sum(); q[q < 0] = 0
rec("T3 Poisson field", abs(q.sum() - m0) / m0 < 1e-10 and negmass < 1e-9, rel_mass_err=abs(q.sum() - m0) / m0,
    clipped=negmass)
# T4
A = rng.poisson(0.2, size=shape).astype(float); Q = rng.poisson(0.03, size=shape).astype(float)
inv0 = (A - Q).sum()
opA = _diffusion_operator(shape, 1.0, 0.3, 1.0)
for _ in range(60):
    A = _diffuse(A, opA); np.maximum(A, 0, out=A)
    Q = _diffuse(Q, opq); np.maximum(Q, 0, out=Q)
    n = np.minimum(A, Q); A -= n; Q -= n
rec("T4 A-Q invariant", abs((A - Q).sum() - inv0) / inv0 < 1e-10, rel_err=abs((A - Q).sum() - inv0) / inv0)
# T5 uniform saturation: all PAG converted, homogeneous
p = Params(kp=0.12725, Lx=1)
G = amplification_number(p)
rho, q_ = p.rho_pag, p.quencher_ratio
P = 1.0; a = rho * (1 - q_)
for _ in range(p.n_steps):
    P *= np.exp(-p.kp * p.kappa * a * p.t_peb / p.n_steps)
rec("T5 saturation vs Eq. 14", abs((1 - P) - (1 - np.exp(-G))) < 1e-12, m_num=1 - P, m_eq14=1 - np.exp(-G))


# T6 clipping audit with the real baseline realization: v1 continuous-k^2 propagator vs v2
def op_v1(shape, dx, D, dt):
    nx, ny, nz = shape
    kx = 2 * np.pi * fft.fftfreq(nx, dx); ky = 2 * np.pi * fft.fftfreq(ny, dx)
    kz = np.pi * np.arange(nz) / (nz * dx)
    K2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    return np.exp(-D * K2 * dt)


from scipy.ndimage import gaussian_filter
from car_stochastic import photons_per_nm2
p = Params(kp=0.12725, seed=1)
for label, opf in (("v1 spectral k^2", op_v1), ("v2 lattice", _diffusion_operator)):
    rng = np.random.default_rng(1)
    pitch = 2 * p.hp; ny = int(p.n_pitch * pitch); nx = p.Lx; nz = p.thickness
    yy = np.arange(ny) + 0.5; I = 1 + p.modulation * np.cos(2 * np.pi * yy / pitch)
    z = np.arange(nz); al = p.alpha_um * 1e-3; absz = np.exp(-al * z) - np.exp(-al * (z + 1))
    mean = np.broadcast_to(photons_per_nm2(35.0048) * I[None, :, None] * absz[None, None, :], (nx, ny, nz))
    nph = rng.poisson(mean).astype(float)
    eps = gaussian_filter(nph, sigma=(2, 2, 2), mode=("wrap", "wrap", "reflect"))
    prob = 1 - np.exp(-p.phi / p.rho_pag * eps)
    A = rng.binomial(rng.poisson(p.rho_pag, size=nph.shape), prob).astype(float)
    Q = rng.poisson(p.quencher_ratio * p.rho_pag, size=nph.shape).astype(float)
    n = np.minimum(A, Q); A -= n; Q -= n; Q0 = Q.sum()
    oA = opf(A.shape, 1.0, p.L_D ** 2 / (2 * p.t_peb), 1.0); oQ = opf(A.shape, 1.0, p.L_Q ** 2 / (2 * p.t_peb), 1.0)
    gA = gQ = 0.0
    for _ in range(60):
        A = _diffuse(A, oA); gA += -A[A < 0].sum(); A[A < 0] = 0
        Q = _diffuse(Q, oQ); gQ += -Q[Q < 0].sum(); Q[Q < 0] = 0
        n = np.minimum(A, Q); A -= n; Q -= n
    ok = True if label.startswith("v1") else gQ / Q0 < 1e-10
    rec(f"T6 clipping audit, {label}", ok, created_A=gA, created_Q=gQ, created_Q_frac=gQ / Q0)

pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "solver_tests.csv"), index=False)
print("ALL PASS" if all(r["passed"] for r in rows) else "SOME FAILED")
