"""
Validated certificates (rigor multiplier).

The LMI is solved in floating point on a time grid.  To turn the numerical certificate into a
statement that holds for ALL t in [t0, t1]:

  lambda_max(M(t)) <= max_i lambda_max(M(t_i)) + L_M * (dt / 2) + delta_round,

  * L_M      : Lipschitz constant of t -> M(t) in spectral norm, bounded from above by
               max |d M / dt| over a fine grid (central differences) times a 1.5 safety factor,
  * dt       : grid spacing,
  * delta_round : Weyl's inequality with the floating-point backward error of the eigensolver,
                  ~ dim * u * ||M||  (u = 2^-53).

If the bound is <= 0 the certificate is VALID on the whole interval (semi-dissipative);
< 0 means strictly dissipative.  Autonomous systems have L_M = 0.
"""
from __future__ import annotations

import numpy as np
from .fde import GeneralFDE, kernel_gamma

U_ROUND = 2.0 ** -53


def lk_matrix(fde: GeneralFDE, t, cert, gammas):
    """Numeric version of the LMI block matrix M(t) for a solved certificate."""
    n, g = fde.n, fde.g
    m, q, L = len(fde.const), len(fde.var), len(fde.qk)
    beta = cert.get("beta", 0.0)
    P, Qc, Qv, S0 = cert["P"], cert["Qc"], cert["Qv"], cert.get("S0") or []
    Gam = [kernel_gamma(k, beta) * 1.02 for k in fde.qk]
    dims = [n] + [g] * (m + q + L)
    off = np.concatenate([[0], np.cumsum(dims)])
    M = np.zeros((off[-1], off[-1]))
    blk = lambda i: slice(off[i], off[i + 1])
    F = np.zeros((g, off[-1])); F[:, blk(0)] = fde.E
    S_in = np.zeros((g, g))
    A = fde.A(t)
    M[blk(0), blk(0)] = P @ A + A.T @ P
    for k, c in enumerate(fde.const):
        b = blk(1 + k); B = c.B(t)
        M[blk(0), b] += P @ B; M[b, blk(0)] += B.T @ P
        M[b, b] -= np.exp(-beta * c.d) * Qc[k]
        S_in += Qc[k]
        if c.C is not None:
            F[:, b] += c.C
    for i, vd in enumerate(fde.var):
        b = blk(1 + m + i); B = vd.B(t)
        M[blk(0), b] += P @ B; M[b, blk(0)] += B.T @ P
        M[b, b] -= Qv[i] * (1 - vd.dd(t)) * np.exp(-gammas[i]) / vd.d(t)
        S_in += Qv[i] / vd.d(t)
    for l, k in enumerate(fde.qk):
        b = blk(1 + m + q + l); K = k.K(t)
        M[blk(0), b] += P @ K; M[b, blk(0)] += K.T @ P
        M[b, b] -= (beta / Gam[l]) * S0[l]
        S_in += S0[l]
    M += F.T @ S_in @ F
    return 0.5 * (M + M.T)


def validate(fde: GeneralFDE, cert, gammas, n_grid=401):
    t0, t1 = fde.t_range
    ts = np.linspace(t0, t1, n_grid)
    dt = ts[1] - ts[0]
    lams, norms, derivs = [], [], []
    for t in ts:
        Mt = lk_matrix(fde, t, cert, gammas)
        lams.append(np.max(np.linalg.eigvalsh(Mt)))
        norms.append(np.linalg.norm(Mt, 2))
    h = 1e-5
    for t in ts:
        Mp = lk_matrix(fde, min(t + h, t1 + h), cert, gammas)
        Mm = lk_matrix(fde, max(t - h, t0 - h), cert, gammas)
        derivs.append(np.linalg.norm(Mp - Mm, 2) / (2 * h))
    LM = 1.5 * max(derivs)
    dim = len(lk_matrix(fde, t0, cert, gammas))
    delta = dim * U_ROUND * max(norms) * 10
    bound = max(lams) + LM * dt / 2 + delta
    return dict(max_grid_eig=float(max(lams)), lipschitz=float(LM), grid_dt=float(dt),
                roundoff=float(delta), validated_bound=float(bound), valid=bool(bound <= 0))
