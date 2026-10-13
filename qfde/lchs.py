"""
Linear Combination of Hamiltonian Simulation (LCHS), improved kernel of An, Childs & Lin (arXiv 2312.03916):

    e^{-T A} = int_R  f(k) / (1 - i k)  e^{-i T (k L + H)}  dk,      A = L + i H,  L >= 0,
    f(z) = 1 / (C_beta exp((1 + i z)^beta)),   C_beta = 2 pi e^{-2^beta},   beta in (0, 1).

Our generators satisfy  v' = G_hat v  with  G_hat = W^{1/2} G W^{-1/2}  and  sym(G_hat) <= 0  (main theorem),
so  A = -G_hat  has  L = -sym(G_hat) >= 0  and LCHS applies.

This module (i) EMULATES the LCHS formula classically on small instances (truncation + composite
Gauss-Legendre quadrature in k, exact Hamiltonian evolutions by eigendecomposition) to verify the
algorithm end to end, and (ii) COUNTS resources (truncation K, number of LCU terms, coefficient 1-norm,
Hamiltonian-simulation queries) for any instance.
"""
from __future__ import annotations

import numpy as np


def weight(k, beta=0.8):
    Cb = 2 * np.pi * np.exp(-2 ** beta)
    k = np.asarray(k, float)
    return np.exp(-(1 + 1j * k) ** beta) / (Cb * (1 - 1j * k))      # overflow-safe form


def tail_mass(K, beta=0.8):
    """int_{|k| > K} |weight(k)| dk  (numerically, on a log grid)."""
    ks = np.geomspace(K, K * 1e6, 20001)
    vals = np.abs(weight(ks, beta)) + np.abs(weight(-ks, beta))
    return float(np.trapezoid(vals, ks))


def truncation(eps, beta=0.8):
    """Smallest K (to 1%) with tail mass <= eps."""
    lo, hi = 1.0, 2.0
    while tail_mass(hi, beta) > eps:
        hi *= 2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if tail_mass(mid, beta) > eps:
            lo = mid
        else:
            hi = mid
    return hi


def coefficient_norm(K, beta=0.8, n=200001):
    ks = np.linspace(-K, K, n)
    return float(np.trapezoid(np.abs(weight(ks, beta)), ks))


def split(Ghat):
    """A = -G_hat = L + i H  ->  (L, H) Hermitian."""
    A = -np.asarray(Ghat, complex)
    L = 0.5 * (A + A.conj().T)
    H = (A - A.conj().T) / 2j
    return L.real if np.allclose(L.imag, 0) else L, H


def emulate(Ghat, T, v0, eps=1e-8, beta=0.8, Q=10, h1=None):
    """Classically evaluate the discretised LCHS sum  sum_j c_j e^{-iT(k_j L + H)} v0  and return it with
    statistics.  The integrand oscillates in k with frequency <= T ||L||, so composite Gauss-Legendre with
    Q nodes per half period  h1 = pi / (T ||L||)  resolves it (the paper's worst-case step is 1/(e T ||L||))."""
    L, H = split(Ghat)
    nL = max(np.linalg.norm(L, 2), 1e-12)
    K = truncation(eps / 3, beta)
    h1 = h1 or np.pi / (T * nL)
    nint = int(np.ceil(2 * K / h1))
    edges = np.linspace(-K, K, nint + 1)
    xg, wg = np.polynomial.legendre.leggauss(Q)
    out = np.zeros(len(v0), complex)
    c1 = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        ks = (xg + 1) * (b - a) / 2 + a
        cs = weight(ks, beta) * wg * (b - a) / 2
        for k, c in zip(ks, cs):
            lam, V = np.linalg.eigh(k * L + H)
            out += c * (V @ (np.exp(-1j * T * lam) * (V.conj().T @ v0)))
            c1 += abs(c)
    return out, dict(K=K, intervals=nint, terms=nint * Q, coef_1norm=c1, normL=nL,
                     normH=float(np.linalg.norm(H, 2)))


def sparse_norm(M):
    """Normalisation of a standard sparse-access block encoding:  s * max|M_ij|."""
    M = np.asarray(M)
    nz = np.abs(M) > 1e-14
    s = int(nz.sum(axis=1).max()) if nz.any() else 1
    return float(s * np.abs(M).max()), s


def query_count(alpha_L, alpha_H, T, eps, amp, beta=0.8):
    """Block-encoding queries of improved LCHS (time-independent, QSP Hamiltonian simulation):
       queries ~ c1 * amp * ( e/2 * (K alpha_L + alpha_H) * T + ln(3/eps) ),
    K from the kernel tail (eps/3), c1 = coefficient 1-norm, amp = ||v(0)|| / ||output block at T||
    (amplitude amplification rounds).  QSP degree for e^{-i tau H} with ||H|| <= a: ~ e a tau / 2 + ln(1/eps)."""
    K = truncation(eps / 3, beta)
    c1 = coefficient_norm(K, beta)
    degree = np.e / 2 * (K * alpha_L + alpha_H) * T + np.log(3 / eps)
    return float(c1 * amp * degree), dict(K=K, c1=c1, qsp_degree=float(degree))
