"""
Gate-level LCHS circuits for small LK-weighted generators (hardware demonstrations).

For v' = G_hat v with sym(G_hat) <= 0 (main theorem), write A = -G_hat = L + iH and use the improved LCHS
identity (qfde.lchs)

    e^{t G_hat} = int f(k)/(1 - ik) e^{-it(kL + H)} dk  ~  sum_j c_j U_j,   U_j = e^{-it(k_j L + H)},

with Gauss-Legendre nodes k_j on [-K, K].  The circuit is the standard linear combination of unitaries:

    |0>_anc |v0>  --PREP-->  sum_j sqrt(|c_j|/s) |j> |v0>
                  --SELECT-> sum_j sqrt(|c_j|/s) |j> e^{i arg c_j} U_j |v0>
                  --PREP^dag, keep anc = 0-->  (1/s) sum_j c_j U_j |v0>,      s = sum_j |c_j|.

So P(anc = 0, sys = i) = |(sum_j c_j U_j v0)_i|^2 / (s ||v0||)^2.  Each U_j is compiled exactly from its
matrix (no oracles): at this size the point is to run the LCHS structure itself (superposition over k,
controlled Hamiltonian evolutions, interference, postselection), not to construct Hamiltonian-simulation
oracles.  A generator that is not dissipative (no LK weight) is shifted, G - mu I with mu = lambda_max(sym G),
and the output is rescaled by e^{mu t}; the postselection probability then drops by e^{-2 mu t}.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import expm, block_diag

from . import lchs


def nodes(M, K, beta=0.7):
    x, w = np.polynomial.legendre.leggauss(M)
    k = K * x
    c = K * w * lchs.weight(k, beta)
    return k, c


def lcu_operator(Gs, t, k, c):
    """The operator the ideal circuit applies (before normalization by s): sum_j c_j U_j."""
    L, H = lchs.split(Gs)
    return sum(cj * expm(-1j * t * (kj * L + H)) for kj, cj in zip(k, c))


def choose_K(Gs, times, M, beta=0.7, Ks=np.linspace(0.5, 8, 31), out=None):
    """Truncation K minimizing the worst LCHS error over the time grid, independent of the initial vector:
    the relative operator-norm error, or, if an output map `out` is given, the relative error of out @ e^{t Gs}."""
    best = None
    for K in Ks:
        k, c = nodes(M, K, beta)
        if out is None:
            e = max(np.linalg.norm(lcu_operator(Gs, t, k, c) - expm(Gs * t), 2) / np.linalg.norm(expm(Gs * t), 2)
                    for t in times)
        else:
            e = max(np.linalg.norm(out @ (lcu_operator(Gs, t, k, c) - expm(Gs * t))) / np.linalg.norm(out @ expm(Gs * t))
                    for t in times)
        if best is None or e < best[0]:
            best = (e, K)
    return best[1], best[0]


def pad(Gs, v0):
    n = Gs.shape[0]
    N = 1 << max(1, math.ceil(math.log2(n)))
    Gp = np.zeros((N, N), complex); Gp[:n, :n] = Gs
    vp = np.zeros(N, complex); vp[:n] = v0
    return Gp, vp


def circuit(Gs, v0, t, k, c, measure=True):
    """LCHS circuit. Qubits 0..ns-1 = system (little-endian index i), ns..ns+na-1 = ancilla (index j)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation, UnitaryGate
    Gp, vp = pad(np.asarray(Gs, complex), np.asarray(v0, complex))
    N = Gp.shape[0]; ns = int(math.log2(N)); M = len(k); na = int(math.log2(M))
    assert 1 << na == M
    L, H = lchs.split(Gp)
    s = float(np.sum(np.abs(c)))
    sel = block_diag(*[np.exp(1j * np.angle(cj)) * expm(-1j * t * (kj * L + H)) for kj, cj in zip(k, c)])
    amp = np.sqrt(np.abs(c) / s)
    qc = QuantumCircuit(ns + na, ns + na if measure else 0)
    sysq = list(range(ns)); ancq = list(range(ns, ns + na))
    qc.append(StatePreparation(vp / np.linalg.norm(vp)), sysq)
    qc.append(StatePreparation(amp), ancq)
    qc.append(UnitaryGate(sel, label="SELECT"), sysq + ancq)
    qc.append(StatePreparation(amp).inverse(), ancq)
    if measure:
        qc.measure(range(ns + na), range(ns + na))
    return qc, dict(ns=ns, na=na, s=s, norm_v0=float(np.linalg.norm(vp)))


def ideal_probs(Gs, v0, t, k, c):
    """Exact P(anc = 0, sys = i) of the ideal circuit."""
    Gp, vp = pad(np.asarray(Gs, complex), np.asarray(v0, complex))
    v = lcu_operator(Gp, t, k, c) @ vp
    s = float(np.sum(np.abs(c)))
    return np.abs(v) ** 2 / (s * np.linalg.norm(vp)) ** 2


def amplitudes(p, meta, shift=0.0, t=0.0):
    """Unsigned amplitudes of v(t) from P(anc=0, sys=i) (signs are known to be + for the demo instances)."""
    return np.sqrt(np.maximum(p, 0.0)) * meta["s"] * meta["norm_v0"] * math.exp(shift * t)
