"""
Quantum circuits for the certified generator (verification-scale; Qiskit).

1. sparse_block_encoding(M)
   Gilyen-Su-Low-Wiebe style sparse-access block encoding (Lemma 48, arXiv 1806.01838):
       registers  f (1 flag qubit) | R1 (n qubits) | R2 (n qubits, the system)
       U = SWAP(R1,R2) . P_r^dagger . O_A . P_c ,     <0_f 0_R1 i| U |0_f 0_R1 j>  =  M_ij / (s * max|M|)
   with P_c = O_c (H^m on R1),  O_c |l>|j> = |c(j,l)>|j>   (l-th nonzero row of column j),
        P_r = O_r (H^m on R2),  O_r |i>|k> = |i>|r(i,k)>   (k-th nonzero column of row i),
        O_A = multiplexed RY on f:  |0>|i>|j> -> (a_ij |0> + sqrt(1-a_ij^2) |1>)|i>|j>,  a = M / max|M|.
   The index oracles are realised here as explicit basis permutations (fine at verification scale;
   at scale they are reversible arithmetic, costed in qfde.ftqc).  The normalisation s*max|M| is exactly
   the one used in every cost figure (qfde.lchs.sparse_norm).

2. dilation(V)
   The main theorem => V = exp(G_hat T) is a contraction (||V|| <= 1).  Every contraction has the one-ancilla
   unitary dilation (Halmos)
        U = [[ V, (I - V V^T)^(1/2) ], [ (I - V^T V)^(1/2), -V^T ]],
   so the certified evolution is a physical quantum process: prepare |v0>, apply U, keep ancilla = 0.
   (Computing V classically is exponential in general -- this is a hardware CORRECTNESS demonstration
   of the certified dynamics, not the scalable algorithm.)
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _pad_pow2(M):
    N = M.shape[0]
    n = max(1, int(np.ceil(np.log2(N))))
    P = np.zeros((2 ** n, 2 ** n), dtype=M.dtype)
    P[:N, :N] = M
    return P, n


def _perm_unitary(perm):
    """Unitary of the basis permutation |x> -> |perm[x]> (Qiskit little-endian integer labels)."""
    D = len(perm)
    U = np.zeros((D, D))
    U[perm, np.arange(D)] = 1.0
    return U


def _complete_bijection(first, size):
    """Map l -> first[l] for l < len(first) (distinct), then fill the rest bijectively."""
    used = set(first)
    rest = [v for v in range(size) if v not in used]
    return list(first) + rest


# ----------------------------------------------------------------------------------------------
# 1. sparse-access block encoding
# ----------------------------------------------------------------------------------------------
def sparse_block_encoding(M, return_info=False):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate, UCRYGate

    Mp, n = _pad_pow2(np.asarray(M, float))
    D = 2 ** n
    nz = np.abs(Mp) > 1e-14
    s_raw = max(int(nz.sum(axis=1).max()), int(nz.sum(axis=0).max()), 1)
    m = int(np.ceil(np.log2(s_raw))) if s_raw > 1 else 0
    s = 2 ** m
    if s > D:
        raise ValueError("sparsity exceeds dimension")
    amax = np.abs(Mp).max()
    a = Mp / amax
    # row / column nonzero lists, padded with distinct zero positions up to s
    rows_of_col = []
    for j in range(D):
        lst = list(np.nonzero(nz[:, j])[0])
        lst += [r for r in range(D) if r not in lst][: s - len(lst)]
        rows_of_col.append(lst)
    cols_of_row = []
    for i in range(D):
        lst = list(np.nonzero(nz[i, :])[0])
        lst += [c for c in range(D) if c not in lst][: s - len(lst)]
        cols_of_row.append(lst)
    # pattern consistency (padding entries must be zeros, and (i,j) in pattern iff i in rows_of_col[j])
    # O_c on (R1 value l, R2 value j):  index = l + D*j  (R1 = low qubits)
    perm_c = np.zeros(D * D, int)
    for j in range(D):
        bij = _complete_bijection(rows_of_col[j], D)
        for l in range(D):
            perm_c[l + D * j] = bij[l] + D * j
    perm_r = np.zeros(D * D, int)
    for i in range(D):
        bij = _complete_bijection(cols_of_row[i], D)
        for k in range(D):
            perm_r[i + D * k] = i + D * bij[k]
    # multiplexed RY on f, controlled by (R1 = i, R2 = j); UCRY control index = i + D*j
    angles = []
    for j in range(D):
        for i in range(D):
            angles.append(2 * np.arccos(np.clip(a[i, j], -1, 1)))
    qc = QuantumCircuit(1 + 2 * n, name="BE")
    f, R1, R2 = 0, list(range(1, 1 + n)), list(range(1 + n, 1 + 2 * n))
    # P_c
    for q in R1[:m]:
        qc.h(q)
    qc.append(UnitaryGate(_perm_unitary(perm_c), label="O_c"), R1 + R2)
    # O_A
    qc.append(UCRYGate(angles), [f] + R1 + R2)
    # P_r^dagger
    qc.append(UnitaryGate(_perm_unitary(perm_r).T, label="O_r^T"), R1 + R2)
    for q in R2[:m]:
        qc.h(q)
    # SWAP(R1, R2)
    for q1, q2 in zip(R1, R2):
        qc.swap(q1, q2)
    alpha = s * amax
    if return_info:
        return qc, dict(n=n, s=s, alpha=float(alpha), qubits=1 + 2 * n)
    return qc


def extract_block(qc, n, N=None):
    """Top-left block <0_f 0_R1 i | U | 0_f 0_R1 j> of a block-encoding circuit (system = top n qubits)."""
    from qiskit.quantum_info import Operator
    U = Operator(qc).data
    D = 2 ** n
    # basis index = f + 2*R1 + 2*D*R2  (qubit 0 = f, then R1, then R2)
    idx = [0 + 2 * 0 + 2 * D * i for i in range(D)]
    B = U[np.ix_(idx, idx)]
    return B if N is None else B[:N, :N]


# ----------------------------------------------------------------------------------------------
# 2. unitary dilation of the certified evolution
# ----------------------------------------------------------------------------------------------
def dilation_unitary(V):
    V = np.asarray(V)
    I = np.eye(V.shape[0])
    A = np.real_if_close(sqrtm(I - V @ V.conj().T))
    B = np.real_if_close(sqrtm(I - V.conj().T @ V))
    U = np.block([[V, A], [B, -V.conj().T]])
    return np.real_if_close(U)


def dilation_circuit(V, v0):
    """Circuit: prepare |v0>, apply the dilation; ancilla is the TOP qubit.  Data = low n qubits."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate, StatePreparation
    Vp, n = _pad_pow2(np.asarray(V, float))
    if Vp.shape != np.asarray(V).shape:
        Vp[np.arange(V.shape[0], 2 ** n), np.arange(V.shape[0], 2 ** n)] = 0.0
    U = dilation_unitary(Vp)
    err = np.abs(U @ U.T - np.eye(len(U))).max()
    vv = np.zeros(2 ** n); vv[:len(v0)] = v0
    vv = vv / np.linalg.norm(vv)
    qc = QuantumCircuit(n + 1)
    qc.append(StatePreparation(vv), list(range(n)))
    # block [[V, A],[B, -V^T]]: top-left block = ancilla |0>; ancilla is the highest qubit (index n)
    qc.append(UnitaryGate(U, label="U_dil"), list(range(n + 1)))
    return qc, dict(n_data=n, unitarity_error=float(err), norm_V=float(np.linalg.norm(Vp, 2)))


def ideal_distribution(V, v0):
    """Probabilities of (ancilla = 0, data = k) for the dilation circuit."""
    Vp, n = _pad_pow2(np.asarray(V, float))
    vv = np.zeros(2 ** n); vv[:len(v0)] = v0
    vv = vv / np.linalg.norm(vv)
    out = Vp @ vv
    return np.abs(out) ** 2
