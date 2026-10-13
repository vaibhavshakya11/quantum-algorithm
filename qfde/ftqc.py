"""
Fault-tolerant resource estimation.

Pipeline:  per-query logical gate counts (Qualtran primitives)  x  number of queries (qfde.lchs /
qfde.resources)  ->  logical counts  ->  physical qubits and runtime (Microsoft Resource Estimator,
surface code, local, no Azure account).

Per-query block-encoding cost of the certified generator G_hat = [history part] + [A part]:
  * history part  : banded sparse-access block encoding on the node register (bandsize ~ p + 1),
                    Qualtran SparseMatrix(SymmetricBandedRowColumnOracle) + a QROM of the element
                    table ((p+1)^2 entries) loaded and unloaded (the entry oracle);
  * A part        : either a Pauli LCU (L terms; PREPARE + SELECT ~ 3L Toffolis, unary iteration) or a
                    sparse banded oracle on the system register;
  * extra diagonal oracles (e.g. sqrt(D) couplings) : popcount + QROM + one rotation;
  * combination   : one LCU qubit, controlled versions (x2 Toffolis), plus per query one QSP phase and
                    one k-weight rotation for LCHS.
All Toffolis are passed to the estimator as CCZ gates; rotations are synthesised by the estimator.
"""
from __future__ import annotations

import math
import numpy as np

_cache = {}


def _qualtran_banded(bits, band, eps=1e-10):
    key = ("band", bits, band)
    if key not in _cache:
        from qualtran.bloqs.block_encoding.sparse_matrix import (SparseMatrix, SymmetricBandedRowColumnOracle,
                                                                 UniformEntryOracle)
        from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount
        rc = SymmetricBandedRowColumnOracle(bits, bandsize=band)
        be = SparseMatrix(rc, rc, UniformEntryOracle(bits, 0.5), eps=eps)
        c = get_cost_value(be, QECGatesCost())
        _cache[key] = dict(toffoli=int(c.and_bloq + c.toffoli), rotations=int(c.rotation),
                           measurements=int(c.measurement), qubits=int(get_cost_value(be, QubitCount())),
                           ancilla=int(be.ancilla_bitsize))
    return _cache[key]


def _qualtran_qrom(L, bits=20):
    key = ("qrom", L, bits)
    if key not in _cache:
        from qualtran.bloqs.data_loading.qrom import QROM
        from qualtran.resource_counting import get_cost_value, QECGatesCost
        q = QROM.build_from_data(np.arange(L) % 3, target_bitsizes=(bits,))
        c = get_cost_value(q, QECGatesCost())
        _cache[key] = dict(toffoli=int(c.and_bloq + c.toffoli), measurements=int(c.measurement))
    return _cache[key]


def per_query_cost(node_bits, p, system_bits=0, pauli_terms=0, sparse_A_band=0, diag_oracles=0, diag_bits=0,
                   extra_rot=2):
    """Logical cost of ONE controlled query to the block encoding of G_hat (and of its Hermitian /
    anti-Hermitian parts, which have the same structure)."""
    tof = rot = meas = 0
    anc = 0
    if node_bits > 0:
        h = _qualtran_banded(node_bits, p + 1)
        q = _qualtran_qrom((p + 1) ** 2)
        tof += h["toffoli"] + 2 * q["toffoli"]; rot += h["rotations"]; meas += h["measurements"] + 2 * q["measurements"]
        anc = max(anc, h["ancilla"] + 20)
    if pauli_terms > 0:
        tof += 3 * pauli_terms + 4 * math.ceil(math.log2(pauli_terms + 1))
        anc = max(anc, math.ceil(math.log2(pauli_terms + 1)) + 2)
    if sparse_A_band > 0 and system_bits > 0:
        a = _qualtran_banded(system_bits, sparse_A_band)
        tof += a["toffoli"]; rot += a["rotations"]; meas += a["measurements"]
        anc = max(anc, a["ancilla"])
    for _ in range(diag_oracles):
        q = _qualtran_qrom(diag_bits + 1)
        tof += 2 * diag_bits + 2 * q["toffoli"]; rot += 1        # popcount + table + rotation
        anc = max(anc, math.ceil(math.log2(diag_bits + 1)) + 22)
    tof = 2 * tof                                                 # controlled (LCU select) versions
    rot += extra_rot                                              # QSP phase + LCHS k-weight
    return dict(toffoli=int(tof), rotations=int(rot), measurements=int(meas), ancilla=int(anc + 1))


def logical_counts(queries, per_query, data_qubits, lchs_terms=2 ** 16, observable_precision=None):
    """Totals for one prepared solution state (or one observable estimate by amplitude estimation)."""
    reps = 1
    extra_q = 0
    if observable_precision is not None:
        reps = math.ceil(math.pi / (2 * observable_precision))
        extra_q = math.ceil(math.log2(reps)) + 1
    k_bits = math.ceil(math.log2(lchs_terms))
    prep_tof = 2 * lchs_terms                                     # PREPARE over LCHS nodes, per round
    tof = reps * (queries * per_query["toffoli"] + 4 * prep_tof)
    rot = reps * queries * per_query["rotations"]
    qubits = data_qubits + per_query["ancilla"] + k_bits + 2 + 20 + extra_q   # +QSP, AA, phase gradient
    return dict(numQubits=int(qubits), tCount=0, rotationCount=int(rot), rotationDepth=int(rot),
                cczCount=int(tof), ccixCount=0, measurementCount=int(reps * queries * per_query["measurements"]))


def physical(lc, qubit="qubit_gate_ns_e3", error_budget=1e-3):
    """Microsoft Resource Estimator (surface code).  Returns physical qubits, runtime (s), T-factories."""
    from qdk.estimator import LogicalCounts
    r = LogicalCounts(lc).estimate(params={"qubitParams": {"name": qubit}, "errorBudget": error_budget})
    d = r if isinstance(r, dict) else dict(r)
    pc = d["physicalCounts"]; bd = pc["breakdown"]
    return dict(qubit_model=qubit, physical_qubits=int(pc["physicalQubits"]), runtime_s=pc["runtime"] * 1e-9,
                runtime_formatted=d["physicalCountsFormatted"]["runtime"],
                logical_qubits=int(bd["algorithmicLogicalQubits"]), logical_depth=int(bd["logicalDepth"]),
                t_states=int(bd["numTstates"]), code_distance=d["logicalQubit"]["codeDistance"]
                if "logicalQubit" in d else None, t_factories=int(bd["numTfactories"]))
