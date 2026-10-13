"""qfde: a quantum algorithm for linear integro-differential and delay differential equations,
built on Lyapunov-Krasovskii-preserving summation-by-parts spectral discretization.

Public API
----------
Problem building     : build_fde (delays, neutral, kernels, time-varying delays/coefficients, forcing)
LK functional        : certify (robust Lyapunov-Krasovskii LMI; kernel condition Gamma >= Gamma_h enforced),
                       check_gamma,
                       verify.validate (validated for all t)
Discretization       : discretize_general, make_generator, weight_general (dissipative by the main theorem)
Quantum algorithm    : lchs.emulate / lchs.query_count (improved LCHS), lchs_circuit (gate-level LCHS), circuits, ftqc
References           : reference_general (method of steps)
Costs of prior methods: resources (QLSA for first-order schemes, Schroedingerization)
"""
from .fde import (build_fde, GeneralFDE, ConstDelay, VarDelay, discretize_general, make_generator,
                  initial_state_general, weight_general, certify, lk_certificate_general,
                  reference_general, solve_general, soe_fit, x_in_row)
from . import core, lchs, resources, verify

__version__ = "0.5.1"
