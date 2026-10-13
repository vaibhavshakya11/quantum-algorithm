"""
Cost models with explicit constants, used for every method compared in the benchmarks.

QLSA (optimal scaling):  Jennings, Lostaglio, Pallister, Sornborger, Subasi, arXiv 2305.11352:
    expected block-encoding queries <= 837 kappa at eps = 1e-10 (Hermitian matrices).  We scale the
    log factor:  Q_QLSA(kappa, eps) = (837 / ln 1e10) * ln(1/eps) * kappa_eff,
    kappa_eff = kappa * (alpha / ||L||)   (a block encoding with normalisation alpha >= ||L||).

LCHS (improved kernel):  see qfde.lchs.query_count.

Schroedingerisation (Setoyama-Fujii, arXiv 2603.17941, their stated query bound with constant 1):
    Q_S = g0 * ( s * T * ||C||_max / eps + ln(1/eps) ).
"""
from __future__ import annotations

import numpy as np

QLSA_C = 837.0 / np.log(1e10)          # ~36.35 per unit ln(1/eps)


def qlsa_queries(kappa, eps, alpha_over_norm=1.0, amp=1.0):
    return float(QLSA_C * np.log(1.0 / eps) * kappa * alpha_over_norm * amp)


def schrodingerisation_queries(s, T, Cmax, eps, g0=1.0):
    return float(g0 * (s * T * Cmax / eps + np.log(1.0 / eps)))


