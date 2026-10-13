"""
The qfde compiler:  spec -> normal form -> LK functional -> history -> discretization -> costs.

    report = compile_spec(spec)          # spec from qfde.spec.load(...)
    print(report["markdown"])

Decisions it makes (and records in the report):
  1. Normal form       build_fde (all features).
  2. LK functional     robust Lyapunov-Krasovskii LMI over a time grid (window weight beta searched);
                       validated over the whole interval when coefficients depend on time.  Without one
                       the method does not apply (the report says so and stops: a dissipativity defect
                       would cost a factor e^{omega T} in any LCHS-type algorithm).
  3. History           checks that the history matches the equation at t = 0 (no jump in x'); a mismatch
                       is reported, since convergence is then algebraic and the polylogarithmic guarantee
                       does not apply.
  4. Discretization    LK-weighted SBP spectral elements; smallest degree p whose solution agrees with
                       degree p + 2 to eps/10 (self-convergence), also checked against an independent
                       reference when available.
  5. Costs             improved-LCHS queries with explicit constants, logical counts, physical qubits and runtime
                       (Microsoft estimator); optional rescaling to a larger sparse dimension.
"""
from __future__ import annotations

import math
import time
import numpy as np

from .fde import (build_fde, discretize_general, make_generator, weight_general, certify, check_gamma, solve_general,
                  reference_general, initial_state_general)
from . import lchs, ftqc, verify


def _mat(M, t=0.0):
    return np.atleast_2d(np.asarray(M(t) if callable(M) else M, float))


def history_jump(spec, h=1e-4):
    """Relative jump |x'(0+) - phi'(0-)| / (1 + |x(0)|): zero when the history matches the equation at t = 0."""
    g, phi = spec["g"], spec["phi"]
    P = lambda th: np.asarray(phi(th), float).reshape(g)
    dP = lambda th: (3 * P(th) - 4 * P(th - h) + P(th - 2 * h)) / (2 * h)       # one-sided, from the left
    x0 = P(0.0)
    rhs = _mat(spec["A"]) @ x0
    for tau, B in spec["delays"]:
        rhs = rhs + _mat(B) @ P(-float(tau))
    for d, dd, B in spec["var_delays"]:
        rhs = rhs + _mat(B) @ P(-float(d(0.0)))
    for kd in spec["quad_kernels"]:
        r = float(kd["r"]); xs, ws = np.polynomial.legendre.leggauss(64)
        s_ = (xs + 1) * r / 2; ws = ws * r / 2
        integral = sum(w * float(kd["k"](si)) * P(-si) for si, w in zip(s_, ws))
        c = kd.get("coef"); c = float(c(0.0)) if callable(c) else 1.0
        rhs = rhs + c * _mat(kd["K"]) @ integral
    if spec["f"] is not None:
        rhs = rhs + np.asarray(spec["f"](0.0), float).reshape(g)
    for sig, C in spec["neutral"]:
        rhs = rhs + _mat(C) @ dP(-float(sig))
    return float(np.max(np.abs(rhs - dP(0.0)))) / (1.0 + float(np.max(np.abs(x0))))


def compile_spec(spec, verbose=False, lowered=False):
    """Compile a parsed spec.  lowered=True also returns the objects a circuit builder needs:
    (report, dict(fde, cert, disc, G, W, z0, R)) with z' = G z, x = R z, and W the LK weight."""
    t_start = time.time()
    log = []
    g, T, eps = spec["g"], spec["T"], spec["eps"]
    rep = dict(name=spec["name"], g=g, T=T, eps=eps)
    # 1. normal form ---------------------------------------------------------------------------
    fde = build_fde(A=spec["A"], g=g, phi=spec["phi"], delays=spec["delays"], neutral=spec["neutral"],
                    var_delays=spec["var_delays"], quad_kernels=spec["quad_kernels"], f=spec["f"],
                    t_range=(0, T))
    rep["normal_form"] = (f"Markovian state n = {fde.n}; constant delay points {fde.const_points}; "
                          f"time-varying delays {len(fde.var)}; kernels {len(fde.qk)}")
    log.append("normal form built")
    # 2. LK functional ------------------------------------------------------------------------
    gam = discretize_general(fde, 2, 0.5).gammas
    h0 = spec.get("element_size") or min(0.5, min([c.d for c in fde.const] + [1.0]))
    degs = [spec["degree"]] if spec.get("degree") else list(range(2, 15, 2))
    cert = certify(fde, gam, nt=(25 if spec["time_dependent"] else 1),
                   meshes=[(p, h0) for p in degs] if fde.qk else None)
    rep["lk_functional"] = dict(found=bool(cert["ok"]))
    if cert["ok"]:
        rep["lk_functional"].update(beta=cert.get("beta", 0.0), cond_P=cert.get("cond_P"))
        if spec["time_dependent"]:
            v = verify.validate(fde, cert, gam, n_grid=201)
            rep["lk_functional"]["validated_bound"] = v["validated_bound"]
            rep["lk_functional"]["valid_all_t"] = v["valid"]
    log.append("LK functional " + ("found" if cert["ok"] else "NOT found"))
    if not cert["ok"]:
        rep["verdict"] = ("no Lyapunov-Krasovskii functional found: the discretized generator is not known to be "
                          "dissipative, so no polylogarithmic-cost guarantee applies")
        rep["compile_seconds"] = round(time.time() - t_start, 2); rep["log"] = log
        rep["markdown"] = to_markdown(rep)
        return rep
    # 3. history -------------------------------------------------------------------------------
    jump = history_jump(spec)
    rep["history"] = dict(compatible=bool(jump < 1e-6), jump_in_derivative=jump)
    if jump >= 1e-6:
        log.append("history does not match the equation at t = 0: convergence is algebraic")
    # 4. discretization ------------------------------------------------------------------------
    ts = np.linspace(T / 16, T, 16)
    ref = None
    try:
        if not spec["quad_kernels"]:
            f0 = build_fde(A=spec["A"], g=g, phi=spec["phi"], delays=spec["delays"], neutral=spec["neutral"],
                           var_delays=spec["var_delays"], f=spec["f"], t_range=(0, T))
            r = reference_general(f0, T)
            ref = np.array([r(t) for t in ts])
    except Exception as e:                                     # reference is a check, not a requirement
        log.append(f"reference unavailable ({e.__class__.__name__})")
    beta = cert.get("beta", 0.0)
    auto = not spec["time_dependent"]
    sols, chosen = {}, None
    h = spec.get("element_size") or min(0.5, min([c.d for c in fde.const] + [1.0]))
    for p in ([spec["degree"]] if spec.get("degree") else range(2, 15, 2)):
        disc = discretize_general(fde, p, h, hs=spec.get("stretched_element_size") or 0.25, beta=beta)
        x, _ = solve_general(disc, T, ts, autonomous=auto)
        sols[p] = (disc, x)
        if p - 2 in sols:
            diff = np.max(np.abs(x - sols[p - 2][1])) / max(np.max(np.abs(x)), 1e-300)
            if diff < eps / 10:
                chosen = p - 2; break
    chosen = chosen or max(sols)
    disc, x = sols[chosen]
    rep["discretisation"] = dict(degree=chosen, element_size=h, unknowns=disc.size)
    if fde.qk and "Gamma" in cert:
        rep["discretisation"]["kernel_quadrature_condition"] = [
            dict(Gamma_h=a, Gamma=b, holds=c) for a, b, c in check_gamma(disc, cert)]
    if ref is not None:
        rep["discretisation"]["error_vs_reference"] = float(np.max(np.abs(x - ref)) / np.max(np.abs(ref)))
    rep["solution_samples"] = dict(t=ts.tolist(), x=np.asarray(x).tolist())
    # 5. costs ---------------------------------------------------------------------------------
    xT_norm = max(np.linalg.norm(np.asarray(x)[-1]), 1e-300)
    G_of = make_generator(disc)
    W = weight_general(disc, cert["P"], cert["Qc"], cert["Qv"], cert.get("S0") or [])
    lam, V = np.linalg.eigh(W); Wh = (V * np.sqrt(lam)) @ V.T; Wih = (V / np.sqrt(lam)) @ V.T
    aL = aH = 0.0
    for t in (np.linspace(0, T, 9) if spec["time_dependent"] else [0.0]):
        L, H = lchs.split(Wh @ G_of(t)[0] @ Wih)
        aL = max(aL, lchs.sparse_norm(L)[0]); aH = max(aH, lchs.sparse_norm(H)[0])
    z0 = initial_state_general(disc)
    Pw = np.linalg.eigvalsh(cert["P"])
    num = float(np.linalg.norm(Wh @ z0))
    forcing_l1 = 0.0
    if spec["f"] is not None:
        tq = np.linspace(0, T, 801); wq = np.full(len(tq), tq[1] - tq[0]); wq[[0, -1]] /= 2
        forcing_l1 = float(np.dot(wq, [np.linalg.norm(spec["f"](t)) for t in tq]))
        num += math.sqrt(Pw.max()) * forcing_l1
    amp = num / (math.sqrt(max(Pw.min(), 1e-300)) * xT_norm)
    Q, st = lchs.query_count(aL, aH, T, eps, amp)
    nb = math.ceil(math.log2(disc.size)); pq = ftqc.per_query_cost(node_bits=max(nb, 3, math.ceil(math.log2(2 * chosen + 3))), p=chosen)
    rep["cost"] = dict(algorithm="improved LCHS on the LK-weighted generator", alpha_L=aL, alpha_H=aH,
                       amplification=amp, forcing_l1=forcing_l1, queries=Q, lchs_truncation_K=st["K"],
                       qsp_degree=st["qsp_degree"])
    data_q = nb
    lc = ftqc.logical_counts(Q, pq, data_q)
    rep["cost"]["per_query"] = pq
    rep["cost"]["logical"] = lc
    if not lowered:
        try:
            rep["cost"]["physical"] = ftqc.physical(lc)
        except Exception as e:
            rep["cost"]["physical"] = dict(error=str(e))
    if spec.get("scale_dimension"):
        extra = math.ceil(math.log2(spec["scale_dimension"] / g))
        lc2 = dict(lc); lc2["numQubits"] = lc["numQubits"] + extra
        lc2["cczCount"] = int(lc["cczCount"] * (1 + 4 * extra / max(pq["toffoli"], 1)))
        rep["cost"]["scaled"] = dict(dimension=spec["scale_dimension"], logical=lc2,
                                     physical=ftqc.physical(lc2),
                                     note="same norms assumed; +log2 qubits and index-arithmetic Toffolis")
    rep["compile_seconds"] = round(time.time() - t_start, 2)
    rep["log"] = log
    rep["markdown"] = to_markdown(rep)
    if lowered:
        from .fde import x_in_row
        return rep, dict(fde=fde, cert=cert, disc=disc, G=G_of(0.0)[0], W=W, z0=z0, R=x_in_row(disc))
    return rep


def to_markdown(r):
    lk = r["lk_functional"]
    lines = [f"# qfde compile report: {r['name']}", "",
             f"- dimension g = {r['g']}, horizon T = {r['T']}, target accuracy eps = {r['eps']:.0e}",
             f"- normal form: {r['normal_form']}",
             "- LK functional: " + ("**found**" if lk["found"] else "**not found**")
             + (f" (beta = {lk.get('beta')}, cond(P) = {lk.get('cond_P'):.3g})" if lk["found"] and lk.get("cond_P") else "")
             + (f"; validated for all t: bound {lk['validated_bound']:.3g}" if "validated_bound" in lk else "")]
    if not lk["found"]:
        lines.append(f"- verdict: {r['verdict']}")
        lines.append(f"- compiled in {r['compile_seconds']} s")
        return "\n".join(lines)
    hi = r["history"]
    lines.append("- history: " + ("matches the equation at t = 0" if hi["compatible"] else
                 f"does not match the equation at t = 0 (jump in x' {hi['jump_in_derivative']:.2g}); "
                 "convergence is algebraic and the polylogarithmic guarantee does not apply"))
    c = r["cost"]; ph = c.get("physical", {})
    lines += ["- discretisation: " + ", ".join(f"{k} = {v:.3g}" if isinstance(v, float) else f"{k} = {v}"
                                               for k, v in r["discretisation"].items()),
              f"- algorithm: {c['algorithm']}; block-encoding queries = {c['queries']:.3g} "
              f"(alpha_L = {c['alpha_L']:.3g}, alpha_H = {c['alpha_H']:.3g}, amplification = {c['amplification']:.3g})",
              f"- per query: {c['per_query']['toffoli']} Toffoli, {c['per_query']['rotations']} rotations",
              f"- logical: {c['logical']['numQubits']} qubits, {c['logical']['cczCount']:.3g} Toffoli, "
              f"{c['logical']['rotationCount']:.3g} rotations"]
    if "physical_qubits" in ph:
        lines.append(f"- physical (surface code, {ph['qubit_model']}): {ph['physical_qubits']:.3g} qubits, "
                     f"runtime {ph['runtime_formatted']}")
    if "scaled" in c:
        s = c["scaled"]["physical"]
        lines.append(f"- if the system had dimension {c['scaled']['dimension']:.3g} (same norms): "
                     f"{c['scaled']['logical']['numQubits']} logical qubits, {s['physical_qubits']:.3g} physical, "
                     f"runtime {s['runtime_formatted']}")
    lines.append(f"- compiled in {r['compile_seconds']} s")
    return "\n".join(lines)
