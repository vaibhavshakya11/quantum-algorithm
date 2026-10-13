"""
Phase-1b: the GENERAL linear FDE class.  Run:  python3 general_tests.py
Writes results/general.json.

G1  neutral:            x'(t) - 0.3 x'(t-0.5) = -2 x + 0.6 x(t-1)
G2  distributed kernel: x' = -2 x + 0.4 x(t-0.8) + int_0^2 k(s) x(t-s) ds,  k = Gaussian bump
G3  time-varying:       x' = A(t) x + 0.6 x(t - d(t)) + f(t),  d(t) = 1 + 0.3 sin t,  A(t) = -2 + 0.5 cos t
G4  everything, g = 2:  time-varying A(t), constant delay, neutral term, matrix Gaussian kernel with
                        time-varying strength, time-varying delay and forcing -- all at once.

For each:  (a) LK certificate (robust LMI over a time grid, window weight beta searched);
           (b) lemma check:  max_t lambda_max(sym(W G(t))) <= 0, vs. the unweighted value;
           (c) accuracy vs. degree p on a MANUFACTURED smooth solution (forcing chosen so a given
               smooth x*(t) is exact): isolates the history-discretisation error;
           (d) a non-manufactured run (constant history => breaking points) vs. an independent
               method-of-steps reference (kernels: high-order real/complex SOE twin, fit err ~1e-9).
Also: distributed kernels via complex sum-of-exponentials ("Markovianisation") -- certifiable or not?
"""
import json, os, sys, time
import numpy as np
from numpy.polynomial import chebyshev as Ch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfde.fde import *

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)
RES = {}
SQ, WQ = np.polynomial.legendre.leggauss(60)


def sym_max(M):
    return float(np.max(np.linalg.eigvalsh(0.5 * (M + M.T))))


def integral(fun, a, b, nsub=8):
    acc = 0
    for m in range(nsub):
        lo, hi = a + m * (b - a) / nsub, a + (m + 1) * (b - a) / nsub
        ss = (SQ + 1) * (hi - lo) / 2 + lo; ws = WQ * (hi - lo) / 2
        for sk, wk in zip(ss, ws):
            acc = acc + wk * fun(sk)
    return acc


# ----------------------------------------------------------------------------------------
# manufactured solutions in the normal form
# ----------------------------------------------------------------------------------------
def manufacture(fde, xs, dxs, aux=()):
    """phi = x*;  forcing f chosen so that the smooth x* solves the normal form exactly."""
    g, n = fde.g, fde.n
    neutral = [(c.d, c.C) for c in fde.const if c.C is not None]

    def xi_star(t):
        v = np.zeros(n)
        w = xs(t).copy()
        for d, C in neutral:
            w -= C @ xs(t - d)
        v[:g] = w
        for j, (lam, r) in enumerate(aux):
            y = integral(lambda s: np.exp(-lam * s) * xs(t - s), 0, r)
            v[g + 2 * g * j: g + 2 * g * j + g] = y.real
            v[g + 2 * g * j + g: g + 2 * g * (j + 1)] = y.imag
        return v

    def dxi_star(t):
        v = np.zeros(n)
        dw = dxs(t).copy()
        for d, C in neutral:
            dw -= C @ dxs(t - d)
        v[:g] = dw
        for j, (lam, r) in enumerate(aux):
            y = integral(lambda s: np.exp(-lam * s) * xs(t - s), 0, r)
            dy = xs(t) - lam * y - np.exp(-lam * r) * xs(t - r)
            v[g + 2 * g * j: g + 2 * g * j + g] = dy.real
            v[g + 2 * g * j + g: g + 2 * g * (j + 1)] = dy.imag
        return v

    def f_exact(t):
        out = dxi_star(t) - fde.A(t) @ xi_star(t)
        for c in fde.const:
            out -= c.B(t) @ xs(t - c.d)
        for vd in fde.var:
            out -= vd.B(t) @ xs(t - vd.d(t))
        for qk in fde.qk:
            out -= qk.K(t) @ integral(lambda s: qk.k(s) * xs(t - s), 0, qk.r)
        return out

    # f is smooth: piecewise Chebyshev interpolant (degree 40 per unit interval), error checked
    T0, T1 = fde.t_range
    pieces = []
    edges = np.arange(T0, T1 + 1.0, 1.0)
    k = np.arange(41); xc = np.cos(np.pi * (k + 0.5) / 41)
    for a, b in zip(edges[:-1], edges[1:]):
        tc = (a + b) / 2 + (b - a) / 2 * xc
        pieces.append((a, b, Ch.chebfit(xc, np.array([f_exact(t) for t in tc]), 40)))

    def f(t):
        for a, b, coef in pieces:
            if t <= b + 1e-12 or b == pieces[-1][1]:
                return Ch.chebval((2 * t - a - b) / (b - a), coef)
    err = max(float(np.max(np.abs(f(t) - f_exact(t)))) for t in np.linspace(T0 + 0.013, T1 - 0.017, 7))
    RES.setdefault("forcing_interp_err", []).append(err)
    fde.f = f
    fde.phi = lambda t: xs(t)
    fde.xi0 = xi_star(0.0)
    return fde


# ----------------------------------------------------------------------------------------
# problems  (mode = "quad": kernels by quadrature on the window;  "soe": Markovianised twin)
# ----------------------------------------------------------------------------------------
gauss = lambda s: 0.5 * np.exp(-((s - 1.0) / 0.35) ** 2)
gauss15 = lambda s: 0.6 * np.exp(-((s - 0.75) / 0.3) ** 2)


def problem(name, mode="quad", J=8):
    if name == "G1":
        return build_fde(A=[[-2.0]], g=1, phi=lambda t: np.array([1.0]),
                         delays=[(1.0, [[0.6]])], neutral=[(0.5, [[0.3]])], t_range=(0, 10)), []
    if name == "G2":
        kw = dict(A=[[-2.0]], g=1, phi=lambda t: np.array([1.0]), delays=[(0.8, [[0.4]])], t_range=(0, 10))
        if mode == "quad":
            return build_fde(quad_kernels=[dict(r=2.0, k=gauss, K=[[1.0]])], **kw), []
        lam, c, ferr = soe_fit(gauss, 2.0, J)
        return build_fde(kernels=[dict(r=2.0, lam=lam, c=c, K=[[1.0]])], **kw), [(z, 2.0) for z in lam]
    if name == "G3":
        return build_fde(A=lambda t: [[-2.0 + 0.5 * np.cos(t)]], g=1, phi=lambda t: np.array([1.0]),
                         var_delays=[(lambda t: 1.0 + 0.3 * np.sin(t), lambda t: 0.3 * np.cos(t), [[0.6]])],
                         f=lambda t: np.array([0.5 * np.sin(2 * t)]), t_range=(0, 10)), []
    if name == "G4":
        A = lambda t: np.array([[-3.0 + 0.4 * np.cos(t), 0.8], [-0.8, -2.5]])
        Kmat = np.array([[0.3, 0.0], [0.1, 0.2]])
        coef = lambda t: 1.0 + 0.3 * np.sin(0.7 * t)          # time-varying kernel strength
        kw = dict(A=A, g=2, phi=lambda t: np.array([1.0, -0.5]),
                  delays=[(0.7, np.array([[0.3, 0.1], [0.0, 0.25]]))],
                  neutral=[(0.4, np.array([[0.2, 0.0], [0.05, 0.15]]))],
                  var_delays=[(lambda t: 0.9 + 0.2 * np.sin(1.3 * t), lambda t: 0.26 * np.cos(1.3 * t),
                               np.array([[0.2, 0.0], [0.0, 0.2]]))],
                  f=lambda t: np.array([0.3 * np.sin(t), 0.2 * np.cos(2 * t)]), t_range=(0, 8))
        if mode == "quad":
            return build_fde(quad_kernels=[dict(r=1.5, k=gauss15, K=Kmat, coef=coef)], **kw), []
        lam, c, ferr = soe_fit(gauss15, 1.5, J)
        return build_fde(kernels=[dict(r=1.5, lam=lam, c=c, K=Kmat, coef=coef)], **kw), [(z, 1.5) for z in lam]


def x_star(g):
    if g == 1:
        return (lambda t: np.array([np.sin(1.3 * t) + 0.5 * np.cos(0.7 * t)]),
                lambda t: np.array([1.3 * np.cos(1.3 * t) - 0.35 * np.sin(0.7 * t)]))
    return (lambda t: np.array([np.sin(1.3 * t) + 0.5, np.cos(0.9 * t) - 0.3 * np.sin(2 * t)]),
            lambda t: np.array([1.3 * np.cos(1.3 * t), -0.9 * np.sin(0.9 * t) - 0.6 * np.cos(2 * t)]))


def run(name):
    out = {}
    fde, _ = problem(name)
    T = fde.t_range[1]
    gam = discretize_general(fde, 2, 0.5).gammas
    cert = certify(fde, gam)
    out["certified"] = cert["ok"]
    beta = cert.get("beta", 0.0)
    if cert["ok"]:
        out.update(beta=beta, cond_P=cert["cond_P"], lmi_status=cert["status"])
    # (b) lemma check across a time grid and several discretisations
    lem = []
    if cert["ok"]:
        for p, h in [(3, 0.3), (6, 0.3), (8, 0.15)]:
            disc = discretize_general(fde, p, h, hs=0.25, beta=beta)
            W = weight_general(disc, cert["P"], cert["Qc"], cert["Qv"], cert["S0"])
            G_of = make_generator(disc)
            wmax, umax = -np.inf, -np.inf
            for t in np.linspace(*fde.t_range, 41):
                G, _ = G_of(t)
                wmax = max(wmax, sym_max(W @ G)); umax = max(umax, sym_max(G))
            lem.append(dict(p=p, h=h, size=disc.size, max_sym_WG=wmax, max_sym_G=umax))
    out["lemma"] = lem
    # (c) manufactured smooth solution
    xs, dxs = x_star(fde.g)
    fde_m, aux_m = problem(name)
    manufacture(fde_m, xs, dxs, aux_m)
    tout = np.linspace(0.5, T, 16)
    exact = np.array([xs(t) for t in tout])
    conv = []
    for p in [2, 3, 4, 5, 6, 8, 10]:
        disc = discretize_general(fde_m, p, 0.3, hs=0.25, beta=beta)
        xnum, _ = solve_general(disc, T, tout)
        conv.append(dict(p=p, size=disc.size, err=float(np.max(np.abs(xnum - exact)) / np.max(np.abs(exact)))))
    out["manufactured_convergence"] = conv
    # (d) constant history vs independent reference (kernels: SOE twin)
    fde_ref, _ = problem(name, mode="soe") if fde.qk else problem(name)
    ref = reference_general(fde_ref, T)
    xr = np.array([ref(t) for t in tout])
    agree = []
    for p, h in [(4, 0.15), (8, 0.15), (8, 0.075)]:
        disc = discretize_general(fde, p, h, hs=0.125, beta=beta)
        xnum, _ = solve_general(disc, T, tout)
        agree.append(dict(p=p, h=h, err=float(np.max(np.abs(xnum - xr)) / np.max(np.abs(xr)))))
    out["vs_reference"] = agree
    RES[name] = out
    lm = max((r["max_sym_WG"] for r in lem), default=float("nan"))
    um = max((r["max_sym_G"] for r in lem), default=float("nan"))
    print(f"[{time.strftime('%X')}] {name}: certified={cert['ok']} beta={beta}  max sym(WG)={lm:.1e}  "
          f"(unweighted {um:.1f})", flush=True)
    print("    manufactured err vs p:", ["%d:%.1e" % (r["p"], r["err"]) for r in conv])
    print("    forcing interpolation err: %.1e" % RES["forcing_interp_err"][-1])
    print("    vs reference (const history):", ["p%d,h%.3f:%.1e" % (r["p"], r["h"], r["err"]) for r in agree],
          flush=True)


def soe_study():
    """Can complex-SOE Markovianisation be certified?  Compare memory budgets."""
    rows = []
    for J in [2, 4, 6, 8]:
        lam, c, e = soe_fit(gauss, 2.0, J)
        budget = float(sum(abs(cc) / l.real if l.real > 0 else np.inf for l, cc in zip(lam, c)))
        fde, _ = problem("G2", mode="soe", J=J)
        cert = lk_certificate_general(fde, [], beta=0.0)
        rows.append(dict(J=J, terms=len(lam), fit_max_err=e, min_re_lam=float(min(l.real for l in lam)),
                         sum_abs_c_over_re_lam=budget, certified=bool(cert[0])))
    l1 = float(integral(lambda s: abs(gauss(s)), 0, 2.0))
    RES["soe_study"] = dict(rows=rows, kernel_L1=l1)
    print("SOE (complex Markovianisation) of the Gaussian bump on [0,2]:  int|k| = %.3f" % l1)
    for r in rows:
        print("    J=%d fit=%.1e  min Re(lam)=%.3g  sum|c|/Re(lam)=%.3g  certified=%s" % (
            r["J"], r["fit_max_err"], r["min_re_lam"], r["sum_abs_c_over_re_lam"], r["certified"]))


if __name__ == "__main__":
    t0 = time.time()
    soe_study()
    for name in sys.argv[1:] or ["G1", "G2", "G3", "G4"]:
        run(name)
    path = os.path.join(OUT, "general.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(RES)
    with open(path, "w") as f:
        json.dump(old, f, indent=1, default=float)
    print(f"done in {time.time() - t0:.0f}s")
