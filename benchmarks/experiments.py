"""
Dissipativity, accuracy and normalization experiments.  Run:  python3 experiments.py
Writes results/results.json.

E1  Discrete LK-SBP lemma: is  W G + G^T W <= 0  for certified systems?  (and how bad is the
    unweighted Euclidean Hermitian part that plain LCHS/Schrodingerization would see)
E2  Accuracy: convergence of the semi-discrete generator (exact time evolution via expm,
    i.e. exactly what LCHS implements) for smooth vs. incompatible history data.
E3  Block-encoding normalisation alpha_W = ||W^1/2 G W^-1/2|| vs. p and element size.
E5  Limits: a memory-stabilised system (A unstable) that the delay-independent LMI cannot certify.
"""
import json, os, sys, time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfde.core import *

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)
R = {}

# ----------------------------------------------------------------------------------------
# Test problems
# ----------------------------------------------------------------------------------------
# P1: scalar, two incommensurate delays, oscillatory, delay-independent stable (mu + sum|b| = -0.1)
P1 = LinearDDE(A=[[-2.0]], delays=[1.0, 1.7], Bs=[[[-1.2]], [[-0.7]]])


def complex_root(a, bs, taus, guess):
    def F(v):
        lam = v[0] + 1j * v[1]
        val = lam - a - sum(b * np.exp(-lam * t) for b, t in zip(bs, taus))
        return [val.real, val.imag]
    v = fsolve(F, guess, xtol=1e-15)
    return v[0] + 1j * v[1]


# smooth oscillatory exact solution x(t) = Re(exp(lam t)) from a complex characteristic root
def rightmost_root(a, bs, taus):
    best = None
    for gr in np.linspace(-2, 0.5, 6):
        for gi in np.linspace(0.2, 8, 25):
            lam = complex_root(a, bs, taus, [gr, gi])
            res = lam - a - sum(b * np.exp(-lam * t) for b, t in zip(bs, taus))
            if abs(res) < 1e-10 and abs(lam.imag) > 1e-6 and (best is None or lam.real > best.real + 1e-9):
                best = lam
    return best


lam_c = rightmost_root(-2.0, [-1.2, -0.7], [1.0, 1.7])
smooth_phi = lambda t: np.array([np.real(np.exp(lam_c * t))])
smooth_exact = lambda t: np.real(np.exp(lam_c * t))
# incompatible history: constant (x'(0+) != phi'(0-), creates moving kinks / breaking points)
const_phi = lambda t: np.array([1.0])

# P2: 2x2 non-normal A with mu(A) > 0 (NOT dissipative in the Euclidean norm), stable for all delays
P2 = LinearDDE(A=[[-1.0, 6.0], [0.0, -2.0]], delays=[0.5, 1.0],
               Bs=[[[0.0, 0.0], [0.15, 0.0]], [[0.2, 0.0], [0.0, 0.2]]])


def laplacian_dde(g=32, nu=0.02, c=0.3, b1=0.12, b2=0.1):
    """Delay reaction-diffusion u_t = nu u_xx - c u + b1 u(t-1) + b2 S u(t-0.6) on (0,1),
    Dirichlet, g interior points; S = shift (non-local coupling)."""
    hx = 1.0 / (g + 1)
    Lap = (np.diag(-2 * np.ones(g)) + np.diag(np.ones(g - 1), 1) + np.diag(np.ones(g - 1), -1)) / hx**2
    A = nu * Lap - c * np.eye(g)
    S = np.diag(np.ones(g - 1), -1)
    return LinearDDE(A=A, delays=[0.6, 1.0], Bs=[b2 * S, b1 * np.eye(g)])


# ----------------------------------------------------------------------------------------
# E1: lemma check
# ----------------------------------------------------------------------------------------
def e1():
    rows = []
    # (a) random scalar-certified systems
    for trial in range(40):
        g = int(rng.integers(1, 6)); m = int(rng.integers(1, 5))
        M = rng.normal(size=(g, g)); A = M - M.T          # skew part
        A = A - (1.0 + rng.random() * 2) * np.eye(g)     # mu(A) = -(1..3)
        muA = sym_max_eig(A)
        budget = -muA * rng.uniform(0.3, 1.0)            # sum ||B|| <= -mu(A)
        Bs, taus = [], sorted(rng.uniform(0.2, 2.0, size=m))
        share = rng.dirichlet(np.ones(m)) * budget
        for s in share:
            B = rng.normal(size=(g, g)); B *= s / np.linalg.norm(B, 2); Bs.append(B)
        dde = LinearDDE(A=A, delays=list(taus), Bs=Bs)
        ok, P, Qs, val = scalar_certificate(dde)
        p = int(rng.integers(2, 9)); h = float(rng.uniform(0.15, 0.6))
        disc = discretize(dde, p, h); G = generator(disc); W = lk_weight(disc, P, Qs)
        rows.append(dict(kind="scalar-cert random", g=g, m=m, p=p, n=disc.size,
                         certified=bool(ok), max_sym_WG=sym_max_eig(W @ G),
                         max_sym_G=sym_max_eig(G)))
    # (b) P2: LMI-certified, Euclidean non-dissipative A
    okc, P, Qs, st = lmi_certificate(P2)
    lkf = np.max(np.linalg.eigvalsh(lk_form(P2, P, Qs)))
    for p, h in [(3, 0.25), (6, 0.25), (8, 0.1)]:
        disc = discretize(P2, p, h); G = generator(disc); W = lk_weight(disc, P, Qs)
        rows.append(dict(kind="P2 LMI-cert (mu(A)>0)", g=2, m=2, p=p, n=disc.size, certified=bool(okc),
                         max_sym_WG=sym_max_eig(W @ G), max_sym_G=sym_max_eig(G),
                         mu_A=sym_max_eig(P2.A), cond_P=float(np.linalg.cond(P)), lk_form_max=float(lkf)))
        # identity P cannot certify P2
    okI, _, _, valI = scalar_certificate(P2)
    # (c) high-dimensional sparse delay reaction-diffusion
    dde = laplacian_dde(g=32)
    ok, P, Qs, val = scalar_certificate(dde)
    for p, h in [(4, 0.2), (8, 0.2)]:
        disc = discretize(dde, p, h); G = generator(disc); W = lk_weight(disc, P, Qs)
        rows.append(dict(kind="delay reaction-diffusion g=32", g=32, m=2, p=p, n=disc.size,
                         certified=bool(ok), max_sym_WG=sym_max_eig(W @ G), max_sym_G=sym_max_eig(G)))
    worst = max(r["max_sym_WG"] for r in rows)
    R["E1"] = dict(rows=rows, worst_weighted=worst, P2_identity_certificate=bool(okI),
                   P2_identity_value=float(valI))
    print(f"E1: {len(rows)} systems, worst max eig of sym(W G) = {worst:.2e}; "
          f"unweighted max eig range = [{min(r['max_sym_G'] for r in rows):.2f}, "
          f"{max(r['max_sym_G'] for r in rows):.2f}]")


# ----------------------------------------------------------------------------------------
# E2: accuracy
# ----------------------------------------------------------------------------------------
SAMPLES = np.arange(0.5, 10.0 + 1e-9, 0.5)      # output times for trajectory-wide error


def traj_error(dde, p, h, phi, ref_vals, P=None, Qs=None):
    """Max over sample times of |x_num - x_ref|, normalised by max |x_ref| (relative trajectory error).
    Time evolution is exact (matrix exponential), so this isolates the history discretisation error,
    i.e. exactly the error an exact LCHS / Hamiltonian-simulation step would inherit."""
    disc = discretize(dde, p, h); G = generator(disc)
    z = initial_state(disc, phi); z0 = z.copy()
    dt = SAMPLES[0]; E = expm(G * dt)
    xs = []
    for _ in SAMPLES:
        z = E @ z; xs.append(z[0])
    xs = np.array(xs)
    err = float(np.max(np.abs(xs - ref_vals)) / np.max(np.abs(ref_vals)))
    return err, disc, G, z0, z


def e2():
    ref_c = reference_solution(P1, const_phi, SAMPLES[-1])
    REF_C = np.array([ref_c(t)[0] for t in SAMPLES])
    REF_S = np.array([smooth_exact(t) for t in SAMPLES])
    out = {"lambda": [lam_c.real, lam_c.imag], "T": float(SAMPLES[-1]), "p_sweep": [], "h_sweep": []}
    for p in [2, 3, 4, 5, 6, 8, 10, 12, 14]:
        es, *_ = traj_error(P1, p, 0.35, smooth_phi, REF_S)
        ec, *_ = traj_error(P1, p, 0.35, const_phi, REF_C)
        out["p_sweep"].append(dict(p=p, err_smooth=es, err_incompat=ec))
    for nel in [1, 2, 4, 8, 16, 32, 64]:
        h = 0.7 / nel
        for p in [2, 4, 8]:
            ec, disc, *_ = traj_error(P1, p, h, const_phi, REF_C)
            out["h_sweep"].append(dict(p=p, h=h, dof=disc.n_nodes, err_incompat=ec))
    for p in [2, 4, 8]:
        pts = [(r["h"], r["err_incompat"]) for r in out["h_sweep"] if r["p"] == p and r["err_incompat"] > 1e-11]
        hs, es = np.log([a for a, _ in pts[-3:]]), np.log([b for _, b in pts[-3:]])
        out[f"order_incompat_p{p}"] = float(np.polyfit(hs, es, 1)[0])
    R["E2"] = out
    print("E2: lambda =", lam_c)
    print("    smooth   err p=2..14:", ["%.1e" % r["err_smooth"] for r in out["p_sweep"]])
    print("    incompat err p=2..14:", ["%.1e" % r["err_incompat"] for r in out["p_sweep"]])
    print("    incompat h-sweep:", [(r["p"], round(r["h"], 4), "%.1e" % r["err_incompat"]) for r in out["h_sweep"]])
    print("    incompat observed h-orders: p2 %.2f, p4 %.2f, p8 %.2f" % (
        out["order_incompat_p2"], out["order_incompat_p4"], out["order_incompat_p8"]))


# ----------------------------------------------------------------------------------------
# E3: alpha scaling
# ----------------------------------------------------------------------------------------
def e3():
    ok, P, Qs, _ = scalar_certificate(P1)
    rows = []
    for p in [2, 4, 6, 8, 12, 16]:
        for h in [0.85, 0.35, 0.175]:
            disc = discretize(P1, p, h); G = generator(disc); W = lk_weight(disc, P, Qs)
            Gw, _ = weighted_generator(G, W)
            hmin = min(e.b - e.a for e in disc.elements)
            rows.append(dict(p=p, h=hmin, alpha_W=float(np.linalg.norm(Gw, 2)),
                             p2_over_h=p * p / hmin))
    x = np.log([r["p2_over_h"] for r in rows]); y = np.log([r["alpha_W"] for r in rows])
    slope, icpt = np.polyfit(x, y, 1)
    R["E3"] = dict(rows=rows, loglog_slope_vs_p2_over_h=float(slope), prefactor=float(np.exp(icpt)))
    print(f"E3: alpha_W ~ {np.exp(icpt):.2f} * (p^2/h)^{slope:.2f}")


# ----------------------------------------------------------------------------------------
# E5: limits - memory-stabilised system
# ----------------------------------------------------------------------------------------
def e5():
    # x' = 0.2 x - 1.0 x(t - 0.5): A unstable alone, but the DDE is asymptotically stable
    dde = LinearDDE(A=[[0.2]], delays=[0.5], Bs=[[[-1.0]]])
    # rightmost characteristic root
    best = -np.inf
    for gr in [0.1, -0.1]:
        for gi in np.linspace(0.1, 12, 40):
            lam = complex_root(0.2, [-1.0], [0.5], [gr, gi])
            val = lam - 0.2 + 1.0 * np.exp(-lam * 0.5)
            if abs(val) < 1e-9:
                best = max(best, lam.real)
    for gr in np.linspace(-3, 1, 9):
        try:
            lr = brentq(lambda l: l - 0.2 + np.exp(-0.5 * l), gr, gr + 0.5)
            best = max(best, lr)
        except ValueError:
            pass
    okc, P, Qs, st = lmi_certificate(dde)
    R["E5"] = dict(rightmost_root_real=float(best), lmi_feasible=bool(okc), lmi_status=st)
    print(f"E5: memory-stabilised DDE: rightmost Re(lambda)={best:.3f} (stable), delay-independent LMI feasible={okc}")


if __name__ == "__main__":
    t0 = time.time()
    e1(); e2(); e3(); e5()

    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.bool_): return bool(o)
        return o
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(clean(R), f, indent=1)
    print(f"done in {time.time() - t0:.1f}s")
