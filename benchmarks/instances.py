"""
The five comparison instances, all with exact analytic starting histories (the dominant characteristic mode,
which is the state a long-running stable system converges to), so the exact solution is known in closed form.

  Delay equations
    two-delay        x' = -2x - 1.2 x(t-1) - 0.7 x(t-1.7)                                   T = 10
    drug             A1' = -0.8 A1 + 0.4 A2(t-2),  A2' = 0.5 A1 - 0.6 A2  (pharmacology)    T = 24
    baroreflex       P' = -P + 0.5 H,  H' = -0.8 H - 0.6 P(t-0.8)          (physiology)      T = 8
  Integro-differential equations
    gaussian         x' = -2x + 0.4 x(t-0.8) + int_0^2 0.5 e^{-((s-1)/0.35)^2} x(t-s) ds    T = 10
    erlang2, erlang4 x' = -x + 0.8 int_0^inf Erlang_m(mu = 2)(s) x(t-s) ds, memory truncated   T = 10
                     to a window of length r (tail below the target accuracy)

Ours: LK-weighted SBP discretization (one history window) + LCHS.  For every (r,) p, h on a grid we record the
measured error against the exact solution, the LCHS normalizations alpha_L, alpha_H (sparse access; kernel
rows block-encoded as rank-one combinations, normalization = nuclear norm), and the amplitude factor chi.
Prior (delay and Gaussian): first-order time stepping, whose all-at-once linear system the optimal QLSA
solves; errors measured directly by stepping (numba) down to about 1e-6 or as far as feasible; condition
numbers measured exactly up to ~2e5 unknowns and fitted (kappa ~ N^r, r = 1.00) beyond.
Run: python3 benchmarks/instances.py [name ...] -> results/instances.json
"""
import sys, os, json, math, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, warnings
warnings.filterwarnings("ignore")
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import null_space
import mpmath as mpm
from qfde import lchs
from qfde.fde import (build_fde, discretize_general, make_generator, weight_general, certify, check_gamma,
                      solve_general)

OUT = os.path.join(ROOT, "results", "instances.json")
GRID = [(p, h) for p in range(3, 13) for h in (0.5, 0.35, 0.25)]
mpm.mp.dps = 40


# ------------------------------------------------------------------------------------------------ instances
def _root(f, guess):
    return complex(mpm.findroot(f, guess))


def gauss_k(s):
    return 0.5 * np.exp(-((s - 1.0) / 0.35) ** 2)


def erlang_k(m, mu=2.0):
    c = mu ** m / math.factorial(m - 1)
    return lambda s: c * s ** (m - 1) * np.exp(-mu * s)


def spec(name):
    """dict(A, delays, kernels, g, lam, v, T, family, label)."""
    if name == "two-delay":
        A = np.array([[-2.0]]); D = [(1.0, np.array([[-1.2]])), (1.7, np.array([[-0.7]]))]
        lam = _root(lambda l: l + 2 + 1.2 * mpm.e ** (-l) + 0.7 * mpm.e ** (-1.7 * l), mpm.mpc(-0.35, 1.79))
        v = np.array([1.0 + 0j]); T = 10.0; fam = "delay"; label = "Two-delay equation"; K = []
    elif name == "drug":
        A = np.array([[-0.8, 0.0], [0.5, -0.6]]); B = np.array([[0.0, 0.4], [0.0, 0.0]]); D = [(2.0, B)]
        lam = _root(lambda l: (l + 0.8) * (l + 0.6) - 0.2 * mpm.e ** (-2 * l), -0.16)
        v = null_space(lam * np.eye(2) - A - B * np.exp(-2 * lam))[:, 0]; T = 24.0; fam = "delay"
        label = "Drug reabsorption"; K = []
    elif name == "baroreflex":
        A = np.array([[-1.0, 0.5], [0.0, -0.8]]); B = np.array([[0.0, 0.0], [-0.6, 0.0]]); D = [(0.8, B)]
        lam = _root(lambda l: (l + 1) * (l + 0.8) + 0.3 * mpm.e ** (-0.8 * l), mpm.mpc(-0.70, 0.69))
        v = null_space(lam * np.eye(2) - A - B * np.exp(-0.8 * lam))[:, 0]; T = 8.0; fam = "delay"
        label = "Baroreflex"; K = []
    elif name == "gaussian":
        A = np.array([[-2.0]]); D = [(0.8, np.array([[0.4]]))]
        kq = lambda s: 0.5 * mpm.e ** (-((s - 1) / 0.35) ** 2)
        lam = _root(lambda l: l + 2 - 0.4 * mpm.e ** (-0.8 * l) - mpm.quad(lambda s: kq(s) * mpm.e ** (-l * s), [0, 1, 2]), -0.68)
        v = np.array([1.0 + 0j]); T = 10.0; fam = "integro-differential"; label = "Gaussian-kernel equation"
        K = [dict(r=2.0, k=gauss_k, K=[[1.0]])]
    elif name.startswith("erlang"):
        m = int(name[-1]); mu = 2.0
        A = np.array([[-1.0]]); D = []
        lam = _root(lambda l: l + 1 - 0.8 * (mu / (mu + l)) ** m, -0.2)
        v = np.array([1.0 + 0j]); T = 10.0; fam = "integro-differential"; label = f"Erlang-{m} memory"
        K = [dict(r=None, k=(lambda s, m=m: 0.8 * erlang_k(m)(s)), K=[[1.0]], m=m)]
    else:
        raise KeyError(name)
    v = v / v[np.argmax(np.abs(v))]
    return dict(name=name, A=A, delays=D, kernels=K, g=A.shape[0], lam=complex(lam), v=v, T=T, family=fam, label=label)


def exact(S, t):
    return np.real(S["v"] * np.exp(S["lam"] * t))


def build(S, r=None):
    kernels = [dict(kd, r=(r if kd["r"] is None else kd["r"])) for kd in S["kernels"]]
    kernels = [{k: kd[k] for k in ("r", "k", "K")} for kd in kernels]
    phi = lambda th, S=S: exact(S, th)
    return build_fde(A=S["A"], g=S["g"], phi=phi, delays=S["delays"], quad_kernels=kernels, t_range=(0, S["T"]))


def certificate(fde, meshes):
    return certify(fde, [], nt=1, meshes=meshes if fde.qk else None)


def lowered(S, p, h, r=None, cert=None):
    """(disc, G, Gh, Wh, Wih, cert) for one configuration."""
    fde = build(S, r)
    if cert is None:
        cert = certificate(fde, [(p, h)])
    disc = discretize_general(fde, p, h, hs=0.25, beta=cert["beta"])
    W = weight_general(disc, cert["P"], cert["Qc"], cert["Qv"], cert["S0"])
    lw, V = np.linalg.eigh(W); Wh = (V * np.sqrt(lw)) @ V.T; Wih = (V / np.sqrt(lw)) @ V.T
    G = make_generator(disc)(0.0)[0]
    return fde, disc, G, Wh @ G @ Wih, Wh, Wih, cert, W


def lchs_norms(S, p, h, r, cert, beta, Gh, Wh, Wih):
    """alpha_L, alpha_H with the kernel rows block-encoded as a rank-one combination (nuclear norm)."""
    if not S["kernels"]:
        L, H = lchs.split(Gh); return lchs.sparse_norm(L)[0], lchs.sparse_norm(H)[0], lchs.sparse_norm(Gh)[1]
    fz = build(S, r)                                    # same window, kernel switched off
    for q in fz.qk:
        q.K = (lambda t, g=S["g"]: np.zeros((g, g)))
    dz = discretize_general(fz, p, h, hs=0.25, beta=beta)
    Gs = Wh @ make_generator(dz)(0.0)[0] @ Wih
    nuc = float(np.linalg.svd(Gh - Gs, compute_uv=False).sum())
    Ls, Hs = lchs.split(Gs)
    return lchs.sparse_norm(Ls)[0] + nuc, lchs.sparse_norm(Hs)[0] + nuc, lchs.sparse_norm(Gh)[1]


def ours_rows(S, rs=(None,)):
    T = S["T"]; ts = np.linspace(T / 16, T, 16)
    REF = np.array([exact(S, t) for t in ts]); scale = np.max(np.abs(REF))
    env = max(np.linalg.norm(exact(S, t)) for t in np.linspace(0.8 * T, T, 40))
    rows = []
    for r in rs:
        fde = build(S, r)
        cert = certificate(fde, GRID)
        if not cert["ok"]:
            print("   no LK functional for r =", r); continue
        lmin = float(np.min(np.linalg.eigvalsh(cert["P"])))
        for p, h in GRID:
            if r is not None and abs(r / h - round(r / h)) > 1e-9:
                continue
            t0 = time.time()
            fde, disc, G, Gh, Wh, Wih, cert_, W = lowered(S, p, h, r, cert)
            x, z0 = solve_general(disc, T, ts, autonomous=True)
            err = float(np.max(np.abs(np.asarray(x) - REF)) / scale)
            aL, aH, s = lchs_norms(S, p, h, r, cert, disc.beta, Gh, Wh, Wih)
            amp = float(np.linalg.norm(Wh @ z0) / (math.sqrt(lmin) * env))
            lw = np.linalg.eigvalsh(W)
            gam = check_gamma(disc, cert) if fde.qk else []
            rows.append(dict(r=r, p=p, h=h, size=disc.size, err=err, aL=aL, aH=aH, amp=amp, sparsity=s,
                             cond_W=float(lw.max() / lw.min()),
                             lam_weighted=float(np.linalg.eigvalsh((Gh + Gh.T) / 2).max()),
                             lam_unweighted=float(np.linalg.eigvalsh((G + G.T) / 2).max()),
                             gamma_ok=all(ok for *_, ok in gam)))
            print(f"   ours r={r} p={p} h={h}: n={disc.size} err={err:.2e} aL={aL:.0f} amp={amp:.1f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    return rows, dict(decay=float(np.linalg.norm(exact(S, 0.0)) / env), env=env)


# ------------------------------------------------------------------------------------------------ prior
def _numba():
    from numba import njit

    @njit(cache=True)
    def be(Ainv, Bs, ds, hist, N, h, g):
        # Backward Euler: (I - hA) x_n = x_{n-1} + h sum_k B_k x_{n-d_k}; hist[j] = phi((j - H) h), j = 0..H
        H = hist.shape[0] - 1
        L = H + N + 1
        X = np.zeros((L, g))
        X[:H + 1] = hist
        for n in range(1, N + 1):
            rhs = X[H + n - 1].copy()
            for k in range(ds.shape[0]):
                rhs += h * (Bs[k] @ X[H + n - ds[k]])
            X[H + n] = Ainv @ rhs
        return X[H:]

    @njit(cache=True)
    def fe_riemann(hist, N, h, d, kv, a, b):
        # x_{j+1} = x_j + h(a x_j + b x_{j-d} + h sum_{k=1}^R kv[k-1] x_{j-k}); scalar; hist[j] = phi((j-H) h)
        H = hist.shape[0] - 1
        R = kv.shape[0]
        X = np.zeros(H + N + 1)
        X[:H + 1] = hist
        for j in range(N):
            i = H + j
            conv = 0.0
            for k in range(1, R + 1):
                conv += kv[k - 1] * X[i - k]
            X[i + 1] = X[i] + h * (a * X[i] + b * X[i - d] + h * conv)
        return X[H:]
    return be, fe_riemann


def be_matrix(S, h, N):
    """All-at-once Backward-Euler matrix (for condition numbers), assembled in COO form."""
    g = S["g"]; IA = np.eye(g) - h * S["A"]
    ri, ci, va = [], [], []
    def put(n, m, M):
        for a in range(g):
            for b in range(g):
                if M[a, b] != 0:
                    ri.append(n * g + a); ci.append(m * g + b); va.append(M[a, b])
    for n in range(N):
        put(n, n, IA)
        if n >= 1:
            put(n, n - 1, -np.eye(g))
        for tau, B in S["delays"]:
            m = n - int(round(tau / h))
            if m >= 0:
                put(n, m, -h * B)
    return sp.csc_matrix((va, (ri, ci)), shape=(N * g, N * g))


def kappa(Lm):
    smax = spla.svds(Lm, k=1, which="LM", return_singular_vectors=False)[0]
    lu = spla.splu(Lm.tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0)
    inv = spla.LinearOperator(Lm.shape, matvec=lambda v: lu.solve(v), rmatvec=lambda v: lu.solve(v, trans="T"))
    smin = 1 / spla.svds(inv, k=1, which="LM", return_singular_vectors=False)[0]
    A = abs(Lm).tocsr(); s = int(np.diff(A.indptr).max())
    return float(smax / smin), float(s * A.max() / smax)


def prior_rows_be(S, hs):
    be, _ = _numba()
    T = S["T"]; g = S["g"]; ts = np.linspace(T / 16, T, 16)
    REF = np.array([exact(S, t) for t in ts]); scale = np.max(np.abs(REF))
    rows = []
    for h in hs:
        N = int(round(T / h)); ds = np.array([int(round(tau / h)) for tau, _ in S["delays"]], dtype=np.int64)
        Hh = int(ds.max())
        hist = np.array([exact(S, (j - Hh) * h) for j in range(Hh + 1)]).reshape(Hh + 1, g)
        Ainv = np.linalg.inv(np.eye(g) - h * S["A"])
        Bs = np.array([B for _, B in S["delays"]], dtype=float)
        t0 = time.time()
        X = be(Ainv, Bs, ds, hist, N, h, g)
        idx = [int(round(t / h)) for t in ts]
        err = float(np.max(np.abs(X[idx] - REF)) / scale)
        row = dict(h=h, N=N, size=N * g, err=err)
        if N * g <= 1.6e4:
            row["kappa"], row["ratio"] = kappa(be_matrix(S, h, N))
        rows.append(row)
        print(f"   BE h={h:.2e} N={N}: err={err:.2e} kappa={row.get('kappa')} ({time.time() - t0:.1f}s)", flush=True)
    _fill_kappa(rows)
    return rows


def prior_rows_fe(S, hs):
    _, fe = _numba()
    T = S["T"]; ts = np.linspace(T / 16, T, 16)
    REF = np.array([exact(S, t)[0] for t in ts]); scale = np.max(np.abs(REF))
    rows = []
    for h in hs:
        N = int(round(T / h)); d = int(round(0.8 / h)); R = int(round(2.0 / h))
        kv = np.array([gauss_k(k * h) for k in range(1, R + 1)])
        Hh = R
        hist = np.array([exact(S, (j - Hh) * h)[0] for j in range(Hh + 1)])
        t0 = time.time()
        X = fe(hist, N, h, d, kv, -2.0, 0.4)
        idx = [int(round(t / h)) for t in ts]
        err = float(np.max(np.abs(X[idx] - REF)) / scale)
        row = dict(h=h, N=N, size=N, err=err)
        if N <= 4000:
            row["kappa"] = fe_kappa(h, N, d, kv)
        rows.append(row)
        print(f"   FE+Riemann h={h:.2e} N={N}: err={err:.2e} kappa={row.get('kappa')} ({time.time() - t0:.1f}s)", flush=True)
    _fill_kappa(rows)
    return rows


def fe_kappa(h, N, d, kv):
    R = len(kv); ri, ci, va = [], [], []
    for j in range(N):
        ri.append(j); ci.append(j); va.append(1.0)
        coef = {j: -(1 - 2 * h)}
        coef[j - d] = coef.get(j - d, 0) - 0.4 * h
        for k in range(1, R + 1):
            coef[j - k] = coef.get(j - k, 0) - h * h * kv[k - 1]
        for i, c in coef.items():
            if i >= 1:
                ri.append(j); ci.append(i - 1); va.append(c)
    return kappa(sp.csr_matrix((va, (ri, ci)), shape=(N, N)))[0]


def _fill_kappa(rows):
    ks = [(r["N"], r["kappa"]) for r in rows if r.get("kappa")]
    fit = np.polyfit(np.log([a for a, _ in ks]), np.log([b for _, b in ks]), 1)
    for r in rows:
        r["kappa_exponent"] = float(fit[0])
        if not r.get("kappa"):
            r["kappa"] = float(np.exp(np.polyval(fit, np.log(r["N"])))); r["kappa_extrapolated"] = True


# ------------------------------------------------------------------------------------------------ main
NAMES = ["two-delay", "drug", "baroreflex", "gaussian", "erlang2"]
# Erlang-4 at strength 0.8 is stable but not certified by this functional family: the scalar LK condition
# 0.64 * Gamma(beta) <= beta, Gamma(beta) = 5120/(4 - beta)^7, fails for every beta (minimum ratio 1.019 at
# beta = 1/2), so it is reported as a limit of the method.


def main(names):
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in names:
        S = spec(name)
        print("==", name, "lambda =", S["lam"], flush=True)
        rec = dict(label=S["label"], family=S["family"], T=S["T"], lam=[S["lam"].real, S["lam"].imag])
        if name.startswith("erlang"):
            rows, meta = ours_rows(S, rs=(2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0))
        else:
            rows, meta = ours_rows(S)
        rec.update(meta); rec["ours"] = rows
        if S["family"] == "delay":
            rec["prior"] = prior_rows_be(S, [0.1 / 2 ** k for k in range(0, 21) if S["T"] / (0.1 / 2 ** k) * S["g"] <= 2.7e7])
            rec["prior_name"] = "first-order (Backward Euler) all at once + optimal QLSA"
        elif name == "gaussian":
            rec["prior"] = prior_rows_fe(S, [0.1 / 2 ** k for k in range(0, 13)])
            rec["prior_name"] = "forward Euler + Riemann sum (Ameri et al.) + optimal QLSA"
        R[name] = rec
        json.dump(R, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or NAMES)
