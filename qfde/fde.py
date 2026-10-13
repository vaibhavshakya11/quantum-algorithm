"""
General linear functional differential equations (FDEs) in a single normal form, its
history-space generator, the Lyapunov-Krasovskii (LK) certificate and an independent
reference solver.

NORMAL FORM (everything below reduces to it)
    xi'(t) = Acal(t) xi + sum_k Bcal_k(t) x(t - d_k(t)) + f(t)            (Markovian state xi in R^n)
    x(t)   = E xi + sum_k C_k x(t - d_k)                                    (output / history variable, R^g)
    x(t)   = phi(t),  t <= 0

  * retarded discrete delays:   d_k constant,      Bcal_k != 0, C_k = 0
  * neutral terms  x'(t) - C x'(t - s) = ... :  xi = x - C x(t - s)  =>  C_k = C (constant delay s)
  * distributed kernels on a finite window [0, r]:  K(s) ~ sum_j Re(c_j e^{-lam_j s}) Kmat
        y_j(t) = int_0^r e^{-lam_j s} x(t-s) ds  is EXACTLY  y_j' = x(t) - lam_j y_j - e^{-lam_j r} x(t - r)
        (Markovianisation on a finite window = auxiliary ODEs + one extra discrete delay at r)
  * time-varying coefficients:  Acal(t), Bcal_k(t), c_j(t)
  * time-varying delays d_k(t) with d_k' < 1: own stretched history block
  * forcing f(t)

Discretisation of the history:
  * all CONSTANT delay points share one window [-r, 0], split into segments at every constant
    delay point, LGL-SBP spectral elements, upwind SAT (core.py machinery);
  * each TIME-VARYING delay gets a stretched block  v(t,s) = x(t + s d(t)), s in [-1, 0]:
        v_t = a(s,t) v_s,   a = (1 + s d'(t)) / d(t) > 0  (needs d' < 1),
    stored as z = e^{gamma s / 2} v, discretised in split (skew-symmetric) SBP form, which
    is energy-stable once  gamma >= max_{s,t} -d'/(1 + s d').
"""
from __future__ import annotations

import bisect
import numpy as np
from dataclasses import dataclass, field
from .core import lgl


# ----------------------------------------------------------------------------------------
# Problem description
# ----------------------------------------------------------------------------------------
@dataclass
class ConstDelay:
    d: float
    B: object            # callable t -> (n, g) or array
    C: np.ndarray = None  # (g, g) neutral coupling or None


@dataclass
class VarDelay:
    d: object            # callable t -> delay
    dd: object           # callable t -> d'(t)
    B: object            # callable t -> (n, g) or array


def _call(M, t):
    return np.asarray(M(t) if callable(M) else M, float)


@dataclass
class QuadKernel:
    r: float             # support [0, r]
    k: object            # scalar kernel s -> float
    K: object            # callable t -> (n, g):  xi' += K(t) int_0^r k(s) x(t - s) ds


@dataclass
class GeneralFDE:
    n: int
    g: int
    A: object                       # callable t -> (n, n) or array
    E: np.ndarray                   # (g, n)
    const: list = field(default_factory=list)
    var: list = field(default_factory=list)
    f: object = None                # callable t -> (n,) or None
    phi: object = None              # callable th -> (g,)
    xi0: np.ndarray = None          # (n,)
    t_range: tuple = (0.0, 10.0)    # for robust certificate checks
    qk: list = field(default_factory=list)   # distributed kernels handled by quadrature on the window
    x_off: object = None                     # optional known offset: x = E xi + sum C u + x_off(t)

    @property
    def const_points(self):
        return sorted(set(round(c.d, 12) for c in self.const))

    @property
    def r(self):
        return max(self.const_points) if self.const else 0.0


# ----------------------------------------------------------------------------------------
# Builders: turn "physics form" into the normal form
# ----------------------------------------------------------------------------------------
def build_fde(A, g, phi, delays=(), neutral=(), kernels=(), var_delays=(), f=None,
              t_range=(0.0, 10.0), quad_kernels=()):
    """
    A        : callable t -> (g, g) or array               (x' = A(t) x + ...)
    delays   : list of (tau, B)    B callable/array (g, g)  retarded discrete delays
    neutral  : list of (sigma, C)  C (g, g)                 x'(t) - C x'(t - sigma) on the LHS
    quad_kernels: list of dict(r=window, k=callable, K=(g,g), coef=callable or None)
               coef(t) K int_0^r k(s) x(t - s) ds  -- the memory mechanism of the paper (history window)
    kernels  : list of dict(r=window, lam=[...complex], c=[...complex], K=(g,g), coef=callable or None)
               int_0^r Re(sum_j c_j e^{-lam_j s}) * coef(t) * K x(t - s) ds, as auxiliary variables;
               used only to build independent reference solutions (benchmarks/general_tests.py)
    var_delays: list of (d(t), d'(t), B)
    f        : callable t -> (g,) forcing
    State xi = [w ; real parts of y ; imag parts of y], w = x - sum C x(t - sigma).
    """
    Afun = (lambda t: _call(A, t))
    # auxiliary SOE variables (complex -> real pairs)
    aux = []                      # (lam, c, K, coef, r)
    for kd in kernels:
        for lam, c in zip(kd["lam"], kd["c"]):
            rr = None if kd.get("r") is None else float(kd["r"])
            aux.append((complex(lam), complex(c), np.asarray(kd["K"], float), kd.get("coef"), rr))
    J = len(aux)
    n = g + 2 * g * J
    E = np.zeros((g, n)); E[:, :g] = np.eye(g)

    def blk_re(j): return slice(g + 2 * g * j, g + 2 * g * j + g)
    def blk_im(j): return slice(g + 2 * g * j + g, g + 2 * g * (j + 1))

    # x_in = w + sum C u(-sigma).  Equations are written with x(t) substituted -> handled in
    # the generator via E and C (x appears through "X" = coupling of xi' to x(t)).
    # Represent:  xi' = Acal xi + Xcal x(t) + sum Bcal u + f,  x(t) = E xi + sum C u.
    # We return Acal' = Acal + Xcal E and Bcal'_k = Bcal_k + Xcal C_k (exact substitution).
    def Acal(t):
        M = np.zeros((n, n))
        for j, (lam, c, K, coef, r) in enumerate(aux):
            a, b = lam.real, lam.imag
            cf = coef(t) if coef is not None else 1.0
            # d/dt [yr; yi] = x - [[a, -b],[b, a]] [yr; yi] - (window-end delay term, below)
            M[blk_re(j), blk_re(j)] += -a * np.eye(g); M[blk_re(j), blk_im(j)] += b * np.eye(g)
            M[blk_im(j), blk_re(j)] += -b * np.eye(g); M[blk_im(j), blk_im(j)] += -a * np.eye(g)
            # w' += Re(c y) K = (cr yr - ci yi) K
            M[:g, blk_re(j)] += cf * c.real * K
            M[:g, blk_im(j)] += -cf * c.imag * K
        return M

    def Xcal(t):
        X = np.zeros((n, g))
        X[:g, :] = Afun(t)
        for j in range(J):
            X[blk_re(j), :] = np.eye(g)
        return X

    const = []
    # neutral delays (constant): C_k on x_in, no direct B except via substitution
    neu = [(float(s), np.asarray(C, float)) for s, C in neutral]
    # gather constant delay points with their B and C
    pts = {}
    for tau, B in delays:
        pts.setdefault(round(float(tau), 12), {"B": [], "C": None})["B"].append(B)
    for s, C in neu:
        pts.setdefault(round(s, 12), {"B": [], "C": None})["C"] = C
    for j, (lam, c, K, coef, r) in enumerate(aux):
        if r is not None:
            pts.setdefault(round(r, 12), {"B": [], "C": None})
    for kd in quad_kernels:
        pts.setdefault(round(float(kd["r"]), 12), {"B": [], "C": None})
    for d, info in sorted(pts.items()):
        Bs_x = info["B"]
        C = info["C"]
        aux_end = [j for j, a in enumerate(aux) if a[4] is not None and abs(a[4] - d) < 1e-12]

        def Bfun(t, Bs_x=Bs_x, C=C, aux_end=aux_end, d=d):
            M = np.zeros((n, g))
            for B in Bs_x:
                M[:g, :] += _call(B, t)
            for j in aux_end:
                lam = aux[j][0]
                ez = np.exp(-lam * d)            # e^{-lam r}: complex -> acts on re/im parts
                M[blk_re(j), :] += -ez.real * np.eye(g)
                M[blk_im(j), :] += -ez.imag * np.eye(g)
            if C is not None:
                M += Xcal(t) @ C               # substitution x(t) = w + C u
            return M
        const.append(ConstDelay(d=d, B=Bfun, C=C))

    var = [VarDelay(d=dfun, dd=ddfun, B=(lambda t, B=B: np.vstack([_call(B, t), np.zeros((n - g, g))])))
           for dfun, ddfun, B in var_delays]

    def Afull(t):
        return Acal(t) + Xcal(t) @ E

    ff = None
    if f is not None:
        ff = lambda t: np.concatenate([np.asarray(f(t), float).reshape(g), np.zeros(n - g)])

    # initial Markovian state: w(0) = phi(0) - sum C phi(-sigma);  y_j(0) = int_0^r e^{-lam s} phi(-s) ds
    xi0 = np.zeros(n)
    w0 = np.asarray(phi(0.0), float).reshape(g).copy()
    for s, C in neu:
        w0 -= C @ np.asarray(phi(-s), float).reshape(g)
    xi0[:g] = w0
    for j, (lam, c, K, coef, r) in enumerate(aux):
        if r is None:
            continue
        sq, wq = np.polynomial.legendre.leggauss(200)
        ss = (sq + 1) * r / 2; ws = wq * r / 2
        val = sum(wk * np.exp(-lam * sk) * np.asarray(phi(-sk), float).reshape(g) for sk, wk in zip(ss, ws))
        xi0[blk_re(j)] = val.real; xi0[blk_im(j)] = val.imag

    qk = []
    for kd in quad_kernels:
        Kx = np.asarray(kd["K"], float); coef = kd.get("coef")
        # the kernel term enters wherever x' does, i.e. the w-equation (rows of Xcal for w only)
        def Kfun(t, Kx=Kx, coef=coef):
            M = np.zeros((n, g)); M[:g, :] = (coef(t) if coef is not None else 1.0) * Kx
            return M
        qk.append(QuadKernel(r=float(kd["r"]), k=kd["k"], K=Kfun))
    return GeneralFDE(n=n, g=g, A=Afull, E=E, const=const, var=var, f=ff, phi=phi, xi0=xi0,
                      t_range=t_range, qk=qk)


# ----------------------------------------------------------------------------------------
# Discretisation and generator
# ----------------------------------------------------------------------------------------
@dataclass
class Elem:
    a: float
    b: float
    idx: np.ndarray       # node indices (global node numbering)
    nodes: np.ndarray     # coordinates (th for window, s for blocks)
    seg: int = 0          # window segment, or block id
    kind: str = "win"


@dataclass
class GenDisc:
    fde: GeneralFDE
    p: int
    xi: np.ndarray
    w: np.ndarray
    D: np.ndarray
    win: list
    blocks: list          # list of lists of Elem (one list per VarDelay)
    gammas: list
    n_nodes: int
    beta: float = 0.0     # exponential weight e^{beta th} on the constant-delay window

    @property
    def size(self):
        return self.fde.n + self.fde.g * self.n_nodes

    def node_slice(self, j):
        s = self.fde.n + self.fde.g * j
        return slice(s, s + self.fde.g)


def gamma_for(vd: VarDelay, t_range, ns=400):
    ts = np.linspace(*t_range, ns)
    best = 0.0
    for t in ts:
        dp = vd.dd(t)
        for s in (-1.0, 0.0):
            den = 1 + s * dp
            if den <= 0:
                raise ValueError("needs d'(t) < 1")
            best = max(best, -dp / den)
    return best


def discretize_general(fde: GeneralFDE, p: int, h: float, hs: float = 0.25, beta: float = 0.0):
    xi, w, D = lgl(p)
    win, count = [], 0
    pts = [0.0] + fde.const_points
    for k in range(len(pts) - 1):
        top, bot = -pts[k], -pts[k + 1]
        nE = max(1, int(np.ceil((top - bot) / h - 1e-12)))
        edges = np.linspace(bot, top, nE + 1)
        for e in range(nE):
            a, b = edges[e], edges[e + 1]
            idx = np.arange(count, count + p + 1); count += p + 1
            win.append(Elem(a=a, b=b, idx=idx, nodes=a + (b - a) * (xi + 1) / 2, seg=k))
    blocks, gammas = [], []
    for i, vd in enumerate(fde.var):
        nE = max(1, int(np.ceil(1.0 / hs - 1e-12)))
        edges = np.linspace(-1.0, 0.0, nE + 1)
        bl = []
        for e in range(nE):
            a, b = edges[e], edges[e + 1]
            idx = np.arange(count, count + p + 1); count += p + 1
            bl.append(Elem(a=a, b=b, idx=idx, nodes=a + (b - a) * (xi + 1) / 2, seg=i, kind="var"))
        blocks.append(bl)
        gammas.append(gamma_for(vd, fde.t_range) * 1.0000001)
    return GenDisc(fde=fde, p=p, xi=xi, w=w, D=D, win=win, blocks=blocks, gammas=gammas, n_nodes=count,
                   beta=beta)


def _left_end(elems, th):
    for e in elems:
        if abs(e.a - th) < 1e-12:
            return e
    raise KeyError(th)


def x_in_row(disc: GenDisc):
    """(g, size) matrix R with x(t) = R z  (x = E xi + sum C u(-d))."""
    fde = disc.fde; g = fde.g
    R = np.zeros((g, disc.size))
    R[:, :fde.n] = fde.E
    for c in fde.const:
        if c.C is not None:
            e = _left_end(disc.win, -c.d)
            R[:, disc.node_slice(e.idx[0])] += c.C * np.exp(disc.beta * c.d / 2)
    return R


def make_generator(disc: GenDisc):
    """Precompute G(t) = G_static + [A(t) in xi block] + [B_k(t) couplings]
                         + sum_i (1/d_i(t)) M1_i + (d_i'(t)/d_i(t)) Ms_i   (stretched blocks)
    and return a callable t -> (G(t), b(t)).  (Exact: the stretched-block operator is affine in
    a = (1 + s d')/d  =  1/d + s d'/d.)"""
    fde = disc.fde; g, n = fde.g, fde.n
    N = disc.size
    I = np.eye(g)
    R = x_in_row(disc)

    def put(M, rs, cs, val):
        M[rs, cs] += val

    # ---- static part: window transport + SATs
    Gs = np.zeros((N, N))
    for e in disc.win:
        h = e.b - e.a
        Dl = (2.0 / h) * disc.D - 0.5 * disc.beta * np.eye(len(disc.xi))
        Gs[np.ix_(_rows(disc, e.idx), _rows(disc, e.idx))] += np.kron(Dl, I)
        pen = (2.0 / h) / disc.w[-1]
        sN = disc.node_slice(e.idx[-1])
        Gs[sN, sN] -= pen * I
        if abs(e.b) < 1e-12:
            Gs[sN, :] += pen * R
        else:
            up = _left_end(disc.win, e.b)
            Gs[sN, disc.node_slice(up.idx[0])] += pen * I

    # ---- stretched blocks: M1 (coefficient 1/d) and Ms (coefficient d'/d)
    M1s, Mss = [], []
    for bl, gam in zip(disc.blocks, disc.gammas):
        M1 = np.zeros((N, N)); Ms = np.zeros((N, N))
        for e in bl:
            h = e.b - e.a
            Dl = (2.0 / h) * disc.D
            s = e.nodes
            one = np.ones_like(s)
            # a = c1 * 1 + cs * s
            for coeff_vec, M, extra in ((one, M1, 0.0), (s, Ms, 1.0)):
                Lop = 0.5 * (Dl * coeff_vec[None, :] + coeff_vec[:, None] * Dl) \
                      - 0.5 * extra * np.eye(len(s)) - 0.5 * gam * np.diag(coeff_vec)
                M[np.ix_(_rows(disc, e.idx), _rows(disc, e.idx))] += np.kron(Lop, I)
                pen = (2.0 / h) / disc.w[-1] * coeff_vec[-1]
                sN = disc.node_slice(e.idx[-1])
                M[sN, sN] -= pen * I
                if abs(e.b) < 1e-12:
                    M[sN, :] += pen * R
                else:
                    up = _left_end(bl, e.b)
                    M[sN, disc.node_slice(up.idx[0])] += pen * I
        M1s.append(M1); Mss.append(Ms)

    cslots = [disc.node_slice(_left_end(disc.win, -c.d).idx[0]) for c in fde.const]
    cfac = [np.exp(disc.beta * c.d / 2) for c in fde.const]
    # quadrature kernels: v = int_0^r k(s) u(-s) ds  ~  sum_nodes omega_j k(-th_j) e^{-beta th_j/2} z_j
    kq = []
    for qk in fde.qk:
        cols, wts = [], []
        for e in disc.win:
            if e.a < -qk.r - 1e-12:
                continue
            hh = e.b - e.a
            for jl, j in enumerate(e.idx):
                th = e.nodes[jl]
                cols.append(disc.node_slice(j))
                wts.append((hh / 2) * disc.w[jl] * qk.k(-th) * np.exp(-disc.beta * th / 2))
        kq.append((qk, cols, wts))
    vslots = [disc.node_slice(_left_end(bl, -1.0).idx[0]) for bl in disc.blocks]

    # rows that receive the inflow value x(t) through a SAT (window top element + top of each stretched block)
    inflow_rows = []
    for e in disc.win:
        if abs(e.b) < 1e-12:
            inflow_rows.append((disc.node_slice(e.idx[-1]), (2.0 / (e.b - e.a)) / disc.w[-1], None))
    for i, (bl, vd) in enumerate(zip(disc.blocks, fde.var)):
        for e in bl:
            if abs(e.b) < 1e-12:
                inflow_rows.append((disc.node_slice(e.idx[-1]), (2.0 / (e.b - e.a)) / disc.w[-1], vd))

    def G_of(t):
        G = Gs.copy()
        G[:n, :n] += fde.A(t)
        for c, sl, fac in zip(fde.const, cslots, cfac):
            G[:n, sl] += c.B(t) * fac
        for qk, cols, wts in kq:
            Kt = qk.K(t)
            for sl, wt in zip(cols, wts):
                G[:n, sl] += wt * Kt
        for vd, sl, gam, M1, Ms in zip(fde.var, vslots, disc.gammas, M1s, Mss):
            d, dp = vd.d(t), vd.dd(t)
            G[:n, sl] += vd.B(t) * np.exp(gam / 2)
            G += (1.0 / d) * M1 + (dp / d) * Ms
        b = np.zeros(N)
        if fde.f is not None:
            b[:n] = fde.f(t)
        if fde.x_off is not None:
            xo = np.asarray(fde.x_off(t), float)
            for sl, pen, vd in inflow_rows:
                b[sl] += pen * (1.0 if vd is None else 1.0 / vd.d(t)) * xo     # stretched-block speed a(0) = 1/d
        return G, b
    return G_of


def _rows(disc, idx):
    g = disc.fde.g
    return np.concatenate([np.arange(disc.fde.n + g * j, disc.fde.n + g * (j + 1)) for j in idx])


def solve_general(disc: GenDisc, T, t_out, rtol=1e-12, atol=1e-14, autonomous=False):
    """Integrate the semi-discrete system (stand-in for exact LCHS evolution) and return x(t_out).
    autonomous=True (constant G and b): exact affine matrix exponential on uniformly spaced t_out."""
    from scipy.integrate import solve_ivp
    G_of = make_generator(disc)
    R = x_in_row(disc)
    z0 = initial_state_general(disc)
    if autonomous:
        from scipy.linalg import expm
        G, b = G_of(0.0)
        N = len(z0)
        Aug = np.zeros((N + 1, N + 1)); Aug[:N, :N] = G; Aug[:N, N] = b
        off = (lambda t: np.asarray(disc.fde.x_off(t), float)) if disc.fde.x_off is not None else (lambda t: 0.0)
        dts = np.diff(np.concatenate([[0.0], t_out]))
        z = np.concatenate([z0, [1.0]]); xs = []
        cache = {}
        for t, dt in zip(t_out, dts):
            key = round(dt, 12)
            if key not in cache:
                cache[key] = expm(Aug * dt)
            z = cache[key] @ z
            xs.append(R @ z[:N] + off(t))
        return np.array(xs), z0
    if not disc.fde.var and callable(disc.fde.A) and getattr(disc.fde, "autonomous", False):
        pass
    sol = solve_ivp(lambda t, z: (lambda Gb: Gb[0] @ z + Gb[1])(G_of(t)), (0.0, T), z0,
                    method="DOP853", rtol=rtol, atol=atol, t_eval=t_out)
    off = (lambda t: np.asarray(disc.fde.x_off(t), float)) if disc.fde.x_off is not None else (lambda t: 0.0)
    return np.array([R @ sol.y[:, k] + off(t_out[k]) for k in range(len(t_out))]), z0


def initial_state_general(disc: GenDisc):
    fde = disc.fde
    z = np.zeros(disc.size)
    z[:fde.n] = fde.xi0
    for e in disc.win:
        for jl, j in enumerate(e.idx):
            th = e.nodes[jl]
            z[disc.node_slice(j)] = np.exp(disc.beta * th / 2) * np.asarray(fde.phi(th), float)
    for vd, bl, gam in zip(fde.var, disc.blocks, disc.gammas):
        d0 = vd.d(fde.t_range[0])
        for e in bl:
            for jl, j in enumerate(e.idx):
                s = e.nodes[jl]
                z[disc.node_slice(j)] = np.exp(gam * s / 2) * np.asarray(fde.phi(s * d0), float)
    return z


# ----------------------------------------------------------------------------------------
# LK certificate (robust over a time grid) and weight matrix
# ----------------------------------------------------------------------------------------
def kernel_gamma(qk: QuadKernel, beta: float, nq=2000):
    """Gamma = int_0^r k(s)^2 e^{beta s} ds  (Cauchy-Schwarz constant in the e^{beta th} weight)."""
    sq, wq = np.polynomial.legendre.leggauss(nq)
    ss = (sq + 1) * qk.r / 2; ws = wq * qk.r / 2
    return float(np.sum(ws * np.array([qk.k(x) ** 2 for x in ss]) * np.exp(beta * ss)))


def kernel_gamma_h(fde: GeneralFDE, qk: QuadKernel, p: int, h: float, beta: float, hs: float = 0.25):
    """Quadrature analogue Gamma_h = sum_j omega_j k(-th_j)^2 e^{-beta th_j} over the window nodes that the
    generator uses for this kernel on the (p, h) mesh.  The main theorem needs Gamma (in the LMI) >= Gamma_h."""
    disc = discretize_general(fde, p, h, hs=hs, beta=beta)
    tot = 0.0
    for e in disc.win:
        if e.a < -qk.r - 1e-12:
            continue
        hh = e.b - e.a
        for jl in range(len(e.idx)):
            th = e.nodes[jl]
            tot += (hh / 2) * disc.w[jl] * qk.k(-th) ** 2 * np.exp(-beta * th)
    return float(tot)


def lk_certificate_general(fde: GeneralFDE, gammas, beta=0.0, nt=25, eta=1e-6, margin=0.0, slack=1.02,
                           meshes=None):
    """Robust (time-gridded) Lyapunov-Krasovskii LMI for the general normal form.

    Functional:  V = xi'P xi + sum_k int_{-d_k}^0 e^{beta th} u'Q_k u dth + int_{-r}^0 e^{beta th} u'S0 u dth
                 + (stretched blocks)  int_{-1}^0 e^{gamma s} v'Qv_i v ds
    Variables of the quadratic form:  eta = (xi, u(-d_1..m), v_end(1..q), kernel inputs vk_1..L)
    with vk_l = int_0^{r_l} k_l(s) u(-s) ds, controlled by Cauchy-Schwarz:
        vk' S0_l vk <= Gamma_l int e^{beta th} u' S0_l u  and the dissipation  -beta int e^{beta th} u' S0 u.
    Blocks:
      xi-xi        : P A + A'P + F'(sum Q_k + sum S0_l + sum Qv_i / d_i(t)) F
      u_k          : -e^{-beta d_k} Q_k
      v_i          : -(1 - d_i') e^{-gamma_i} Qv_i / d_i
      vk_l         : -(beta / Gamma_l) S0_l
      couplings    : P B_k, P Bv_i, P K_l
    """
    import cvxpy as cp
    n, g = fde.n, fde.g
    m, q, L = len(fde.const), len(fde.var), len(fde.qk)
    if L and beta <= 0:
        return False, None, None, None, None, "kernels need beta > 0"
    Gam = [kernel_gamma(k, beta) * slack for k in fde.qk]
    if meshes:      # Gamma must dominate the quadrature constant of every mesh the weight will be used on
        Gam = [max([G] + [kernel_gamma_h(fde, k, mp, mh, beta) for mp, mh in meshes]) for G, k in zip(Gam, fde.qk)]
    P = cp.Variable((n, n), symmetric=True)
    Qc = [cp.Variable((g, g), symmetric=True) for _ in range(m)]
    Qv = [cp.Variable((g, g), symmetric=True) for _ in range(q)]
    S0 = [cp.Variable((g, g), symmetric=True) for _ in range(L)]
    tvar = cp.Variable()
    cons = [P >> np.eye(n), P << tvar * np.eye(n)]
    cons += [Q >> eta * np.eye(g) for Q in Qc + Qv]
    cons += [S >> 0 for S in S0]
    nb = 1 + m + q + L
    Fbl = [fde.E] + [(c.C if c.C is not None else np.zeros((g, g))) for c in fde.const] + \
          [np.zeros((g, g))] * (q + L)
    for t in np.linspace(*fde.t_range, nt):
        A = fde.A(t)
        S_in = np.zeros((g, g))
        for Q in Qc: S_in = S_in + Q
        for S in S0: S_in = S_in + S
        for i in range(q): S_in = S_in + Qv[i] / fde.var[i].d(t)
        Bl = [c.B(t) for c in fde.const] + [vd.B(t) for vd in fde.var] + [k.K(t) for k in fde.qk]
        rows = []
        for a_ in range(nb):
            row = []
            for b_ in range(nb):
                term = Fbl[a_].T @ S_in @ Fbl[b_]
                if a_ == 0 and b_ == 0:
                    term = term + P @ A + A.T @ P + 2 * margin * P
                elif a_ == 0:
                    term = term + P @ Bl[b_ - 1]
                elif b_ == 0:
                    term = term + Bl[a_ - 1].T @ P
                elif a_ == b_:
                    k = a_ - 1
                    if k < m:
                        term = term - np.exp(-beta * fde.const[k].d) * Qc[k]
                    elif k < m + q:
                        i = k - m; vd = fde.var[i]
                        term = term - Qv[i] * ((1 - vd.dd(t)) * np.exp(-gammas[i]) / vd.d(t))
                    else:
                        l = k - m - q
                        term = term - (beta / Gam[l]) * S0[l]
                row.append(term)
            rows.append(row)
        Mt = cp.bmat(rows)
        cons.append((Mt + Mt.T) / 2 << 0)
    prob = cp.Problem(cp.Minimize(tvar), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve(solver=cp.SCS, eps=1e-9)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return False, None, None, None, None, prob.status
    sym = lambda X: 0.5 * (X + X.T)
    return (True, sym(P.value), [sym(Q.value) for Q in Qc], [sym(Q.value) for Q in Qv],
            [sym(S.value) for S in S0], dict(status=prob.status, Gamma=Gam))


def certify(fde: GeneralFDE, gammas, betas=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0), **kw):
    """Search the window weight beta; return the feasible certificate with the smallest cond(P)."""
    best = None
    for b in betas:
        if fde.qk and b == 0.0:
            continue
        res = lk_certificate_general(fde, gammas, beta=b, **kw)
        if res[0]:
            cond = float(np.linalg.cond(res[1]))
            if best is None or cond < best[0]:
                best = (cond, b, res)
            if not fde.qk:
                break
    if best is None:
        return dict(ok=False)
    cond, b, (ok, P, Qc, Qv, S0, st) = best
    return dict(ok=True, beta=b, P=P, Qc=Qc, Qv=Qv, S0=S0, status=st["status"], Gamma=st["Gamma"], cond_P=cond)


def check_gamma(disc: GenDisc, cert):
    """Main-theorem hypothesis Gamma_l >= Gamma_{l,h} for this discretization: list of (Gamma_h, Gamma, ok)."""
    out = []
    for qk, G in zip(disc.fde.qk, cert.get("Gamma", [])):
        tot = 0.0
        for e in disc.win:
            if e.a < -qk.r - 1e-12:
                continue
            hh = e.b - e.a
            for jl in range(len(e.idx)):
                th = e.nodes[jl]
                tot += (hh / 2) * disc.w[jl] * qk.k(-th) ** 2 * np.exp(-disc.beta * th)
        out.append((float(tot), float(G), bool(tot <= G * (1 + 1e-12))))
    return out


def weight_general(disc: GenDisc, P, Qc, Qv, S0=()):
    fde = disc.fde; g, n = fde.g, fde.n
    pts = fde.const_points
    order = {round(c.d, 12): k for k, c in enumerate(fde.const)}
    Qby = [Qc[order[d]] for d in pts]
    extra = sum(S0) if len(S0) else np.zeros((g, g))
    S = [sum(Qby[i] for i in range(k, len(pts))) + extra for k in range(len(pts))]
    W = np.zeros((disc.size, disc.size))
    W[:n, :n] = P
    for e in disc.win:
        h = e.b - e.a
        for jl, j in enumerate(e.idx):
            W[disc.node_slice(j), disc.node_slice(j)] = S[e.seg] * (h / 2) * disc.w[jl]
    for i, bl in enumerate(disc.blocks):
        for e in bl:
            h = e.b - e.a
            for jl, j in enumerate(e.idx):
                W[disc.node_slice(j), disc.node_slice(j)] = Qv[i] * (h / 2) * disc.w[jl]
    return W


# ----------------------------------------------------------------------------------------
# Independent reference: method of steps on the ORIGINAL physics form, true kernels by quadrature
# ----------------------------------------------------------------------------------------
class History:
    def __init__(self, phi, g):
        self.phi, self.g = phi, g
        self.t0s, self.t1s, self.sols = [], [], []

    def add(self, a, b, sol):
        self.t0s.append(a); self.t1s.append(b); self.sols.append(sol)

    def __call__(self, t):
        if t <= 0.0:
            return np.asarray(self.phi(t), float).reshape(self.g)
        k = bisect.bisect_right(self.t0s, t) - 1
        k = max(0, k)
        a, b = self.t0s[k], self.t1s[k]
        if t > b + 1e-12:
            raise ValueError(f"t={t} beyond computed history {b}")
        return np.asarray(self.sols[k](min(max(t, a), b)), float)[:self.g]


def reference_general(fde: GeneralFDE, T, rtol=1e-12, atol=1e-14, stats=None):
    """Method of steps (DOP853 + dense output) on the NORMAL FORM, independent of the history
    discretisation:  xi' = A xi + sum B x(t-d) + f,  x = E xi + sum C x(t - d).
    Returns a callable t -> x(t)."""
    from scipy.integrate import solve_ivp
    g = fde.g
    hist = History(fde.phi, g)
    mins = [c.d for c in fde.const]
    for vd in fde.var:
        mins.append(min(vd.d(t) for t in np.linspace(0, T, 400)))
    step = min(mins)

    def xnow(t, xi):
        x = fde.E @ xi + (np.asarray(fde.x_off(t), float) if fde.x_off is not None else 0.0)
        for c in fde.const:
            if c.C is not None:
                x = x + c.C @ hist(t - c.d)
        return x

    def rhs(t, xi):
        if stats is not None:
            stats["nfev"] = stats.get("nfev", 0) + 1
        out = fde.A(t) @ xi
        for c in fde.const:
            out = out + c.B(t) @ hist(t - c.d)
        for vd in fde.var:
            out = out + vd.B(t) @ hist(t - vd.d(t))
        if fde.f is not None:
            out = out + fde.f(t)
        return out

    bps = {0.0, T}
    for c in fde.const:
        k = 1
        while k * c.d < T:
            bps.add(round(k * c.d, 12)); k += 1
    # breaking points of time-varying delays: t_{k+1} solves t - d(t) = t_k (t - d(t) is increasing as d' < 1)
    from scipy.optimize import brentq
    for vd in fde.var:
        tk = 0.0
        for _ in range(200):
            fz = lambda t: t - vd.d(t) - tk
            lo, hi = tk, tk + 1.0
            while fz(hi) < 0 and hi < T + 10:
                hi += 1.0
            if hi >= T + 10 or fz(lo) > 0:
                break
            tk = brentq(fz, lo, hi, xtol=1e-14)
            if tk >= T:
                break
            bps.add(tk)
    if fde.qk:
        raise ValueError("reference_general integrates the normal form with discrete delays only; "
                         "use a high-order SOE (Markovianised) twin for distributed kernels")
    grid = sorted(bps)
    fine = [grid[0]]
    for a, b in zip(grid[:-1], grid[1:]):
        nn = max(1, int(np.ceil((b - a) / step - 1e-12)))
        fine += list(np.linspace(a, b, nn + 1)[1:])
    xi = fde.xi0.copy()
    for a, b in zip(fine[:-1], fine[1:]):
        sol = solve_ivp(rhs, (a, b), xi, method="DOP853", rtol=rtol, atol=atol, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        hist.add(a, b, (lambda t, sol=sol: xnow(t, sol.sol(t))))
        xi = sol.y[:, -1]
    return hist


# ----------------------------------------------------------------------------------------
# Sum-of-exponentials fit on a finite window (matrix pencil), complex exponents allowed
# ----------------------------------------------------------------------------------------
def soe_fit(kfun, r, J, M=400):
    """Fit k(s) ~ Re(sum_j c_j e^{-lam_j s}) on [0, r] with J complex terms (conjugate pairs are
    merged by taking real parts).  Returns (lam, c, max_abs_err)."""
    s = np.linspace(0, r, M); dt = s[1] - s[0]
    y = np.array([kfun(si) for si in s])
    L = M // 2
    Y = np.array([y[i:i + L + 1] for i in range(M - L)])
    U, S, Vh = np.linalg.svd(Y, full_matrices=False)
    K2 = min(2 * J, len(S))
    V = Vh[:K2].conj().T
    V1, V2 = V[:-1], V[1:]
    zs = np.linalg.eigvals(np.linalg.pinv(V1) @ V2)
    lam = -np.log(zs.astype(complex)) / dt
    Z = np.exp(-np.outer(s, lam))
    c, *_ = np.linalg.lstsq(Z, y.astype(complex), rcond=None)
    # keep one of each conjugate pair (Re(c e^{-lam s}) with doubled weight)
    keep_lam, keep_c = [], []
    used = np.zeros(len(lam), bool)
    for i in range(len(lam)):
        if used[i]:
            continue
        j = None
        if abs(lam[i].imag) > 1e-9:
            cands = [k for k in range(len(lam)) if not used[k] and k != i and abs(lam[k] - lam[i].conjugate()) < 1e-6 * (1 + abs(lam[i]))]
            j = cands[0] if cands else None
        if j is not None:
            used[i] = used[j] = True
            keep_lam.append(lam[i]); keep_c.append(2 * c[i])
        else:
            used[i] = True
            keep_lam.append(complex(lam[i].real, 0.0) if abs(lam[i].imag) < 1e-9 else lam[i])
            keep_c.append(c[i] if abs(lam[i].imag) > 1e-9 else complex(c[i].real, 0.0))
    approx = np.real(np.exp(-np.outer(s, keep_lam)) @ np.array(keep_c))
    return np.array(keep_lam), np.array(keep_c), float(np.max(np.abs(approx - y)))
