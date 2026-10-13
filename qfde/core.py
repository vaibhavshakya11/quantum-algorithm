"""
Core of the phase-1 prototype: history-space (infinitesimal-generator) formulation of
linear DDEs with multiple discrete delays, discretized with Legendre-Gauss-Lobatto (LGL)
summation-by-parts (SBP) spectral elements and upwind SAT coupling, plus the
Lyapunov-Krasovskii (LK) weighted norm that makes the semi-discrete generator dissipative.

Problem class (phase 1):
    x'(t) = A x(t) + sum_i B_i x(t - tau_i),   t > 0,
    x(t)  = phi(t)  for t in [-r, 0],  r = max tau_i.

History variable u(t, th) = x(t + th), th in [-r, 0], obeys the transport equation
    u_t = u_th,  with inflow u(t, 0) = x(t).

The th-interval is split into segments [-tau_k, -tau_{k-1}] (tau_0 = 0) so each delay sits
exactly on a segment boundary (an LGL end node); each segment is split into elements.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from numpy.polynomial import legendre as L


# ----------------------------------------------------------------------------------------
# LGL nodes, weights and the SBP differentiation matrix on the reference element [-1, 1]
# ----------------------------------------------------------------------------------------
def lgl(p: int):
    """Return LGL nodes xi (ascending), quadrature weights w and differentiation matrix D
    for polynomial degree p (p+1 nodes).  (H = diag(w), Q = H D) satisfies the SBP
    property Q + Q^T = diag(-1, 0, ..., 0, 1)."""
    if p < 1:
        raise ValueError("p >= 1")
    cP = np.zeros(p + 1); cP[p] = 1.0                 # Legendre P_p
    interior = L.legroots(L.legder(cP)) if p > 1 else np.array([])
    xi = np.concatenate(([-1.0], np.sort(interior.real), [1.0]))
    Pp = L.legval(xi, cP)
    w = 2.0 / (p * (p + 1) * Pp**2)
    n = p + 1
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = Pp[i] / (Pp[j] * (xi[i] - xi[j]))
    D[0, 0] = -p * (p + 1) / 4.0
    D[-1, -1] = p * (p + 1) / 4.0
    return xi, w, D


# ----------------------------------------------------------------------------------------
# Problem and discretization containers
# ----------------------------------------------------------------------------------------
@dataclass
class LinearDDE:
    A: np.ndarray                      # (g, g)
    delays: list                       # list of floats tau_i (any order)
    Bs: list                           # list of (g, g) matrices, aligned with delays

    def __post_init__(self):
        self.A = np.atleast_2d(np.asarray(self.A, float))
        self.Bs = [np.atleast_2d(np.asarray(B, float)) for B in self.Bs]
        order = np.argsort(self.delays)
        self.delays = [float(self.delays[k]) for k in order]
        self.Bs = [self.Bs[k] for k in order]

    @property
    def g(self):
        return self.A.shape[0]

    @property
    def r(self):
        return self.delays[-1]


@dataclass
class Element:
    a: float            # left end (more negative th): outflow end
    b: float            # right end (closer to 0): inflow end
    seg: int            # segment index k (0-based; segment k covers [-tau_{k+1}, -tau_k])
    idx: np.ndarray     # global node indices (into the node list, not including x)
    nodes: np.ndarray   # th values of its nodes (ascending)


@dataclass
class Discretization:
    dde: LinearDDE
    p: int
    elements: list = field(default_factory=list)
    n_nodes: int = 0
    xi: np.ndarray = None
    w: np.ndarray = None
    D: np.ndarray = None

    @property
    def g(self):
        return self.dde.g

    @property
    def size(self):
        return self.g * (1 + self.n_nodes)

    def node_theta(self):
        th = np.zeros(self.n_nodes)
        for e in self.elements:
            th[e.idx] = e.nodes
        return th


def discretize(dde: LinearDDE, p: int, h_target: float):
    """Split each delay segment into ceil(len/h_target) equal elements of degree p."""
    xi, w, D = lgl(p)
    disc = Discretization(dde=dde, p=p, xi=xi, w=w, D=D)
    taus = [0.0] + dde.delays
    count = 0
    for k in range(len(dde.delays)):
        top, bot = -taus[k], -taus[k + 1]
        length = top - bot
        if length <= 0:
            continue
        nE = max(1, int(np.ceil(length / h_target - 1e-12)))
        edges = np.linspace(bot, top, nE + 1)
        for e in range(nE):
            a, b = edges[e], edges[e + 1]
            idx = np.arange(count, count + p + 1)
            count += p + 1
            nodes = a + (b - a) * (xi + 1) / 2
            disc.elements.append(Element(a=a, b=b, seg=k, idx=idx, nodes=nodes))
    disc.n_nodes = count
    return disc


# ----------------------------------------------------------------------------------------
# Semi-discrete generator  z' = G z,  z = [x; u_nodes]  (each entry a g-vector)
# ----------------------------------------------------------------------------------------
def _find_element_with_left_end(disc, th):
    for e in disc.elements:
        if abs(e.a - th) < 1e-12:
            return e
    return None


def generator(disc: Discretization):
    dde, g = disc.dde, disc.g
    N = disc.size
    G = np.zeros((N, N))
    I = np.eye(g)

    def blk(i):  # slice for unknown i; i = -1 is x, else node index
        s = g * (i + 1)
        return slice(s, s + g)

    xs = blk(-1)
    G[xs, xs] += dde.A
    # delayed terms: x(t - tau_i) = node 0 of the element whose left end is -tau_i
    for tau, B in zip(dde.delays, dde.Bs):
        e = _find_element_with_left_end(disc, -tau)
        G[xs, blk(e.idx[0])] += B

    for e in disc.elements:
        h = e.b - e.a
        Dloc = (2.0 / h) * disc.D
        for jl, j in enumerate(e.idx):
            for ll, l in enumerate(e.idx):
                if Dloc[jl, ll] != 0.0:
                    G[blk(j), blk(l)] += Dloc[jl, ll] * I
        # upwind SAT at the inflow (right) end:  -H^{-1} e_p (u_p - u_in)
        pen = (2.0 / h) / disc.w[-1]
        jp = e.idx[-1]
        G[blk(jp), blk(jp)] -= pen * I
        if abs(e.b) < 1e-12:
            G[blk(jp), xs] += pen * I          # inflow value is x(t)
        else:
            up = _find_element_with_left_end(disc, e.b)
            G[blk(jp), blk(up.idx[0])] += pen * I
    return G


def initial_state(disc: Discretization, phi):
    """phi: callable th -> g-vector, history on [-r, 0]."""
    g = disc.g
    z = np.zeros(disc.size)
    z[:g] = np.asarray(phi(0.0), float).reshape(g)
    th = disc.node_theta()
    for j, t in enumerate(th):
        z[g * (j + 1): g * (j + 2)] = np.asarray(phi(t), float).reshape(g)
    return z


# ----------------------------------------------------------------------------------------
# Lyapunov-Krasovskii weight W = blockdiag(P, S_k (x) h/2 * w_j)
# ----------------------------------------------------------------------------------------
def lk_weight(disc: Discretization, P, Qs):
    """P: (g,g) SPD; Qs: list of (g,g) PSD aligned with sorted delays.
    Segment weight S_k = sum_{i >= k} Q_i (k-th segment from the top)."""
    g, m = disc.g, len(disc.dde.delays)
    S = [sum(Qs[i] for i in range(k, m)) for k in range(m)]
    W = np.zeros((disc.size, disc.size))
    W[:g, :g] = P
    for e in disc.elements:
        h = e.b - e.a
        for jl, j in enumerate(e.idx):
            s = g * (j + 1)
            W[s:s + g, s:s + g] = S[e.seg] * (h / 2.0) * disc.w[jl]
    return W


def sym_max_eig(M):
    return float(np.max(np.linalg.eigvalsh(0.5 * (M + M.T))))


def weighted_generator(G, W):
    """Return W^{1/2} G W^{-1/2} (the matrix LCHS would actually see)."""
    lam, V = np.linalg.eigh(W)
    Wh = (V * np.sqrt(lam)) @ V.T
    Wih = (V / np.sqrt(lam)) @ V.T
    return Wh @ G @ Wih, Wh


# ----------------------------------------------------------------------------------------
# LK certificate: LMI  [[PA + A^T P + sum Q_i, P B_1 ... P B_m],
#                        [B_i^T P, -diag(Q_i)]]  <= 0
# ----------------------------------------------------------------------------------------
def scalar_certificate(dde: LinearDDE):
    """P = I, Q_i = ||B_i|| I.  Valid iff mu(A) + sum ||B_i|| <= 0 (log-norm condition)."""
    g = dde.g
    muA = sym_max_eig(dde.A)
    nb = [np.linalg.norm(B, 2) for B in dde.Bs]
    ok = muA + sum(nb) <= 1e-12
    Qs = [max(n, 1e-9) * np.eye(g) for n in nb]
    return ok, np.eye(g), Qs, muA + sum(nb)


def lmi_certificate(dde: LinearDDE, margin=0.0, eta=1e-6, verbose=False):
    """Solve the delay-independent LK LMI with cvxpy; minimize cond(P) (P in [I, t I])."""
    import cvxpy as cp
    g, m = dde.g, len(dde.delays)
    P = cp.Variable((g, g), symmetric=True)
    Qs = [cp.Variable((g, g), symmetric=True) for _ in range(m)]
    t = cp.Variable()
    top = P @ dde.A + dde.A.T @ P + sum(Qs) + 2 * margin * P
    rows = [[top] + [P @ B for B in dde.Bs]]
    for i in range(m):
        row = [dde.Bs[i].T @ P]
        for j in range(m):
            row.append(-Qs[i] if i == j else np.zeros((g, g)))
        rows.append(row)
    M = cp.bmat(rows)
    Msym = (M + M.T) / 2
    cons = [P >> np.eye(g), P << t * np.eye(g), Msym << 0] + [Q >> eta * np.eye(g) for Q in Qs]
    prob = cp.Problem(cp.Minimize(t), cons)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=verbose)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=verbose, eps=1e-9)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return False, None, None, prob.status
    Pv = 0.5 * (P.value + P.value.T)
    Qv = [0.5 * (Q.value + Q.value.T) for Q in Qs]
    return True, Pv, Qv, prob.status


def lk_form(dde, P, Qs):
    """Return the LK quadratic-form matrix (should be <= 0)."""
    g, m = dde.g, len(dde.delays)
    n = g * (m + 1)
    M = np.zeros((n, n))
    M[:g, :g] = P @ dde.A + dde.A.T @ P + sum(Qs)
    for i, B in enumerate(dde.Bs):
        s = g * (i + 1)
        M[:g, s:s + g] = P @ B
        M[s:s + g, :g] = B.T @ P
        M[s:s + g, s:s + g] = -Qs[i]
    return M


# ----------------------------------------------------------------------------------------
# Independent reference solver: method of steps with DOP853 dense output
# ----------------------------------------------------------------------------------------
def reference_solution(dde: LinearDDE, phi, T, rtol=1e-13, atol=1e-15):
    from scipy.integrate import solve_ivp
    g = dde.g
    step = dde.delays[0]
    pieces = []          # list of (t0, t1, dense)

    def xval(t):
        if t <= 0.0:
            return np.asarray(phi(t), float).reshape(g)
        for (t0, t1, sol) in pieces:
            if t0 - 1e-14 <= t <= t1 + 1e-14:
                return sol(min(max(t, t0), t1))
        raise ValueError(t)

    # breakpoints: all k*tau_i combinations within [0, T], so the integrator never steps over a kink
    bps = {0.0, T}
    for tau in dde.delays:
        k = 1
        while k * tau < T:
            bps.add(round(k * tau, 12)); k += 1
    # second-generation breakpoints (sums of two delays)
    for t1 in dde.delays:
        for t2 in dde.delays:
            if t1 + t2 < T:
                bps.add(round(t1 + t2, 12))
    grid = sorted(bps)
    # refine so each piece is at most `step` long (delayed arguments always in the past)
    fine = [grid[0]]
    for a, b in zip(grid[:-1], grid[1:]):
        n = max(1, int(np.ceil((b - a) / step - 1e-12)))
        fine += list(np.linspace(a, b, n + 1)[1:])
    y = np.asarray(phi(0.0), float).reshape(g)
    for a, b in zip(fine[:-1], fine[1:]):
        def rhs(t, x):
            out = dde.A @ x
            for tau, B in zip(dde.delays, dde.Bs):
                out = out + B @ xval(t - tau)
            return out
        sol = solve_ivp(rhs, (a, b), y, method="DOP853", rtol=rtol, atol=atol, dense_output=True)
        pieces.append((a, b, sol.sol))
        y = sol.y[:, -1]
    return xval
