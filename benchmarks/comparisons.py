"""
High-precision comparison on the five instances of benchmarks/instances.py (results/instances.json).

For every target accuracy eps in EPS:
  ours   cheapest measured configuration ((r,) p, h) whose error against the exact solution is <= eps;
         block-encoding queries of improved LCHS (qfde.lchs.query_count) and initial-data loads (= chi).
  prior  delay and Gaussian instances: first-order time stepping whose all-at-once system the optimal QLSA
         solves (queries = 837 kappa log-rescaled x normalization ratio x decay; the same count of initial-data
         loads, since the adiabatic QLSA uses the right-hand side in every step); errors measured directly
         by time stepping, kappa measured to N*g <= 1.6e4 and fitted (kappa ~ N^1.00) beyond; where eps is
         below the measured range, N follows from the fitted first-order error rate (flagged).
         Erlang instances: the Schrodingerization bound of Setoyama-Fujii (their Eq. 36) on the exact linear
         chain, with their state prepared O(decay) times (comparable to ours).
Also: the Lyapunov-equation weight (X with G^T X + X G = -I) at the eps = 1e-6 configuration; the ablation
(shift penalty without any weight); error-corrected resource estimates at eps = 1e-6 and 1e-10.
Run: python3 benchmarks/comparisons.py -> results/comparisons.json
"""
import sys, os, json, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import numpy as np, warnings
warnings.filterwarnings("ignore")
from functools import lru_cache
from scipy.linalg import solve_continuous_lyapunov
from qfde import lchs, resources as rs, ftqc
lchs.truncation = lru_cache(maxsize=None)(lchs.truncation)
lchs.coefficient_norm = lru_cache(maxsize=None)(lchs.coefficient_norm)
import instances as INS

RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "comparisons.json")
EPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
D = json.load(open(os.path.join(RES, "instances.json")))


def ours(name, eps):
    rows = [r for r in D[name]["ours"] if r["err"] <= eps]
    if not rows:
        return None
    T = D[name]["T"]
    q = [(lchs.query_count(r["aL"], r["aH"], T, eps, r["amp"])[0], r) for r in rows]
    Q, r = min(q, key=lambda z: z[0])
    return dict(queries=Q, loads=r["amp"], cfg={k: r[k] for k in ("r", "p", "h", "size", "err", "aL", "aH", "amp", "cond_W", "sparsity")})


def first_order(rows, eps):
    ok = [r for r in rows if r["err"] <= eps]
    kap_fit = np.polyfit(np.log([r["N"] for r in rows if not r.get("kappa_extrapolated")]),
                         np.log([r["kappa"] for r in rows if not r.get("kappa_extrapolated")]), 1)
    if ok:
        r = min(ok, key=lambda r: r["N"])
        return dict(N=r["N"], kappa=r["kappa"], err_measured=True, kappa_measured=not r.get("kappa_extrapolated", False))
    tail = rows[-4:]
    q = -np.polyfit(np.log([r["N"] for r in tail]), np.log([r["err"] for r in tail]), 1)[0]
    N = rows[-1]["N"] * (rows[-1]["err"] / eps) ** (1 / q)
    return dict(N=float(N), kappa=float(np.exp(np.polyval(kap_fit, np.log(N)))), err_measured=False,
                kappa_measured=False, rate=float(q))


def prior(name, eps):
    d = D[name]
    if name.startswith("erlang"):
        m = int(name[-1]); mu = 2.0
        n = m + 1; M = np.zeros((n, n)); M[0, 0] = -1.0; M[0, m] = 0.8
        for i in range(1, m + 1):
            M[i, 0 if i == 1 else i - 1] += mu; M[i, i] += -mu
        s = int((np.abs(M) > 0).sum(axis=1).max()); Cmax = float(np.abs(M).max())
        g0 = d["decay"]
        return dict(queries=rs.schrodingerisation_queries(s, d["T"], Cmax, eps, g0), loads=g0, method="Schrodingerization",
                    err_measured=True, bound=True)
    fo = first_order(d["prior"], eps)
    ratio = [r["ratio"] for r in d["prior"] if r.get("ratio")]
    ratio = ratio[-1] if ratio else 1.0
    Q = rs.qlsa_queries(fo["kappa"], eps, ratio, d["decay"])
    loads = rs.QLSA_C * math.log(1 / eps) * fo["kappa"] * d["decay"]
    return dict(queries=Q, loads=loads, method="first-order + QLSA", **fo)


def lyapunov_alt(name, cfg):
    """Our algorithm with the weight from a Lyapunov equation instead of the LK functional."""
    S = INS.spec(name)
    fde, disc, G, Gh, Wh, Wih, cert, W = INS.lowered(S, cfg["p"], cfg["h"], cfg["r"])
    X = solve_continuous_lyapunov(G.T, -np.eye(G.shape[0])); X = (X + X.T) / 2
    lx, V = np.linalg.eigh(X); Xh = (V * np.sqrt(lx)) @ V.T; Xih = (V / np.sqrt(lx)) @ V.T
    GX = Xh @ G @ Xih
    LX, HX = lchs.split(GX)
    aL, aH = lchs.sparse_norm(LX)[0], lchs.sparse_norm(HX)[0]
    from qfde.fde import initial_state_general
    z0 = initial_state_general(disc)
    amp = float(np.linalg.norm(Xh @ z0) / (math.sqrt(lx.min()) * D[name]["env"]))
    eps = 1e-6
    return dict(cond=float(lx.max() / lx.min()), sparsity=lchs.sparse_norm(GX)[1], aL=aL, aH=aH, amp=amp,
                lam=float(np.linalg.eigvalsh((GX + GX.T) / 2).max()),
                queries=lchs.query_count(aL, aH, D[name]["T"], eps, amp)[0],
                ours=dict(cond=cfg["cond_W"], sparsity=cfg["sparsity"], aL=cfg["aL"], aH=cfg["aH"], amp=cfg["amp"],
                          queries=lchs.query_count(cfg["aL"], cfg["aH"], D[name]["T"], eps, cfg["amp"])[0]),
                size=G.shape[0])


def ablation(name, cfg):
    S = INS.spec(name)
    fde, disc, G, Gh, Wh, Wih, cert, W = INS.lowered(S, cfg["p"], cfg["h"], cfg["r"])
    lam = float(np.linalg.eigvalsh((G + G.T) / 2).max()); lw = np.linalg.eigvalsh(W)
    return dict(p=cfg["p"], h=cfg["h"], r=cfg["r"], size=G.shape[0], lam_unweighted=lam,
                lam_weighted=float(np.linalg.eigvalsh((Gh + Gh.T) / 2).max()),
                log10_shift_penalty=lam * D[name]["T"] / math.log(10), sqrt_condW=float(math.sqrt(lw.max() / lw.min())))


# ------------------------------------------------------------------ error-corrected resources
def _phys_worker(args, q):
    queries, nb, p = args
    q.put(ftqc.physical(ftqc.logical_counts(queries, ftqc.per_query_cost(node_bits=nb, p=p), nb)))


def fmt_time(s):
    for unit, sec in (("years", 3.156e7), ("days", 86400), ("hours", 3600), ("min", 60)):
        if s >= sec: return f"{s / sec:.0f} {unit}"
    return f"{s:.0f} s"


def physical_safe(queries, nb, p, timeout=90):
    """Microsoft estimator with a timeout.  For very large counts its factory search can stall; then we
    estimate at queries/10^k and scale runtime and T-states linearly at that code distance.  This
    UNDER-estimates the true runtime (the distance only grows with depth); it is used for prior methods only."""
    import multiprocessing as mp
    for k in (0, 3, 6):
        qq = mp.Queue(); pr = mp.Process(target=_phys_worker, args=((queries / 10 ** k, nb, p), qq))
        pr.start(); pr.join(timeout)
        if pr.is_alive():
            pr.terminate(); pr.join(); continue
        if pr.exitcode != 0:
            raise RuntimeError("estimator worker failed")
        r = qq.get(); r["scaled_by"] = 10 ** k
        if k:
            r["runtime_s"] *= 10 ** k; r["t_states"] = int(r["t_states"] * 10 ** k)
            r["runtime_formatted"] = fmt_time(r["runtime_s"]) + " (lower bound)"
        return r
    raise RuntimeError("estimator did not finish")


def ft(o, pr, cfg, prior_size):
    no = max(3, math.ceil(math.log2(max(2, cfg["size"]))), math.ceil(math.log2(2 * cfg["p"] + 3)))
    npr = max(3, math.ceil(math.log2(max(2, prior_size))))
    ph_o = physical_safe(o, no, cfg["p"])
    ph_p = physical_safe(pr, npr, 1)
    keep = ("runtime_s", "runtime_formatted", "physical_qubits", "logical_qubits", "scaled_by")
    return dict(ours={k: ph_o[k] for k in keep}, prior={k: ph_p[k] for k in keep}, ratio=ph_p["runtime_s"] / ph_o["runtime_s"])


def main():
    R = dict(eps=EPS, instances={})
    for name in INS.NAMES:
        if name not in D:
            continue
        rows = []
        for eps in EPS:
            o = ours(name, eps); p_ = prior(name, eps)
            rows.append(dict(eps=eps, ours=o, prior=p_,
                             ratio=(p_["queries"] / o["queries"]) if o else None,
                             loads_ratio=(p_["loads"] / o["loads"]) if o else None))
            print(f"{name:11s} eps={eps:.0e}: ours {o['queries'] if o else float('nan'):.3g} prior {p_['queries']:.3g} "
                  f"x{(p_['queries'] / o['queries']) if o else float('nan'):.3g}  loads {o['loads'] if o else float('nan'):.3g} vs "
                  f"{p_['loads']:.3g}  {'' if p_.get('err_measured', True) else '(prior extrapolated)'}", flush=True)
        rec = dict(label=D[name]["label"], family=D[name]["family"], prior_method=prior(name, 1e-3)["method"],
                   sweep=rows)
        c6 = ours(name, 1e-6)
        if c6:
            rec["lyapunov"] = lyapunov_alt(name, c6["cfg"])
            rec["ablation"] = ablation(name, c6["cfg"])
            c10 = ours(name, 1e-10) or ours(name, 1e-9)
            rec["ablation_hi"] = ablation(name, c10["cfg"])
            rec["ft"] = {}
            for e in (1e-6, 1e-10):
                o = ours(name, e)
                if o:
                    pe = prior(name, e)
                    psize = (int(name[-1]) + 1) if name.startswith("erlang") else pe["N"] * (2 if name in ("drug", "baroreflex") else 1)
                    rec["ft"][str(e)] = ft(o["queries"], pe["queries"], o["cfg"], psize)
                    print(f"   FT {e:.0e}: ours {rec['ft'][str(e)]['ours']['runtime_formatted']} vs prior "
                          f"{rec['ft'][str(e)]['prior']['runtime_formatted']}", flush=True)
            L = rec["lyapunov"]
            print(f"   Lyapunov weight: cond {L['cond']:.3g} (ours {L['ours']['cond']:.3g}), sparsity {L['sparsity']} "
                  f"(ours {L['ours']['sparsity']}), queries {L['queries']:.3g} (ours {L['ours']['queries']:.3g})", flush=True)
        R["instances"][name] = rec
        json.dump(R, open(OUT, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
