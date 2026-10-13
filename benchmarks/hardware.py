"""
Validation on IBM hardware (ibm_marrakesh: IBM Heron r2, 156 physical qubits, NOT error-corrected).

Every instance is compiled from a spec file in examples/ by qfde.compiler (LK functional -> SBP discretization
-> weighted generator), at the smallest size that fits (4-6 unknowns, 2-3 system qubits).

V1  Lyapunov-Krasovskii energy law (main theorem).  For each class, the weighted propagator V(t) = W^{1/2} U(t)
    W^{-1/2} is a contraction, so it has a one-ancilla unitary dilation [[V, (I-VV*)^1/2], [(I-V*V)^1/2, -V*]].
    Preparing v and measuring the ancilla gives P(0) = ||V v||^2/||v||^2 = E(t)/E(0), the ratio of the discrete
    LK energy z^T W z.  Theorem: E(t)/E(0) <= 1 and non-increasing, for every state.  Series:
      w-mode   weighted, starting from the solution's own history (dominant characteristic mode)
      w-worst  weighted, starting from the top eigenvector of sym(G_hat) (slowest possible energy decay)
      u-worst  unweighted control: Euclidean energy ||U(t) z||^2/||z||^2 from the top eigenvector of sym(G);
               U(t) is not a contraction, so it is dilated after division by c = ||U(t)||_2 and the
               measured P(0) is multiplied by c^2.
    These circuits apply a classically computed propagator: they test that the weighted dynamics is a
    physically realizable contraction, not the algorithm.
V2  The algorithm: 4-term LCHS circuits (qfde.lchs_circuit) for the four classes that fit in 2 system qubits.
V3  Memory readout: x(t + theta) at every window node from the V2 populations.

Error mitigation: Pauli twirling of gates and measurements, XY4 dynamical decoupling, tensored readout
correction from calibration circuits in the same job, and zero-noise extrapolation by global folding
(noise factors 1 and 3, linear extrapolation).

Usage: python3 benchmarks/hardware.py ideal | noise | submit | fetch     (submit/fetch need IBM_TOKEN, IBM_CRN)
"""
import sys, os, json, math, tomllib, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from qfde import spec as S, compiler
from qfde import lchs_circuit as LC
from qfde.fde import make_generator

OUT = os.path.join(ROOT, "results", "hardware.json")
BACKEND = "ibm_marrakesh"
V1_CLASSES = [("delay, one delay", "demo_dde.toml"), ("neutral", "demo_neutral.toml"),
              ("time-varying delay", "demo_time_varying.toml"), ("delay, two delays", "demo_two_delay.toml"),
              ("integro-differential, bounded kernel", "demo_ide.toml"),
              ("integro-differential, fading memory", "demo_fading.toml"),
              ("drug reabsorption", "demo_drug.toml"), ("baroreflex", "demo_baroreflex.toml")]
V2_CLASSES = [("delay, one delay", "demo_dde.toml", -0.33031401751431055),
              ("neutral", "demo_neutral.toml", -0.39483263668935853),
              ("integro-differential, bounded kernel", "demo_ide.toml", -0.5563891948650157),
              ("integro-differential, fading memory", "demo_fading.toml", -0.13985294912645574)]
V1_TIMES = [0.1, 0.25, 0.5, 1.0]
V2_TIMES = [0.5, 1.0, 1.5, 2.0]
SHOTS_V1, SHOTS_V2 = 2000, 4000
FOLDS = (1, 3)


# ------------------------------------------------------------------------------------------------ instances
def compile_file(f):
    sp = S.parse(tomllib.load(open(os.path.join(ROOT, "examples", f), "rb")))
    rep, lo = compiler.compile_spec(sp, lowered=True)
    W = lo["W"]; lw, V = np.linalg.eigh(W)
    Wh = (V * np.sqrt(lw)) @ V.T; Wih = (V / np.sqrt(lw)) @ V.T
    Gof = make_generator(lo["disc"])
    return dict(sp=sp, rep=rep, lo=lo, Wh=Wh, Wih=Wih, Gof=(lambda t: Gof(t)[0]), td=sp["time_dependent"],
                n=lo["G"].shape[0])


def propagator(I, t):
    if not I["td"]:
        return expm(I["lo"]["G"] * t)
    n = I["n"]
    f = lambda s, y: (I["Gof"](s) @ y.reshape(n, n)).reshape(-1)
    sol = solve_ivp(f, (0, t), np.eye(n).reshape(-1), method="DOP853", rtol=1e-12, atol=1e-14)
    return sol.y[:, -1].reshape(n, n)


def pad(M, N):
    P = np.eye(N, dtype=complex); P[:M.shape[0], :M.shape[1]] = M
    return P


def dilation(C):
    """Halmos dilation [[C, D_{C*}], [D_C, -C*]] with defect operators from the SVD of C (singular values
    clipped to <= 1), then projected onto the nearest unitary to remove round-off."""
    U, s, Vh = np.linalg.svd(C)
    s = np.clip(s, 0.0, 1.0); d = np.sqrt(1.0 - s ** 2)
    Cc = (U * s) @ Vh
    DC = (Vh.conj().T * d) @ Vh                 # (I - C*C)^{1/2}
    DCs = (U * d) @ U.conj().T                  # (I - CC*)^{1/2}
    M = np.block([[Cc, DCs], [DC, -Cc.conj().T]])
    u, _, vh = np.linalg.svd(M)
    return u @ vh


def v1_setups(I):
    """List of dict(series, t, C (contraction, padded), v (padded state), scale c^2, ideal energy ratio)."""
    n = I["n"]; N = 1 << max(1, math.ceil(math.log2(n)))
    G0 = I["Gof"](0.0); Gh0 = I["Wh"] @ G0 @ I["Wih"]
    z0 = I["lo"]["z0"]
    states = {"w-mode": I["Wh"] @ z0,
              "w-worst": np.linalg.eigh((Gh0 + Gh0.T) / 2)[1][:, -1],
              "u-worst": np.linalg.eigh((G0 + G0.T) / 2)[1][:, -1]}
    out = []
    for t in V1_TIMES:
        U = propagator(I, t); Vw = I["Wh"] @ U @ I["Wih"]
        for series, v in states.items():
            if series.startswith("w"):
                C = Vw; c2 = 1.0
            else:
                c = float(np.linalg.norm(U, 2)); C = U / c; c2 = c * c
            vv = np.zeros(N, complex); vv[:n] = v; vv /= np.linalg.norm(vv)
            Cp = pad(C, N)
            p0 = float(np.linalg.norm(Cp @ vv) ** 2)
            out.append(dict(series=series, t=t, C=Cp, v=vv, c2=c2, p0_ideal=p0, energy_ideal=p0 * c2,
                            norm_check=float(np.linalg.norm(C, 2))))
    return out


def v1_circuit(C, v):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation, UnitaryGate
    N = C.shape[0]; ns = int(math.log2(N))
    qc = QuantumCircuit(ns + 1)
    qc.append(StatePreparation(v), list(range(ns)))
    qc.append(UnitaryGate(dilation(C), label="dilation"), list(range(ns + 1)))   # ancilla = qubit ns (MSB)
    return qc, [ns]


def v2_setups(I, lam):
    """4-term LCHS (as in benchmarks/lchs_demo.py), K chosen per t on the output functional."""
    wd = np.diag(I["lo"]["W"]); assert np.allclose(I["lo"]["W"], np.diag(wd))
    Gh = (np.sqrt(wd)[:, None] * I["lo"]["G"]) / np.sqrt(wd)[None, :]
    v0 = np.sqrt(wd) * I["lo"]["z0"]; R = I["lo"]["R"]
    th = np.full(I["n"], np.nan)
    for e in I["lo"]["disc"].win:
        for jl, j in enumerate(e.idx):
            th[j] = e.nodes[jl]
    out = []
    for t in V2_TIMES:
        K, _ = LC.choose_K(Gh, [t], 4, 0.8, Ks=np.geomspace(0.25, 80, 60), out=R / np.sqrt(wd)[None, :])
        k, c = LC.nodes(4, K, 0.8)
        out.append(dict(t=t, K=float(K), k=k, c=c, Gh=Gh, v0=v0, wd=wd, R=R, theta=th, beta=I["lo"]["disc"].beta,
                        x_true=math.exp(lam * t)))
    return out


def v2_readout(p, st):
    s = float(np.sum(np.abs(st["c"]))); nv = float(np.linalg.norm(st["v0"]))
    n = len(st["wd"])
    v = np.sqrt(np.maximum(p[:n], 0)) * s * nv
    z = v / np.sqrt(st["wd"])
    x = float(np.real((st["R"] @ z)[0]))
    win = {f"{th:.3f}": float(z[j] * math.exp(-st["beta"] * th / 2)) for j, th in enumerate(st["theta"]) if not np.isnan(th)}
    return x, win


# ------------------------------------------------------------------------------------------------ circuits
def fold(tq, k, backend):
    """Global folding C (C^-1 C)^((k-1)/2), kept at ISA level (no optimization)."""
    from qiskit import transpile
    if k == 1:
        return tq
    inv = transpile(tq.inverse(), backend, optimization_level=0)
    out = tq.copy()
    for _ in range((k - 1) // 2):
        out = out.compose(inv).compose(tq)
    return out


def isa(qc, backend, seeds=6):
    from qiskit import transpile
    return min((transpile(qc, backend, optimization_level=3, seed_transpiler=s) for s in range(seeds)),
               key=lambda c: c.count_ops().get("cz", 0))


def measured(tq, virtual, src):
    """Append measurements of the given virtual qubits (in order) to an ISA circuit; return circuit, physical qubits.
    src is the unfolded transpiled circuit, whose layout maps virtual to physical qubits."""
    from qiskit import ClassicalRegister
    lay = src.layout.final_index_layout()
    phys = [int(lay[q]) for q in virtual]
    c = tq.copy(); cr = ClassicalRegister(len(phys), "c"); c.add_register(cr)
    for i, q in enumerate(phys):
        c.measure(q, cr[i])
    return c, phys


def calibration(phys_sets, backend):
    from qiskit import QuantumCircuit, transpile
    circs, idx = [], []
    for ps in phys_sets:
        for one in (0, 1):
            qc = QuantumCircuit(backend.num_qubits, len(ps))
            for i, q in enumerate(ps):
                if one:
                    qc.x(q)
                qc.measure(q, i)
            circs.append(transpile(qc, backend, optimization_level=0)); idx.append(dict(cal=one, phys=list(ps)))
    return circs, idx


def counts_p(cnt, nb):
    tot = sum(cnt.values()); p = np.zeros(1 << nb)
    for b, v in cnt.items():
        p[int(b.replace(" ", ""), 2)] += v / tot
    return p


def correct(p, cal0, cal1, nb):
    """Tensored readout correction (bit i = i-th measured qubit, little-endian)."""
    def marg(cnt, q, bit):
        tot = sum(cnt.values())
        return sum(v for b, v in cnt.items() if b.replace(" ", "")[::-1][q] == bit) / tot
    t = p.reshape([2] * nb)
    for q in range(nb):
        e01 = marg(cal0, q, "1"); e10 = marg(cal1, q, "0")
        Ainv = np.linalg.inv(np.array([[1 - e01, e10], [e01, 1 - e10]]))
        ax = nb - 1 - q
        t = np.moveaxis(np.tensordot(Ainv, np.moveaxis(t, ax, 0), axes=1), 0, ax)
    t = np.clip(t.reshape(-1), 0, None)
    return t / t.sum()


def build_all(backend):
    """All V1 and V2 circuits (ISA, folded, measured) with their metadata."""
    circs, meta = [], []
    for name, f in V1_CLASSES:
        I = compile_file(f)
        for st in v1_setups(I):
            qc, anc = v1_circuit(st["C"], st["v"])
            tq = isa(qc, backend)
            for k in FOLDS:
                c, phys = measured(fold(tq, k, backend), anc, tq)
                circs.append(c)
                meta.append(dict(exp="V1", cls=name, series=st["series"], t=st["t"], fold=k, phys=phys,
                                 c2=st["c2"], p0_ideal=st["p0_ideal"], energy_ideal=st["energy_ideal"],
                                 cz=int(tq.count_ops().get("cz", 0))))
    for name, f, lam in V2_CLASSES:
        I = compile_file(f)
        for st in v2_setups(I, lam):
            qc, m = LC.circuit(st["Gh"], st["v0"], st["t"], st["k"], st["c"], measure=False)
            tq = isa(qc, backend)
            nq = m["ns"] + m["na"]
            for k in FOLDS:
                c, phys = measured(fold(tq, k, backend), list(range(nq)), tq)
                circs.append(c)
                p_ideal = LC.ideal_probs(st["Gh"], st["v0"], st["t"], st["k"], st["c"])
                x_ideal, win_ideal = v2_readout(p_ideal, st)
                meta.append(dict(exp="V2", cls=name, t=st["t"], fold=k, phys=phys, K=st["K"],
                                 cz=int(tq.count_ops().get("cz", 0)), x_true=st["x_true"], x_ideal=x_ideal,
                                 win_ideal=win_ideal, p_ideal=p_ideal.tolist(), ns=m["ns"], na=m["na"]))
    return circs, meta


# ------------------------------------------------------------------------------------------------ analysis
def analyze(meta, counts, cals):
    """Readout-correct, extrapolate to zero noise, compare with ideal."""
    groups = {}
    for m, cnt in zip(meta, counts):
        key = (m["exp"], m["cls"], m.get("series"), m["t"])
        nb = len(m["phys"]); cal = cals[tuple(m["phys"])]
        p_raw = counts_p(cnt, nb); p = correct(p_raw, cal[0], cal[1], nb)
        groups.setdefault(key, {})[m["fold"]] = (m, p, p_raw)
    rows = []
    for key, d in groups.items():
        m = d[1][0]
        if key[0] == "V1":
            P = {k: d[k][1][0] for k in d}; Praw = d[1][2][0]
            pz = (3 * P[1] - P[3]) / 2 if 3 in P else P[1]
            rows.append(dict(exp="V1", cls=m["cls"], series=m["series"], t=m["t"], cz=m["cz"],
                             energy_ideal=m["energy_ideal"], energy_raw=Praw * m["c2"], energy_corr=P[1] * m["c2"],
                             energy_zne=pz * m["c2"], fold3=P.get(3, np.nan) * m["c2"]))
        else:
            P1 = d[1][1]; P3 = d[3][1] if 3 in d else None
            pz = np.clip((3 * P1 - P3) / 2, 0, None) if P3 is not None else P1
            rows.append(dict(exp="V2", cls=m["cls"], t=m["t"], cz=m["cz"], x_true=m["x_true"], x_ideal=m["x_ideal"],
                             win_ideal=m["win_ideal"], p_ideal=m["p_ideal"], p_corr=P1[: 1 << m["ns"]].tolist(),
                             p_zne=pz[: 1 << m["ns"]].tolist(), p_raw=d[1][2][: 1 << m["ns"]].tolist()))
    return rows


def v2_x(rows):
    """Attach x and window readouts to V2 rows (needs the setups again)."""
    for name, f, lam in V2_CLASSES:
        I = compile_file(f)
        for st in v2_setups(I, lam):
            for r in rows:
                if r["exp"] == "V2" and r["cls"] == name and abs(r["t"] - st["t"]) < 1e-12:
                    for tag in ("corr", "zne", "raw"):
                        p = np.zeros(1 << 4); p[: len(r["p_" + tag])] = r["p_" + tag]
                        x, win = v2_readout(p, st)
                        r["x_" + tag] = x; r["win_" + tag] = win
    return rows


# ------------------------------------------------------------------------------------------------ modes
def ideal():
    R = {"V1": [], "V2": []}
    for name, f in V1_CLASSES:
        I = compile_file(f)
        for st in v1_setups(I):
            R["V1"].append(dict(cls=name, series=st["series"], t=st["t"], energy=st["energy_ideal"], p0=st["p0_ideal"],
                                c2=st["c2"], n=I["n"], contraction=st["norm_check"]))
    for name, f, lam in V2_CLASSES:
        I = compile_file(f)
        for st in v2_setups(I, lam):
            p = LC.ideal_probs(st["Gh"], st["v0"], st["t"], st["k"], st["c"]); x, win = v2_readout(p, st)
            R["V2"].append(dict(cls=name, t=st["t"], K=st["K"], x_ideal=x, x_true=st["x_true"], p_success=float(p.sum())))
    return R


def noise():
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    fake = FakeMarrakesh(); sim = AerSimulator.from_backend(fake)
    circs, meta = build_all(fake)
    phys_sets = sorted({tuple(m["phys"]) for m in meta})
    cc, ci = calibration(phys_sets, fake)
    res = sim.run(circs + cc, shots=SHOTS_V2, seed_simulator=7).result()
    counts = [res.get_counts(i) for i in range(len(circs))]
    cals = {}
    for j, ix in enumerate(ci):
        cals.setdefault(tuple(ix["phys"]), [None, None])[ix["cal"]] = res.get_counts(len(circs) + j)
    rows = v2_x(analyze(meta, counts, cals))
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    R["noise_model"] = rows; R["ideal"] = ideal()
    R["circuits"] = dict(n=len(circs), cz=[m["cz"] for m in meta if m["fold"] == 1])
    json.dump(R, open(OUT, "w"), indent=1, default=float)
    return rows


def service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["IBM_TOKEN"], instance=os.environ["IBM_CRN"])


def submit():
    from qiskit_ibm_runtime import SamplerV2
    bk = service().backend(BACKEND)
    circs, meta = build_all(bk)
    phys_sets = sorted({tuple(m["phys"]) for m in meta})
    cc, ci = calibration(phys_sets, bk)
    jobs = []
    for exp, shots in (("V1", SHOTS_V1), ("V2", SHOTS_V2)):
        sel = [i for i, m in enumerate(meta) if m["exp"] == exp]
        sampler = SamplerV2(mode=bk)
        o = sampler.options
        o.default_shots = shots
        o.dynamical_decoupling.enable = True; o.dynamical_decoupling.sequence_type = "XY4"
        o.twirling.enable_gates = True; o.twirling.enable_measure = True
        o.twirling.num_randomizations = 32; o.twirling.shots_per_randomization = shots // 32
        job = sampler.run([circs[i] for i in sel] + cc)
        jobs.append(dict(exp=exp, job_id=job.job_id(), index=sel, shots=shots))
        print("submitted", exp, job.job_id(), len(sel), "circuits +", len(cc), "calibration")
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    R["hardware_jobs"] = dict(backend=BACKEND, jobs=jobs, cal_index=ci,
                              meta=[{k: v for k, v in m.items()} for m in meta],
                              submitted=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    json.dump(R, open(OUT, "w"), indent=1, default=float)


def fetch():
    R = json.load(open(OUT)); H = R["hardware_jobs"]; svc = service()
    meta = H["meta"]; counts = [None] * len(meta); cals = {}; usage = 0
    for jb in H["jobs"]:
        job = svc.job(jb["job_id"]); print(jb["exp"], job.status())
        res = job.result(); usage += job.usage() or 0
        cnts = [list(r.data.values())[0].get_counts() for r in res]
        for i, c in zip(jb["index"], cnts[: len(jb["index"])]):
            counts[i] = c
        for j, ix in enumerate(H["cal_index"]):
            cals.setdefault((jb["exp"], tuple(ix["phys"])), [None, None])[ix["cal"]] = cnts[len(jb["index"]) + j]
    # each experiment uses the calibration measured in its own job
    rows = []
    for exp in ("V1", "V2"):
        sel = [i for i, m in enumerate(meta) if m["exp"] == exp]
        cal = {k[1]: v for k, v in cals.items() if k[0] == exp}
        rows += analyze([meta[i] for i in sel], [counts[i] for i in sel], cal)
    R["hardware"] = v2_x(rows); R["hardware_usage_seconds"] = usage
    json.dump(R, open(OUT, "w"), indent=1, default=float)
    return R["hardware"]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ideal"
    if mode == "ideal":
        R = json.load(open(OUT)) if os.path.exists(OUT) else {}
        R["ideal"] = ideal(); json.dump(R, open(OUT, "w"), indent=1, default=float)
        for r in R["ideal"]["V1"]:
            print(f"{r['cls']:38s} {r['series']:8s} t={r['t']:<5} E/E0={r['energy']:.4f}  (n={r['n']}, ||C||={r['contraction']:.4f})")
        for r in R["ideal"]["V2"]:
            print(f"{r['cls']:38s} t={r['t']} K={r['K']:.2f} x ideal={r['x_ideal']:.4f} true={r['x_true']:.4f} Psucc={r['p_success']:.3f}")
    else:
        {"noise": noise, "submit": submit, "fetch": fetch}[mode]()


def ideal_dense(ts=np.linspace(0, 1.0, 41)):
    """Exact energy curves on a fine grid (for plotting)."""
    out = {}
    for name, f in V1_CLASSES:
        I = compile_file(f)
        G0 = I["Gof"](0.0); Gh0 = I["Wh"] @ G0 @ I["Wih"]
        states = {"w-mode": I["Wh"] @ I["lo"]["z0"], "w-worst": np.linalg.eigh((Gh0 + Gh0.T) / 2)[1][:, -1],
                  "u-worst": np.linalg.eigh((G0 + G0.T) / 2)[1][:, -1]}
        cur = {s: [] for s in states}
        for t in ts:
            U = propagator(I, t) if t > 0 else np.eye(I["n"])
            Vw = I["Wh"] @ U @ I["Wih"]
            for s, v in states.items():
                M = Vw if s.startswith("w") else U
                cur[s].append(float(np.linalg.norm(M @ v) ** 2 / np.linalg.norm(v) ** 2))
        out[name] = dict(t=list(map(float, ts)), **cur)
    R = json.load(open(OUT)); R["ideal_dense"] = out; json.dump(R, open(OUT, "w"), indent=1, default=float)
