"""
LCHS circuits for three equation classes, compiled end to end by the qfde compiler (spec file -> LK functional
-> SBP discretization -> LK-weighted generator -> LCHS circuit), in simulation.  The hardware runs are in
benchmarks/hardware.py.

Instances (examples/demo_*.toml; exact characteristic-mode histories, so the true solution is e^{lambda t}):
  delay equation          x' = -2x + 1.2 x(t-1)
  neutral delay equation  d/dt[x - 0.3 x(t-1)] = -2x + 1.2 x(t-1)
  integro-differential    x' = -1.5x + int_0^1 1.2 cos(pi s/2) x(t-s) ds
Each at degree 2 on one element: 4 unknowns = 2 system qubits.

Experiments
  E1  with the LK weight vs without it (control: shifted by mu = lambda_max(sym G), output rescaled by e^{mu t})
  E2  the same compiled pipeline on three classes
  E3  the whole memory window x(t + theta) read out from the system register
  C   noiseless circuit simulation: output vs number of LCHS nodes M and degree p (end-to-end convergence)

Usage:
  python3 benchmarks/lchs_demo.py            # ideal + FakeMarrakesh noisy + convergence -> results/lchs_demo.json
"""
import sys, os, json, math, tomllib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.linalg import expm
from qfde import spec as S, compiler
from qfde import lchs_circuit as LC

OUT = os.path.join(ROOT, "results", "lchs_demo.json")
SPECS = [("delay", "demo_dde.toml", -0.33031401751431055),
         ("neutral", "demo_neutral.toml", -0.39483263668935853),
         ("integro-differential", "demo_ide.toml", -0.5563891948650157)]
TIMES = [0.5, 1.0, 1.5, 2.0]
M_HW = 4
BETA = 0.8
SHOTS = 4000
BACKEND = "ibm_marrakesh"
KGRID = np.geomspace(0.25, 80, 60)


def load(spec_file, p=None):
    sp = S.parse(tomllib.load(open(os.path.join(ROOT, "examples", spec_file), "rb")))
    if p:
        sp["degree"] = p
    rep, lo = compiler.compile_spec(sp, lowered=True)
    G, W, z0, R = lo["G"], lo["W"], lo["z0"], lo["R"]
    assert np.allclose(W, np.diag(np.diag(W))), "demo instances have a diagonal LK weight"
    wd = np.diag(W)
    Gh = (np.sqrt(wd)[:, None] * G) / np.sqrt(wd)[None, :]
    mu = float(np.linalg.eigvalsh((G + G.T) / 2).max())
    # window nodes (theta values) and the e^{-beta theta/2} factor of the stored history
    disc = lo["disc"]
    th = np.full(disc.size, np.nan)
    for e in disc.win:
        for jl, j in enumerate(e.idx):
            th[j] = e.nodes[jl]
    return dict(rep=rep, G=G, wd=wd, Gh=Gh, z0=z0, R=R, mu=mu, theta=th, beta_w=disc.beta, n=disc.size,
                lam_w=float(np.linalg.eigvalsh((Gh + Gh.T) / 2).max()))


def setups(inst, t, M=None):
    """Weighted run and unweighted (shifted) control for time t: generator, initial vector, readout scale."""
    M = M or M_HW
    out = {}
    for mode in ("weighted", "control"):
        if mode == "weighted":
            Gs, v0, shift, unw = inst["Gh"], np.sqrt(inst["wd"]) * inst["z0"], 0.0, np.sqrt(inst["wd"])
        else:
            Gs, v0, shift, unw = inst["G"] - inst["mu"] * np.eye(inst["n"]), inst["z0"], inst["mu"], np.ones(inst["n"])
        K, kerr = LC.choose_K(Gs, [t], M, BETA, Ks=KGRID, out=inst["R"] / unw[None, :])
        k, c = LC.nodes(M, K, BETA)
        out[mode] = dict(Gs=Gs, v0=v0, shift=shift, unw=unw, K=float(K), op_err=float(kerr), k=k, c=c)
    return out


def readout(p, st, inst, t, n):
    """x(t) and the window values from P(anc=0, sys=i) (amplitudes are positive for these instances)."""
    meta = dict(s=float(np.sum(np.abs(st["c"]))), norm_v0=float(np.linalg.norm(st["v0"])))
    v = LC.amplitudes(p[:n], meta, st["shift"], t)
    z = v / st["unw"]
    x = float(np.real((inst["R"] @ z)[0]))
    win = {f"{th:.3f}": float(z[j] * math.exp(-inst["beta_w"] * th / 2))
           for j, th in enumerate(inst["theta"]) if not np.isnan(th)}
    return x, win


def exact_window(inst, lam, t):
    return {f"{th:.3f}": math.exp(lam * (t + th)) for th in inst["theta"] if not np.isnan(th)}


def mitigate(counts, cal0, cal1, nq):
    """Tensored readout correction from all-0 / all-1 calibration counts; returns corrected probabilities."""
    def marg(cnt, q, bit):
        tot = sum(cnt.values())
        return sum(v for b, v in cnt.items() if b.replace(" ", "")[::-1][q] == bit) / tot
    Ainv = []
    for q in range(nq):
        e01 = marg(cal0, q, "1"); e10 = marg(cal1, q, "0")
        A = np.array([[1 - e01, e10], [e01, 1 - e10]])
        Ainv.append(np.linalg.inv(A))
    tot = sum(counts.values())
    p = np.zeros(1 << nq)
    for b, v in counts.items():
        p[int(b.replace(" ", ""), 2)] += v / tot
    p = p.reshape([2] * nq)            # axes: highest qubit first
    for q in range(nq):
        ax = nq - 1 - q
        p = np.moveaxis(np.tensordot(Ainv[q], np.moveaxis(p, ax, 0), axes=1), 0, ax)
    p = np.clip(p.reshape(-1), 0, None)
    return p / p.sum()


def counts_to_p(counts, nq):
    tot = sum(counts.values()); p = np.zeros(1 << nq)
    for b, v in counts.items():
        p[int(b.replace(" ", ""), 2)] += v / tot
    return p


# ------------------------------------------------------------------------------------------------ simulation
def sim():
    from qiskit import transpile, QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    fake = FakeMarrakesh(); noisy = AerSimulator.from_backend(fake)
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    R["meta"] = dict(times=TIMES, M_hw=M_HW, beta_lchs=BETA, shots=SHOTS, backend=BACKEND)
    if "conv-only" in sys.argv:
        SPEC_LOOP = []
    else:
        R["instances"] = {}
        SPEC_LOOP = SPECS
    for name, f, lam in SPEC_LOOP:
        inst = load(f)
        I = dict(spec=f, lam=lam, unknowns=inst["n"], mu_unweighted=inst["mu"], lam_weighted=inst["lam_w"],
                 discretization=inst["rep"]["discretisation"], runs=[])
        print("==", name, "mu =", round(inst["mu"], 3), "weighted lambda_max =", inst["lam_w"], flush=True)
        for t in TIMES:
            sts = setups(inst, t)
            x_disc = float((inst["R"] @ expm(inst["G"] * t) @ inst["z0"])[0])
            for mode, st in sts.items():
                qc, meta = LC.circuit(st["Gs"], st["v0"], t, st["k"], st["c"])
                ns, na = meta["ns"], meta["na"]; nq = ns + na
                # ideal: exact probabilities, and a statevector check of the circuit itself
                p_ideal = LC.ideal_probs(st["Gs"], st["v0"], t, st["k"], st["c"])
                sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False)).probabilities()
                assert np.allclose(sv[: 1 << ns], p_ideal, atol=1e-10)
                x_ideal, win_ideal = readout(p_ideal, st, inst, t, inst["n"])
                # noisy (FakeMarrakesh) with the same readout correction used on hardware
                tq = min((transpile(qc, fake, optimization_level=3, seed_transpiler=s) for s in range(6)),
                         key=lambda c: c.count_ops().get("cz", 0))
                layout = tq.layout.final_index_layout()[:nq]
                cal = []
                for allone in (False, True):
                    cq = QuantumCircuit(nq, nq)
                    if allone:
                        cq.x(range(nq))
                    cq.measure(range(nq), range(nq))
                    cal.append(transpile(cq, fake, initial_layout=layout, optimization_level=0))
                res = noisy.run([tq] + cal, shots=SHOTS, seed_simulator=11).result()
                cnt = res.get_counts(0)
                p_raw = counts_to_p(cnt, nq)
                p_mit = mitigate(cnt, res.get_counts(1), res.get_counts(2), nq)
                x_raw, _ = readout(p_raw, st, inst, t, inst["n"])
                x_mit, win_mit = readout(p_mit, st, inst, t, inst["n"])
                row = dict(t=t, mode=mode, K=st["K"], lchs_op_err=st["op_err"], shift=st["shift"],
                           cz=int(tq.count_ops().get("cz", 0)), depth=int(tq.depth()),
                           p_success_ideal=float(p_ideal.sum()), p_success_noisy=float(p_mit[: 1 << ns].sum()),
                           x_true=math.exp(lam * t), x_disc=x_disc, x_ideal_circuit=x_ideal,
                           x_noisy_raw=x_raw, x_noisy_mitigated=x_mit,
                           window_true=exact_window(inst, lam, t), window_ideal=win_ideal, window_noisy=win_mit)
                I["runs"].append(row)
                print(f"  t={t} {mode:8s} K={st['K']:.2f} CZ={row['cz']} Psucc ideal={row['p_success_ideal']:.3f} "
                      f"noisy={row['p_success_noisy']:.3f} | x true={row['x_true']:.4f} ideal={x_ideal:.4f} "
                      f"noisy={x_mit:.4f}", flush=True)
        R["instances"][name] = I
        json.dump(R, open(OUT, "w"), indent=1)
    # ---- noiseless circuit simulation: convergence in M and p (weighted), and E1 (weighted vs control at p = 2)
    conv = {}
    for name, f, lam in SPECS:
        rows = []
        for p in (2, 3, 4, 6):
            inst = load(f, p=p)
            for M in (4, 8, 16, 32, 64, 128, 256):
                for mode in (("weighted", "control") if p == 2 else ("weighted",)):
                    errs, ps, Ks = [], [], []
                    for t in TIMES:
                        st = setups(inst, t, M)[mode]
                        qc, meta = LC.circuit(st["Gs"], st["v0"], t, st["k"], st["c"], measure=False)
                        amp = Statevector.from_instruction(qc).data[: 1 << meta["ns"]]
                        v = amp[: inst["n"]] * meta["s"] * meta["norm_v0"] * math.exp(st["shift"] * t)
                        x = float(np.real((inst["R"] @ (v / st["unw"]))[0]))
                        errs.append(abs(x - math.exp(lam * t)) / math.exp(lam * t))
                        ps.append(float(np.sum(np.abs(amp) ** 2))); Ks.append(st["K"])
                    rows.append(dict(p=p, unknowns=inst["n"], M=M, mode=mode, err=float(max(errs)),
                                     err_by_t=[float(e) for e in errs], p_success=ps, K=Ks))
                    print(f"  conv {name} p={p} M={M} {mode}: max rel err {max(errs):.2e}, "
                          f"P(success) {min(ps):.3g}-{max(ps):.3g}", flush=True)
        conv[name] = rows
        R["convergence"] = conv
        json.dump(R, open(OUT, "w"), indent=1)


# ------------------------------------------------------------------------------------------------ hardware
if __name__ == "__main__":
    sim()
