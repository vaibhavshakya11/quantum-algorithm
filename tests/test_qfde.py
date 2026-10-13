"""Test suite for qfde.  Run:  python3 -m pytest -q tests/"""
import os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from scipy.linalg import expm
from qfde import core, lchs, verify
from qfde.fde import (build_fde, discretize_general, make_generator, weight_general, certify,
                      solve_general)

RNG = np.random.default_rng(2026)


def sym_max(M):
    return float(np.max(np.linalg.eigvalsh(0.5 * (M + M.T))))


# ---------------------------------------------------------------- SBP operators
@pytest.mark.parametrize("p", [1, 2, 3, 5, 8, 12, 16])
def test_sbp_identity_and_exactness(p):
    xi, w, D = core.lgl(p)
    Q = np.diag(w) @ D
    B = np.zeros((p + 1, p + 1)); B[0, 0] = -1; B[-1, -1] = 1
    assert np.abs(Q + Q.T - B).max() < 1e-12
    assert np.abs(D @ xi ** p - p * xi ** (p - 1)).max() < 1e-9 * max(1, p ** 2)
    assert abs(w.sum() - 2) < 1e-13


# ---------------------------------------------------------------- main theorem (property-based)
def random_system(seed):
    rng = np.random.default_rng(seed)
    g = int(rng.integers(1, 3))
    M = rng.normal(size=(g, g)); A = M - M.T - (1.5 + rng.random()) * np.eye(g)
    budget = -sym_max(A) * rng.uniform(0.2, 0.6)
    delays, neutral, var, qk = [], [], [], []
    for tau in sorted(rng.uniform(0.3, 1.5, size=int(rng.integers(1, 3)))):
        B = rng.normal(size=(g, g)); B *= 0.5 * budget / np.linalg.norm(B, 2); delays.append((float(tau), B))
    if rng.random() < 0.5:
        C = rng.normal(size=(g, g)); C *= 0.15 / np.linalg.norm(C, 2); neutral.append((float(rng.uniform(0.2, 0.6)), C))
    if rng.random() < 0.5:
        Bv = rng.normal(size=(g, g)); Bv *= 0.25 * budget / np.linalg.norm(Bv, 2)
        amp = float(rng.uniform(0.05, 0.25))
        var.append((lambda t, amp=amp: 0.8 + amp * np.sin(t), lambda t, amp=amp: amp * np.cos(t), Bv))
    if rng.random() < 0.5:
        K = rng.normal(size=(g, g)); K *= 0.2 * budget / np.linalg.norm(K, 2)
        qk.append(dict(r=1.0, k=lambda s: np.exp(-((s - 0.5) / 0.2) ** 2), K=K))
    return build_fde(A=A, g=g, phi=lambda t: np.ones(g), delays=delays, neutral=neutral, var_delays=var,
                     quad_kernels=qk, t_range=(0, 6))


@pytest.mark.parametrize("seed", list(range(12)))
def test_main_theorem_random_general_systems(seed):
    fde = random_system(seed)
    gam = discretize_general(fde, 2, 0.5).gammas
    cert = certify(fde, gam, nt=9)
    if not cert["ok"]:
        pytest.skip("no LK functional for this random system (allowed: the LMI is only sufficient)")
    for p, h in [(3, 0.4), (6, 0.2)]:
        disc = discretize_general(fde, p, h, hs=0.25, beta=cert["beta"])
        W = weight_general(disc, cert["P"], cert["Qc"], cert["Qv"], cert["S0"])
        G_of = make_generator(disc)
        scale = max(1.0, np.linalg.norm(W, 2))
        for t in np.linspace(0, 6, 7):
            assert sym_max(W @ G_of(t)[0]) <= 1e-9 * scale


# ---------------------------------------------------------------- accuracy
def test_spectral_on_exact_mode():
    from scipy.optimize import fsolve
    def F(v):
        lam = v[0] + 1j * v[1]
        r = lam + 2 + 1.2 * np.exp(-lam) + 0.7 * np.exp(-1.7 * lam)
        return [r.real, r.imag]
    v = fsolve(F, [-0.35, 1.8], xtol=1e-15); lam = v[0] + 1j * v[1]
    dde = core.LinearDDE(A=[[-2.0]], delays=[1.0, 1.7], Bs=[[[-1.2]], [[-0.7]]])
    disc = core.discretize(dde, 8, 0.35)
    G = core.generator(disc)
    z = expm(G * 6.0) @ core.initial_state(disc, lambda t: np.array([np.real(np.exp(lam * t))]))
    assert abs(z[0] - np.real(np.exp(lam * 6.0))) < 1e-11


# ---------------------------------------------------------------- LCHS
def test_lchs_identity_reproduces_dissipative_evolution():
    n = 8
    M = RNG.normal(size=(n, n))
    G = (M - M.T) - 0.5 * np.eye(n) - 0.3 * (lambda X: X @ X.T)(RNG.normal(size=(n, n))) / n
    assert sym_max(G) < 0
    v0 = RNG.normal(size=n).astype(complex)
    approx, st = lchs.emulate(G, 1.0, v0, eps=1e-8, beta=0.8)
    assert np.linalg.norm(approx - expm(G) @ v0) / np.linalg.norm(v0) < 1e-6
    ks = np.linspace(-500, 500, 1_000_001)
    assert abs(np.trapezoid(lchs.weight(ks), ks) - 1) < 1e-6


# ---------------------------------------------------------------- validated LK functional
def test_validated_lk_time_varying():
    import general_tests as gt
    fde, _ = gt.problem("G3")
    gam = discretize_general(fde, 2, 0.5).gammas
    cert = certify(fde, gam)
    v = verify.validate(fde, cert, gam, n_grid=101)
    assert v["valid"]


# ---------------------------------------------------------------- spec safety and compiler
def test_spec_rejects_code_injection():
    from qfde.spec import compile_expr, SpecError
    for bad in ["__import__('os').system('echo x')", "t.__class__", "open('f')", "[x for x in (1,)]",
                "lambda: 1", "s", "eval('1')"]:
        with pytest.raises(SpecError):
            compile_expr(bad, ("t",))
    f = compile_expr("1.0 if t >= 0 else 0.5*exp(-t**2) + sin(pi*t)", ("t",))
    assert f(1.0) == 1.0 and abs(f(-1.0) - (0.5 * np.exp(-1) + np.sin(-np.pi))) < 1e-12


def test_compiler_smooth_two_delays():
    from qfde import spec as specmod
    from qfde.compiler import compile_spec
    s = specmod.load(os.path.join(ROOT, "examples", "two_delays_smooth.toml"))
    s["eps"] = 1e-4
    r = compile_spec(s)
    assert r["lk_functional"]["found"] and r["history"]["compatible"]
    assert r["discretisation"]["error_vs_reference"] < 1e-4
    assert r["cost"]["queries"] > 0 and r["cost"]["logical"]["numQubits"] > 0


def test_compiler_reports_missing_lk_functional():
    from qfde import spec as specmod
    from qfde.compiler import compile_spec
    s = specmod.load(os.path.join(ROOT, "examples", "unstable.toml"))
    r = compile_spec(s)
    assert not r["lk_functional"]["found"] and "verdict" in r


def test_compiler_detects_mismatched_history():
    from qfde import spec as specmod
    from qfde.compiler import history_jump
    s = specmod.load(os.path.join(ROOT, "examples", "demo_dde.toml"))
    assert history_jump(s) < 1e-6                                   # characteristic-mode history matches
    s["phi"] = lambda t: np.array([1.0])                            # constant past: x'(0+) = -0.8, phi'(0-) = 0
    assert abs(history_jump(s) - 0.8 / 2) < 1e-6
    s = specmod.load(os.path.join(ROOT, "examples", "demo_neutral.toml"))
    assert history_jump(s) < 1e-6


# ---------------------------------------------------------------- circuits (optional toolchain)
def test_sparse_block_encoding_circuit_exact():
    pytest.importorskip("qiskit")
    from qfde import circuits
    dde = core.LinearDDE(A=[[-2.0]], delays=[1.0, 1.7], Bs=[[[-1.2]], [[-0.7]]])
    ok, P, Qs, _ = core.scalar_certificate(dde)
    disc = core.discretize(dde, 1, 1.0)
    Gh, _ = core.weighted_generator(core.generator(disc), core.lk_weight(disc, P, Qs))
    qc, info = circuits.sparse_block_encoding(Gh, return_info=True)
    B = circuits.extract_block(qc, info["n"], Gh.shape[0])
    assert np.abs(B * info["alpha"] - Gh).max() < 1e-12


def test_dilation_circuit_reproduces_dissipative_evolution():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector
    from qfde import circuits
    dde = core.LinearDDE(A=[[-2.0]], delays=[1.0], Bs=[[[1.2]]])
    ok, P, Qs, _ = core.scalar_certificate(dde)
    disc = core.discretize(dde, 2, 1.0)
    Gh, Wh = core.weighted_generator(core.generator(disc), core.lk_weight(disc, P, Qs))
    V = expm(Gh * 1.0)
    assert np.linalg.norm(V, 2) <= 1 + 1e-12                      # main theorem => contraction
    v0 = Wh @ core.initial_state(disc, lambda t: np.array([1.0]))
    qc, info = circuits.dilation_circuit(V, v0)
    assert info["unitarity_error"] < 1e-12
    p = Statevector(qc).probabilities()[:4]
    assert np.abs(p - circuits.ideal_distribution(V, v0)).max() < 1e-12


# ---------------------------------------------------------------- resources
def test_ftqc_pipeline():
    pytest.importorskip("qualtran"); pytest.importorskip("qdk")
    from qfde import ftqc
    pq = ftqc.per_query_cost(node_bits=5, p=4)
    assert pq["toffoli"] > 0 and pq["rotations"] > 0
    ph = ftqc.physical(ftqc.logical_counts(1e5, pq, 5))
    assert ph["physical_qubits"] > 0 and ph["runtime_s"] > 0


# ---------------------------------------------------------------- kernel condition Gamma >= Gamma_h
def test_kernel_condition_enforced_on_coarse_mesh():
    from qfde.fde import check_gamma
    k = lambda s: 1.5 * np.exp(-((s - 0.5) / 0.3) ** 2)
    fde = build_fde(A=[[-2.0]], g=1, phi=lambda t: np.array([1.0]),
                    quad_kernels=[dict(r=1.0, k=k, K=[[1.0]])], t_range=(0, 2))
    cert = certify(fde, [], nt=1, betas=(0.25, 0.5, 1.0), meshes=[(2, 1.0)])
    assert cert["ok"]
    disc = discretize_general(fde, 2, 1.0, beta=cert["beta"])
    assert all(ok for _, _, ok in check_gamma(disc, cert))
    W = weight_general(disc, cert["P"], cert["Qc"], cert["Qv"], cert["S0"])
    G = make_generator(disc)(0.0)[0]
    lam, V = np.linalg.eigh(W)
    Gh = (V * np.sqrt(lam)) @ V.T @ G @ (V / np.sqrt(lam)) @ V.T
    assert sym_max(Gh) <= 1e-10


# ---------------------------------------------------------------- LCHS circuit (gate level)
def test_lchs_circuit_matches_lcu_and_converges():
    import tomllib
    from qiskit.quantum_info import Statevector
    from qfde import spec as S, compiler, lchs_circuit as LC
    sp = S.parse(tomllib.load(open(os.path.join(ROOT, "examples", "demo_dde.toml"), "rb")))
    rep, lo = compiler.compile_spec(sp, lowered=True)
    assert rep["discretisation"]["unknowns"] == 4
    wd = np.diag(lo["W"]); Gh = (np.sqrt(wd)[:, None] * lo["G"]) / np.sqrt(wd)[None, :]
    v0 = np.sqrt(wd) * lo["z0"]; t = 1.0
    for M, tol in [(4, None), (64, 5e-2), (256, 1e-3)]:
        K, _ = LC.choose_K(Gh, [t], M, 0.8, Ks=np.geomspace(0.25, 80, 40), out=lo["R"] / np.sqrt(wd)[None, :])
        k, c = LC.nodes(M, K, 0.8)
        qc, meta = LC.circuit(Gh, v0, t, k, c, measure=False)
        amp = Statevector.from_instruction(qc).data[: 1 << meta["ns"]]
        p = np.abs(amp) ** 2
        assert np.allclose(p, LC.ideal_probs(Gh, v0, t, k, c), atol=1e-10)
        if tol:
            v = amp[:4] * meta["s"] * meta["norm_v0"]
            x = float(np.real(lo["R"] @ (v / np.sqrt(wd)))[0])
            assert abs(x - np.exp(-0.33031401751431055 * t)) < tol


# ---------------------------------------------------------------- fading memory by window truncation
def test_fading_memory_truncation_converges():
    import instances as INS
    S = INS.spec("erlang2")
    errs = []
    for r in (2.0, 4.0, 8.0):
        fde, disc, G, Gh, Wh, Wih, cert, W = INS.lowered(S, 8, 0.5, r)
        assert sym_max(Gh) <= 1e-10
        x, _ = solve_general(disc, 4.0, np.array([4.0]), autonomous=True)
        errs.append(abs(float(np.asarray(x).ravel()[0]) - INS.exact(S, 4.0)[0]))
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-5


# ---------------------------------------------------------------- hardware energy circuits
def test_energy_dilation_is_unitary_and_measures_energy():
    import hardware as HW
    I = HW.compile_file("demo_dde.toml")
    for st in HW.v1_setups(I)[:6]:
        U = HW.dilation(st["C"])
        assert np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-10)
        out = U @ np.concatenate([st["v"], np.zeros_like(st["v"])])
        p0 = float(np.linalg.norm(out[: len(st["v"])]) ** 2)
        assert abs(p0 - st["p0_ideal"]) < 1e-10
        if st["series"].startswith("w"):
            assert st["energy_ideal"] <= 1 + 1e-12
