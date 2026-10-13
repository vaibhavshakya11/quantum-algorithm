"""
Stability and accuracy against the degree p for the three demo equations (exact characteristic-mode
histories): lambda_max(sym G) without the LK weight, lambda_max(sym G_hat) with it, and the relative error of
x(t) on [0, 4] against the exact solution e^{lambda t}.
Run: python3 benchmarks/p_sweep.py -> results/p_sweep.json
"""
import sys, os, json, math, tomllib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.linalg import expm
from qfde import spec as S
from qfde.fde import build_fde, discretize_general, make_generator, weight_general, certify, check_gamma, \
    initial_state_general, x_in_row

SPECS = [("delay", "demo_dde.toml", -0.33031401751431055),
         ("neutral", "demo_neutral.toml", -0.39483263668935853),
         ("integro-differential", "demo_ide.toml", -0.5563891948650157)]
PS = list(range(1, 13))
H = 0.5
T = 4.0


def main():
    out = {}
    for name, f, lam in SPECS:
        sp = S.parse(tomllib.load(open(os.path.join(ROOT, "examples", f), "rb")))
        fde = build_fde(A=sp["A"], g=1, phi=sp["phi"], delays=sp["delays"], neutral=sp["neutral"],
                        quad_kernels=sp["quad_kernels"], t_range=(0, T))
        cert = certify(fde, [], nt=1, meshes=[(p, H) for p in PS] if fde.qk else None)
        rows = []
        for p in PS:
            disc = discretize_general(fde, p, H, beta=cert["beta"])
            G = make_generator(disc)(0.0)[0]
            W = weight_general(disc, cert["P"], cert["Qc"], cert["Qv"], cert["S0"])
            lw, V = np.linalg.eigh(W)
            Gh = (V * np.sqrt(lw)) @ V.T @ G @ (V / np.sqrt(lw)) @ V.T
            z0 = initial_state_general(disc); R = x_in_row(disc)
            ts = np.linspace(0.25, T, 16)
            err = max(abs((R @ expm(G * t) @ z0)[0] - math.exp(lam * t)) / math.exp(lam * t) for t in ts)
            rows.append(dict(p=p, unknowns=disc.size, lam_unweighted=float(np.linalg.eigvalsh((G + G.T) / 2).max()),
                             lam_weighted=float(np.linalg.eigvalsh((Gh + Gh.T) / 2).max()), err=float(err),
                             gamma_ok=all(ok for *_, ok in check_gamma(disc, cert)) if fde.qk else None))
            print(name, rows[-1], flush=True)
        out[name] = dict(beta=cert["beta"], h=H, rows=rows)
    json.dump(out, open(os.path.join(ROOT, "results", "p_sweep.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
