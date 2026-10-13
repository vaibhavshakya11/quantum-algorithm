"""Command-line interface.

    qfde compile examples/drug_reabsorption.toml [--json out.json] [--md out.md]
    qfde check  examples/drug_reabsorption.toml           # parse + LK functional only (fast)
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qfde", description="Quantum compilation of linear FDEs via LK-preserving discretization")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile", help="full compile: LK functional, history, discretization, costs")
    c.add_argument("spec"); c.add_argument("--json"); c.add_argument("--md")
    k = sub.add_parser("check", help="parse the spec and search for an LK functional")
    k.add_argument("spec")
    args = ap.parse_args(argv)

    import warnings
    warnings.filterwarnings("ignore")
    from . import spec as specmod
    try:
        s = specmod.load(args.spec)
    except specmod.SpecError as e:
        print(f"spec error: {e}", file=sys.stderr)
        return 2
    if args.cmd == "check":
        from .fde import build_fde, discretize_general, certify
        fde = build_fde(A=s["A"], g=s["g"], phi=s["phi"], delays=s["delays"], neutral=s["neutral"],
                        var_delays=s["var_delays"], quad_kernels=s["quad_kernels"], f=s["f"], t_range=(0, s["T"]))
        cert = certify(fde, discretize_general(fde, 2, 0.5).gammas, nt=(25 if s["time_dependent"] else 1))
        print(f"{s['name']}: LK functional {'FOUND' if cert['ok'] else 'NOT FOUND'}"
              + (f" (beta = {cert['beta']}, cond(P) = {cert['cond_P']:.3g})" if cert["ok"] else ""))
        return 0 if cert["ok"] else 1
    from .compiler import compile_spec
    rep = compile_spec(s)
    print(rep["markdown"])
    if args.json:
        json.dump({k: v for k, v in rep.items() if k != "markdown"}, open(args.json, "w"), indent=1, default=float)
    if args.md:
        open(args.md, "w").write(rep["markdown"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
