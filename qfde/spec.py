"""
Problem specification files (TOML) and a SAFE expression evaluator.

Expressions (history, time-varying entries, kernels, delays, forcing) are parsed with Python's `ast`
and only a whitelist is accepted: numbers, the variables t / s, + - * / ** and unary minus,
comparisons, `a if cond else b`, and the functions below.  Anything else (attribute access, imports,
subscripts, calls to other names, lambdas, comprehensions ...) is rejected before evaluation.

Example spec
------------
    [problem]
    name    = "drug reabsorption"
    dim     = 2
    T       = 24.0
    A       = [[-0.8, 0.0], [0.5, -0.6]]          # numbers or expression strings in t
    history = ["1.0 if t >= 0 else 0.0", "0.0"]   # x(t) for t <= 0 (expressions in t)

    [[delay]]
    tau = 2.0
    B   = [[0.0, 0.4], [0.0, 0.0]]

    [[neutral]]            # d/dt[x - C x(t - sigma)] = ...
    sigma = 0.5
    C     = [[0.3, 0.0], [0.0, 0.0]]

    [[kernel]]             # + coef(t) K int_0^window k(s) x(t - s) ds
    window = 2.0
    k      = "0.5*exp(-((s - 1)/0.35)**2)"
    K      = [[1.0, 0.0], [0.0, 1.0]]

    [[var_delay]]          # + B x(t - d(t)),   d'(t) < 1
    d  = "1 + 0.3*sin(t)"
    dd = "0.3*cos(t)"
    B  = [[0.6, 0.0], [0.0, 0.0]]

    forcing = ["0.5*sin(2*t)", "0"]     # optional, inside [problem]

    [target]
    eps = 1e-6
    scale_dimension = 1e12              # optional: report resources as if A were this large (sparse)
"""
from __future__ import annotations

import ast
import math
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:            # Python < 3.11
    import tomli as tomllib

_FUNCS = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
          "abs": np.abs, "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh, "arctan": np.arctan,
          "min": min, "max": max, "heaviside": lambda x: 1.0 if x >= 0 else 0.0}
_CONST = {"pi": math.pi, "e": math.e}
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call, ast.IfExp,
            ast.Compare, ast.BoolOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod,
            ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.And, ast.Or)


class SpecError(ValueError):
    pass


def compile_expr(src, variables=("t",)):
    """Return a callable f(*variables) for a whitelisted arithmetic expression (or a constant)."""
    if isinstance(src, (int, float)):
        val = float(src)
        return lambda *a: val
    if not isinstance(src, str):
        raise SpecError(f"expected number or expression string, got {src!r}")
    tree = ast.parse(src, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise SpecError(f"disallowed syntax {type(node).__name__} in {src!r}")
        if isinstance(node, ast.Name) and node.id not in variables and node.id not in _FUNCS and node.id not in _CONST:
            raise SpecError(f"unknown name {node.id!r} in {src!r} (allowed: {', '.join(variables)})")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
                raise SpecError(f"disallowed call in {src!r}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise SpecError(f"only numeric constants allowed in {src!r}")
    code = compile(tree, "<spec>", "eval")
    env = {"__builtins__": {}, **_FUNCS, **_CONST}

    def f(*args):
        v = eval(code, env, dict(zip(variables, args)))
        if np.iscomplexobj(v):              # complex arguments (Cauchy-integral derivatives) stay complex
            return complex(v)
        return float(v)
    return f


def matrix_fn(M, g, name):
    """Matrix of numbers/expressions in t -> (callable t -> ndarray, is_constant)."""
    M = M if isinstance(M[0], list) else [M]
    if len(M) != g or any(len(r) != g for r in M):
        raise SpecError(f"{name} must be {g}x{g}")
    const = all(isinstance(v, (int, float)) for r in M for v in r)
    if const:
        arr = np.array(M, float)
        return arr, True
    fs = [[compile_expr(v) for v in r] for r in M]
    return (lambda t: np.array([[f(t) for f in r] for r in fs])), False


def vector_fn(v, g, name):
    if len(v) != g:
        raise SpecError(f"{name} must have length {g}")
    fs = [compile_expr(x) for x in v]
    return lambda t: np.array([f(t) for f in fs])


def load(path_or_text, is_text=False):
    data = tomllib.loads(path_or_text) if is_text else tomllib.load(open(path_or_text, "rb"))
    return parse(data)


def parse(data):
    if "problem" not in data:
        raise SpecError("missing [problem] table")
    pr = data["problem"]
    g = int(pr["dim"]); T = float(pr["T"])
    A, A_const = matrix_fn(pr["A"], g, "A")
    phi = vector_fn(pr["history"], g, "history")
    spec = dict(name=pr.get("name", "problem"), g=g, T=T, A=A, phi=phi, time_dependent=not A_const,
                delays=[], neutral=[], quad_kernels=[], var_delays=[], f=None)
    for d in data.get("delay", []):
        B, c = matrix_fn(d["B"], g, "delay.B"); spec["delays"].append((float(d["tau"]), B))
        spec["time_dependent"] |= not c
    for d in data.get("neutral", []):
        C, c = matrix_fn(d["C"], g, "neutral.C")
        if not c:
            raise SpecError("neutral coefficient must be constant")
        spec["neutral"].append((float(d["sigma"]), C))
    for d in data.get("kernel", []):
        K, c = matrix_fn(d["K"], g, "kernel.K")
        if not c:
            raise SpecError("kernel matrix K must be constant (use coef for time modulation)")
        kd = dict(r=float(d["window"]), k=compile_expr(d["k"], ("s",)), K=K)
        if "coef" in d:
            kd["coef"] = compile_expr(d["coef"]); spec["time_dependent"] = True
        spec["quad_kernels"].append(kd)
    for d in data.get("var_delay", []):
        B, c = matrix_fn(d["B"], g, "var_delay.B")
        spec["var_delays"].append((compile_expr(d["d"]), compile_expr(d["dd"]), B))
        spec["time_dependent"] = True
    if "forcing" in pr:
        spec["f"] = vector_fn(pr["forcing"], g, "forcing"); spec["time_dependent"] = True
    ds = data.get("discretization", {})
    spec["degree"] = int(ds["degree"]) if "degree" in ds else None
    spec["element_size"] = float(ds["element_size"]) if "element_size" in ds else None
    spec["stretched_element_size"] = float(ds["stretched_element_size"]) if "stretched_element_size" in ds else None
    tg = data.get("target", {})
    spec["eps"] = float(tg.get("eps", 1e-6))
    spec["scale_dimension"] = float(tg["scale_dimension"]) if "scale_dimension" in tg else None
    return spec
