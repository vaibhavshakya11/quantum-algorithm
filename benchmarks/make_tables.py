"""Tables of the preprint from results/comparisons.json, and results/instances.json.
python3 benchmarks/make_tables.py -> paper/tables/*.tex"""
import os, json, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
C = json.load(open(os.path.join(RES, "comparisons.json")))
D = json.load(open(os.path.join(RES, "instances.json")))
OUT = os.path.join(ROOT, "paper", "tables"); os.makedirs(OUT, exist_ok=True)
ORDER = ["two-delay", "drug", "baroreflex", "gaussian", "erlang2"]
SHORT = {"two-delay": "Two delays", "drug": "Drug reabsorption", "baroreflex": "Baroreflex",
         "gaussian": "Gaussian kernel", "erlang2": "Erlang-2 memory"}
FAM = {"two-delay": "D", "drug": "D", "baroreflex": "D", "gaussian": "I", "erlang2": "I"}


def sci(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    if x < 1e4:
        return (f"{x:,.0f}".replace(",", "{,}") if x >= 100 else (f"{x:.2g}" if x < 10 else f"{x:.0f}"))
    e = int(math.floor(math.log10(x))); m = x / 10 ** e
    if round(m, d - 1) >= 10:
        m /= 10; e += 1
    return f"${m:.{d - 1}f}\\times10^{{{e}}}$"


def epsx(e):
    return f"$10^{{{int(round(math.log10(e)))}}}$"


def write(name, body):
    open(os.path.join(OUT, name), "w").write(body)


def row(name, eps):
    return next(r for r in C["instances"][name]["sweep"] if abs(r["eps"] - eps) < eps * 1e-6)


# ---------------------------------------------------------------- queries ratio table (the comparison)
E = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]
L = ["\\begin{tabular}{ll" + "r" * len(E) + "}\\toprule",
     "Instance & Prior method & " + " & ".join(epsx(e) for e in E) + "\\\\\\midrule"]
for n in ORDER:
    I = C["instances"][n]
    meth = "Schr\\\"odingerization" if n.startswith("erlang") else ("first order + QLSA")
    cells = []
    for e in E:
        r = row(n, e)
        if r["ratio"] is None:
            cells.append("---"); continue
        s = sci(r["ratio"]) + "$\\times$"
        if not r["prior"].get("err_measured", True):
            s += "$^\\ast$"
        cells.append(s)
    L.append(f"{SHORT[n]} & {meth} & " + " & ".join(cells) + "\\\\")
    if n == "baroreflex":
        L.append("\\midrule")
L.append("\\bottomrule\\end{tabular}")
write("hp_ratio.tex", "\n".join(L))

# ---------------------------------------------------------------- weight comparison (none / Lyapunov / ours)
L = ["\\begin{tabular}{lrrrrrrrr}\\toprule",
     "& \\multicolumn{1}{c}{No weight} & \\multicolumn{3}{c}{Lyapunov-equation weight} & \\multicolumn{3}{c}{LK weight (ours)} & \\multicolumn{1}{c}{Lyapunov $\\div$ LK}\\\\",
     "\\cmidrule(lr){2-2}\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-9}",
     "Instance & Shift factor & cond & Nonzeros/row & Queries & cond & Nonzeros/row & Queries & Queries\\\\\\midrule"]
for n in ORDER:
    I = C["instances"][n]; Ly = I["lyapunov"]; a = I["ablation"]
    L.append(f"{SHORT[n]} & $10^{{{a['log10_shift_penalty']:.0f}}}$ & {sci(Ly['cond'])} & {Ly['sparsity']} & {sci(Ly['queries'])} & "
             f"{sci(Ly['ours']['cond'])} & {Ly['ours']['sparsity']} & {sci(Ly['ours']['queries'])} & {Ly['queries']/Ly['ours']['queries']:,.0f}$\\times$\\\\")
L.append("\\bottomrule\\end{tabular}")
write("weights.tex", "\n".join(L))

# ---------------------------------------------------------------- error-corrected estimates
def rt(s):
    s = s.replace("years", "yr").replace("days", "d").replace("hours", "h").replace("mins", "min")
    return s.replace(" (lower bound)", "$^\\dagger$")


L = ["\\begin{tabular}{llrrrrr}\\toprule",
     "& & \\multicolumn{2}{c}{Prior method} & \\multicolumn{2}{c}{This work} & \\\\",
     "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
     "Instance & $\\varepsilon$ & Runtime & Phys.\\ qubits & Runtime & Phys.\\ qubits & Speed-up\\\\\\midrule"]
for n in ORDER:
    first = True
    for e, f in C["instances"][n].get("ft", {}).items():
        L.append(f"{SHORT[n] if first else ''} & {epsx(float(e))} & {rt(f['prior']['runtime_formatted'])} & "
                 f"{f['prior']['physical_qubits'] / 1e6:.1f}M & {rt(f['ours']['runtime_formatted'])} & "
                 f"{f['ours']['physical_qubits'] / 1e6:.1f}M & {sci(f['ratio'])}$\\times$\\\\")
        first = False
L.append("\\bottomrule\\end{tabular}")
write("ft2.tex", "\n".join(L))

print("tables written to", OUT)
