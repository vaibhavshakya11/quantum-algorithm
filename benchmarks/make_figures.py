"""Regenerate every figure of the preprint from results/*.json.   python3 benchmarks/make_figures.py"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)
BLUE, GREY, INK, INK2, GRID = "#2a78d6", "#9a9890", "#0b0b0b", "#52514e", "#e4e3df"
SHADES = ["#c9c7c0", "#a9a79f", "#86847c"]


def style(ax, title, sub=None):
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=18 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.8); ax.tick_params(colors=INK2)
    for s in ax.spines.values():
        s.set_color(GRID)


def fig_p_sweep():
    P = json.load(open(os.path.join(RES, "p_sweep.json")))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    cols = {"delay": BLUE, "neutral": INK2, "integro-differential": GREY}
    for name, d in P.items():
        p = [r["p"] for r in d["rows"]]
        axs[0].semilogy(p, [max(r["lam_unweighted"], 1e-16) for r in d["rows"]], "-o", color=cols[name], ms=4,
                        label=f"{name}, no weight")
        axs[0].semilogy(p, [max(r["lam_weighted"], 1e-16) for r in d["rows"]], "--s", color=cols[name], ms=3,
                        label=f"{name}, LK weight (<=0 shown at 1e-16)")
        axs[1].semilogy(p, [r["err"] for r in d["rows"]], "-o", color=cols[name], ms=4, label=name)
    axs[0].set_xlabel("degree p", color=INK2); axs[0].set_ylabel("largest eigenvalue of sym part", color=INK2)
    style(axs[0], "Dissipativity: lost without the weight, exact with it")
    axs[0].legend(frameon=False, fontsize=6.5, labelcolor=INK2)
    axs[1].set_xlabel("degree p (element size 0.5)", color=INK2); axs[1].set_ylabel("max relative error of x(t)", color=INK2)
    style(axs[1], "Spectral convergence to rounding by p = 6")
    axs[1].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "p_sweep.png"), dpi=170); plt.close(fig)


def fig_hw_energy():
    R = json.load(open(os.path.join(RES, "hardware.json")))
    rows = R["hardware"]
    order = ["delay, one delay", "neutral", "time-varying delay", "integro-differential, bounded kernel",
             "integro-differential, fading memory", "delay, two delays", "drug reabsorption", "baroreflex"]
    fig, axs = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
    sty = {"w-mode": (BLUE, "o", "weighted, solution mode"), "w-worst": ("#5b97de", "s", "weighted, slowest direction"),
           "u-worst": (GREY, "^", "unweighted, fastest-growing direction")}
    for ax, c in zip(axs.flat, order):
        for s, (col, mk, lab) in sty.items():
            dn = R["ideal_dense"][c]
            hv = sorted([r for r in rows if r["exp"] == "V1" and r["cls"] == c and r["series"] == s], key=lambda r: r["t"])
            ax.plot(dn["t"], dn[s], "-", color=col, lw=1.6, label=f"{lab} (exact)")
            ax.plot([r["t"] for r in hv], [r["energy_zne"] for r in hv], mk, color=col, ms=6, mfc="white" if s == "w-worst" else col,
                    label=f"{lab} (ibm_marrakesh)")
        ax.axhline(1, color=INK2, lw=0.8, ls=":")
        cz = next(r["cz"] for r in rows if r["exp"] == "V1" and r["cls"] == c)
        style(ax, c.replace("integro-differential", "integro-diff."), f"{cz} two-qubit gates")
        ax.title.set_fontsize(9.5)
        ax.set_xlabel("t", color=INK2)
    for ax in axs[:, 0]:
        ax.set_ylabel("energy ratio E(t)/E(0)", color=INK2)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(FIG, "hw_energy.png"), dpi=170); plt.close(fig)


def fig_hw_lchs():
    R = json.load(open(os.path.join(RES, "hardware.json")))
    rows = [r for r in R["hardware"] if r["exp"] == "V2"]
    lams = {"delay, one delay": -0.33031401751431055, "neutral": -0.39483263668935853,
            "integro-differential, bounded kernel": -0.5563891948650157,
            "integro-differential, fading memory": -0.13985294912645574}
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, (c, lam) in zip(axs, lams.items()):
        rr = sorted([r for r in rows if r["cls"] == c], key=lambda r: r["t"])
        tt = np.linspace(0, 2.1, 100)
        ax.plot(tt, np.exp(lam * tt), "-", color=GREY, lw=2, label="exact solution")
        ax.plot([r["t"] for r in rr], [r["x_ideal"] for r in rr], "o", mfc="none", color=INK2, ms=9, label="ideal 4-term LCHS circuit")
        ax.plot([r["t"] for r in rr], [r["x_corr"] for r in rr], "s", color="#8db7e8", ms=5, label="ibm_marrakesh, readout corrected")
        ax.plot([r["t"] for r in rr], [r["x_zne"] for r in rr], "s", color=BLUE, ms=6, label="ibm_marrakesh, + zero-noise extrapolation")
        style(ax, c.replace("integro-differential", "integro-diff."), "38 two-qubit gates")
        ax.title.set_fontsize(10); ax.set_xlabel("t", color=INK2)
    axs[0].set_ylabel("x(t)", color=INK2)
    axs[0].legend(frameon=False, fontsize=7, labelcolor=INK2, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "hw_lchs.png"), dpi=170); plt.close(fig)


def fig_hp():
    C = json.load(open(os.path.join(RES, "comparisons.json")))
    names = [("two-delay", "Delay: two delays"), ("baroreflex", "Delay: baroreflex"),
             ("gaussian", "Integro-differential: Gaussian kernel"), ("erlang2", "Integro-differential: Erlang-2 memory")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.7), sharey=True)
    for ax, (n, lab) in zip(axs, names):
        sw = C["instances"][n]["sweep"]
        pm = [(r["eps"], r["prior"]["queries"]) for r in sw if r["prior"].get("err_measured", True)]
        px = [(r["eps"], r["prior"]["queries"]) for r in sw if not r["prior"].get("err_measured", True)]
        if px:
            px = [pm[-1]] + px
        meth = "Schrodingerization bound" if n.startswith("erlang") else "first order + QLSA"
        ax.loglog([a for a, _ in pm], [b for _, b in pm], "-s", color=GREY, ms=4, lw=2, label=meth)
        if px:
            ax.loglog([a for a, _ in px], [b for _, b in px], "--", color=GREY, lw=1.5, label="extrapolated (proven rate)")
        oo = [(r["eps"], r["ours"]["queries"]) for r in sw if r["ours"]]
        ax.loglog([a for a, _ in oo], [b for _, b in oo], "-o", color=BLUE, ms=4, lw=2.4, label="this work")
        cross = [r["eps"] for r in sw if r["ratio"] and r["ratio"] >= 1]
        if cross:
            ax.axvline(max(cross), color=BLUE, lw=0.8, ls=":")
        ax.invert_xaxis(); ax.set_xlabel("target accuracy ε", color=INK2)
        style(ax, lab); ax.title.set_fontsize(9.5)
    axs[0].set_ylabel("block-encoding queries", color=INK2)
    axs[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "hp_queries.png"), dpi=170); plt.close(fig)


if __name__ == "__main__":
    fig_p_sweep()
    if os.path.exists(os.path.join(RES, "comparisons.json")):
        fig_hp()
    if os.path.exists(os.path.join(RES, "hardware.json")):
        fig_hw_energy(); fig_hw_lchs()
    print("figures written to", FIG)
