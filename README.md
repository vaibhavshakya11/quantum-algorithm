# qfde v0.5.1: a new quantum algorithm for integro-differential and delay differential equations

`qfde` implements the algorithm of the preprint in `paper/`: a quantum algorithm for linear delay differential equations (several, neutral and time-varying delays) and integro-differential equations with convolution memory (bounded or exponentially fading kernels), with time-varying coefficients and forcing. Given an equation it

1. finds a Lyapunov–Krasovskii functional (by a small matrix inequality or in closed form); by the main theorem this makes the discretized generator dissipative. If none exists in the searched family it says so;
2. discretizes the stored history with summation-by-parts spectral elements (spectral convergence), truncating exponentially fading memory to a window of length O(log 1/ε);
3. reports the cost of the quantum algorithm (improved LCHS on the weighted generator): block-encoding queries, preparations of the initial data, logical counts, physical qubits and runtime on an error-corrected machine;
4. for small instances, builds the gate-level LCHS circuit and the energy-test circuits that were run on IBM hardware.

Quantum costs come from proven query bounds with explicit constants, checked against exact classical emulation. They assume an error-corrected (fault-tolerant) quantum computer. The runs on `ibm_marrakesh` (a noisy processor without error correction) validate the energy law of the main theorem and the execution of the algorithm's circuits; they do not demonstrate high precision or a speedup.

## Quick start

```
pip install -e ".[quantum,dev]"
qfde compile examples/two_delays_smooth.toml        # full report
qfde check   examples/unstable.toml                 # LK inequality only
python3 -m pytest -q                                 # 33 tests, about 1 min
```

## What is in the box

| Path | Role |
| --- | --- |
| `qfde/spec.py`, `qfde/cli.py`, `qfde/compiler.py` | Spec files (safe parser), CLI, compiler with recorded decisions |
| `qfde/fde.py` | Normal form; the LK-weighted history discretization; LMI and closed-form LK functionals |
| `qfde/lchs.py`, `qfde/resources.py` | Improved LCHS (emulation and query counts); cost models of prior methods (QLSA for first-order schemes, Schrödingerization) |
| `qfde/circuits.py` | Sparse-access block encoding (block = Ĝ/α to 1e-14); unitary dilation of the dissipative evolution |
| `qfde/lchs_circuit.py` | Gate-level LCHS circuit (PREP, SELECT of e^{-it(kL+H)}, PREP†, postselection) for small LK-weighted generators |
| `qfde/ftqc.py` | Qualtran primitives → logical counts → Microsoft resource estimator |
| `qfde/verify.py` | Validated LK inequality for all t (grid + Lipschitz + round-off) |
| `benchmarks/instances.py` | The five comparison instances (two delays, drug reabsorption, baroreflex, Gaussian kernel, Erlang-2 memory) with exact characteristic-mode histories: our discretization over a (window,) degree, mesh grid, and the prior first-order methods measured by time stepping (numba) to below 1e-6 → `results/instances.json` |
| `benchmarks/comparisons.py` | High-precision comparison (queries and initial-data preparations, ε = 1e-1 … 1e-10), the Lyapunov-equation weight, the ablation and error-corrected estimates → `results/comparisons.json` |
| `benchmarks/make_tables.py`, `benchmarks/make_figures.py` | Regenerate the paper's tables and figures |
| `benchmarks/hardware.py` | IBM hardware validation: `ideal`, `noise` (FakeMarrakesh), `submit`, `fetch` (needs `IBM_TOKEN`, `IBM_CRN`); energy test on 8 classes and LCHS circuits on 4 → `results/hardware.json` |
| `benchmarks/lchs_demo.py` | LCHS circuit simulation: convergence in the number of terms, with and without the LK weight |
| `benchmarks/p_sweep.py` | Stability (with/without the weight) and accuracy against the degree p |
| `benchmarks/experiments.py`, `benchmarks/general_tests.py` | The dissipativity table (random and structured systems), accuracy on manufactured solutions, the delay-stabilized example of the Scope section |
| `results/predictions.md` | Predictions for the circuit experiments, recorded before they were run, and the outcomes |
