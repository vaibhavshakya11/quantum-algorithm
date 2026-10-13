# Design decisions (and rejected alternatives)

1. **History as state rather than an all-at-once time system.** Storing the history window turns any covered FDE into a memoryless system, which lets modern quantum ODE solvers (LCHS) apply. *Rejected:* first-order all-at-once systems solved by a quantum linear-system solver. Their condition number grows like the number of steps, so the cost grows like 1/ε.

2. **Summation-by-parts spectral elements.** They are the only discretization we found that inherits the continuous Lyapunov–Krasovskii functional *exactly*, for every degree and mesh (the main theorem). *Rejected:* plain collocation in the history variable. It is accurate, but it is not provably dissipative, so LCHS might not apply.

3. **LK-weighted norm.** Delay generators are typically not dissipative in the Euclidean norm (largest Hermitian eigenvalue up to 7×10⁴ in our tests). The LK weight W makes them dissipative. *Rejected:* shifting the generator by its largest eigenvalue, which multiplies the cost by exp(shift·T).

4. **Histories that match the equation.** A history that does not match the equation at t = 0 creates derivative jumps at the breaking points and makes convergence algebraic. The main theorem still holds, but the polylogarithmic guarantee needs analytic solutions, so the paper assumes matching histories (the forecast of a running system) and the compiler reports a mismatch. Corrections for mismatched histories are outside the paper and are not in the software.

5. **Kernels by quadrature on the history window.** *Rejected:* complex sum-of-exponential fits of non-monotone kernels. They fit to 10⁻⁹, but their implied memory strength (86–301 versus a true 0.31) makes the LMI infeasible. Such fits are used only to build independent reference solutions.

6. **One memory mechanism: the history window.** Fading memory (Erlang, exponential sums) is truncated to a window of length O(log 1/ε) with a proven error bound, so every class uses the main theorem. Memory that would need auxiliary modes is outside the scope and not in the software. *Found:* the functional family is conservative for strong, delayed memory (Erlang-4 at strength 0.8 is stable but not certified).

7. **Explicit constants in every cost.** Constants for the prior methods' linear solver (QLSA, 837κ at 10⁻¹⁰) and for LCHS (K and the coefficient 1-norm) come from the cited papers. Sparse block-encoding normalizations are measured. Competitors receive the same solver and a best-case readout.

8. **Validated, not just numerical, LK inequalities.** For time-dependent problems the LMI is checked on a grid and then extended to all t with a Lipschitz bound plus a round-off bound.

9. **References independent of the method under test.** These are method of steps with breaking points, manufactured solutions, exact characteristic modes, and high-order sum-of-exponential twins for kernels.

10. **Hardware validation, two parts.** (a) The energy law: the weighted propagator is a contraction, so a one-ancilla dilation measures the LK energy ratio directly; eight classes, with the unweighted Euclidean energy as a control. These circuits use a classically computed propagator and test the theorem's physical content, not the algorithm. (b) The algorithm: the LCHS circuit itself (PREP, SELECT of e^{-it(kL+H)}, PREP†, postselection), compiled from spec files for four equation classes, with zero-noise extrapolation. Each Hamiltonian evolution is compiled exactly from its small matrix; at 4 qubits only 4 LCHS terms fit the hardware noise budget, so the quadrature error (5-10%) is reported separately from hardware error, and end-to-end convergence is shown in noiseless circuit simulation.

11. **Safe spec parsing.** Expressions are checked against an AST whitelist before evaluation, so a spec file cannot run arbitrary code.

12. **High-precision framing, error-corrected setting.** Our fixed overhead makes first-order methods cheaper at ε ≥ 1e-2; the crossover is between 1e-2 and 1e-4, and the advantage grows tenfold per decade below it. The number of initial-data preparations is O(χ) at every ε. The paper states both, and that high precision requires error-corrected hardware.

13. **Analytic test histories.** Every comparison instance starts from its dominant characteristic mode (the state a long-running stable system converges to), so the exact solution is known in closed form and the spectral-convergence lemma applies.
