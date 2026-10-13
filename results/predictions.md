# Predictions for the IBM hardware validation

The section below was recorded before the hardware jobs were submitted and is kept verbatim. Its references to
"round 1" are to an earlier exploratory run with a different circuit design, which is not part of the paper. The results file named below is now `results/hardware.json`.

## Round 2: validation on ibm_marrakesh (recorded 2026-10-13, before submission)

ibm_marrakesh is a noisy IBM Heron r2 processor without error correction. Circuits: benchmarks/hardware.py.
Mitigation: Pauli twirling, XY4 dynamical decoupling, tensored readout correction, zero-noise extrapolation
(global folding, noise factors 1 and 3, linear). Noise-model (FakeMarrakesh) results are in
results/hardware_v2.json; in round 1 the noise model was 2-3x optimistic, and the predictions below allow
for that.

V1 (LK energy E(t)/E(0), dilation circuits, t = 0.1, 0.25, 0.5, 1.0; 24-28 CZ for the five 2-qubit classes,
about 140 CZ for the three 3-qubit classes):
 P1. 2-qubit classes (one delay, neutral, time-varying delay, bounded kernel, fading memory): weighted
     energies after ZNE within 0.06 of ideal at every point; no weighted value above 1.03; each weighted
     curve non-increasing in t to within 0.05.
 P2. Unweighted Euclidean energy from the worst direction rises above 1 at t = 0.1 and 0.25 for one delay,
     neutral, time-varying delay and bounded kernel (ideal 1.17-1.31): measured (ZNE) >= 1.10 at the
     larger of the two times. For fading memory the ideal rise (1.03-1.04) is below the resolution.
 P3. 3-qubit classes (two delays, drug reabsorption, baroreflex): ZNE within 0.12 of ideal. Noise biases
     these energies downward, so "weighted <= 1" is not a meaningful test there. The unweighted rise is at
     most marginally visible for baroreflex (ideal 1.16-1.19) and two delays (1.07-1.13), and not visible for
     drug reabsorption (1.01-1.03).
V2 (4-term LCHS, 38 CZ; one delay, neutral, bounded kernel, fading memory; t = 0.5-2):
 P4. x(t) after ZNE within 4% of the ideal 4-term circuit on average (max 10%); readout correction alone,
     about 5% on average as in round 1.
 P5. Against the exact solution the error is dominated by the 4-term quadrature: mean about 6%, up to
     14% for fading memory.
V3 (window readout): median deviation from the ideal circuit <= 8% after ZNE.

## Round 2 outcomes (jobs db6ipn04qg6s73c4shcg (V1) and db6ipnklf4us73c5eqe0 (V2), ibm_marrakesh, 176 s QPU)

| Prediction | Outcome | Verdict |
| --- | --- | --- |
| P1: 2-qubit weighted energies within 0.06 of ideal after ZNE; none above 1.03; non-increasing | Mean deviation 0.028, max 0.063 (one point of 40 just above 0.06); largest weighted value 0.954; every weighted curve non-increasing | Confirmed (one marginal point) |
| P2: unweighted energy >= 1.10 for one delay, neutral, time-varying, bounded kernel; fading not resolved | Peaks 1.229, 1.194, 1.290, 1.255 (ideal 1.280, 1.237, 1.310, 1.263); fading 1.002 (ideal 1.044) | Confirmed |
| P3: 3-qubit classes within 0.12 of ideal | Mean deviation 0.114, max 0.204 (two delays and baroreflex exceed 0.12); unweighted rise not visible for any of the three | Refuted for two delays and baroreflex: ~140-CZ circuits are too noisy for a quantitative test |
| P4: V2 ZNE within 4% of ideal on average, max 10%; readout correction alone about 5% | ZNE mean 3.6%, max 7.7%; readout correction alone mean 7.1% (worse than round 1's 5.0%) | Confirmed for ZNE; readout-only worse than predicted |
| P5: vs exact mean about 6%, up to 14% (fading) | Mean 6.2%, max 17.3% (fading memory, t = 1.5) | Mostly confirmed (max higher) |
| V3: window median <= 8% after ZNE | Median 4.7%, 90th percentile 13.1% (max 91% on a near-zero amplitude) | Confirmed |
| Hardware below ideal (not predicted this round) | 15 of 16 V2 points below ideal after ZNE | Consistent with round 1 |
