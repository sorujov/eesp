# Plan 4: fresh-data confirmation of the recovery step (written 2026-09-27T17:25:18Z, before any 1g data were generated)

Reason: review round 3 (referee concern 4) noted that Plan 3 confirmed Algorithm 2 on the same
data sets (part 3) on which the small-group failure of Algorithm 1 was first observed.

Code frozen: recover2.py sha256 2c296504932757927a2c66cddcaf35cdfefd859ff6b5e293191c4512ba0b78f1
(identical to Plan 3); py_methods.py sha256 fa89931995c2e592f97fa9f939c9583f4d9eae8f356abdcdb323785c5829eec8
(Plan 3 version plus the zero-scale guard, which changes nothing on data with continuous errors).
Procedure = Algorithm 1 (py_methods.eesp_adaptive_x, B=100, constants as in the paper) followed by
Algorithm 2 (recover2.recover, at most 3 applications), exactly as in run_rec.py; one run, seed 0.

Data: V4_STUDY=1g in simv4.py, SEED0 20261005 (never used before), 50 data sets per cell,
eps in {0, 0.10, 0.20}, outliers in the response as in D1:
- G1f: groups 80/20 (as G1, fresh seeds), m=16
- G2f: groups 85/15 (as G2, fresh seeds), m=20
- G4: groups 80/20 with three covariates (as D7), m=20 (new design)
- G5: two parallel lines six noise scales apart, groups 85/15, m=20 (new design)
600 data sets. Competitors: the same as in the paper (TCLUST-REG at five levels with and without
reweighting, trimmed CWM, TLE, CTLE, bisquare, Laplace, contaminated normal and noise-component
mixtures, least squares).

Criteria (all must hold):
(a) G1f and G2f: in each of the 6 cells, mean paired accuracy gain of the procedure over
    Algorithm 1 alone >= 0.05.
(b) Every one of the 12 cells: mean paired difference procedure - Algorithm 1 >= -0.01.
(c) Every one of the 12 cells: mean paired difference procedure - best competitor (chosen after
    the run, per cell) >= -0.03, and difference + 1.96 SE >= -0.02.
Reported, not gated: flagged fraction, coefficient error, how often the step changed the fit.
