# Plan 5: designs outside the Gaussian-crossing family (written 2026-09-27T18:49:40Z, before any 1h data were generated)

Reason: external reading of the study (CSDA referee) noted that most designs are variants of two
crossing lines with Gaussian errors; the range of validity of the procedure and of its flagged
fraction under non-Gaussian and heteroscedastic errors and with three covariates and unequal
groups was not tested.

Code: recover2.py and py_methods.py as in Plan 4 (hashes below); procedure = Algorithm 1
(B=100) + Algorithm 2, one run, seeds as in the study (run_py.py, run_rec.py).
Data: V4_STUDY=1h, SEED0 20261006 (never used), 50 data sets per cell, eps in {0, 0.10, 0.20},
outliers in the response as in D1:
- S1: D1 with skewed errors (chi-square 3 df, centred, unit variance), m=8
- S2: groups 70/30 with three covariates (as D7 with the weights of D3), m=16
- S3: D1 with error scale 0.5 + 0.5|x| (heteroscedastic in the covariate), m=8
Competitors: as in Plan 4 (all methods of the study, incl. Laplace, contaminated normal and
noise-component mixtures).
Criterion (c): in each of the 9 cells, mean paired difference procedure - best competitor
(chosen after the run) >= -0.03 and difference + 1.96 SE >= -0.02.
Prediction stated in advance: S3 may fail (the paper lists heteroscedastic errors as a limit).
Reported, not gated: flagged fraction (residual-flagged and covariate-flagged separately),
gain over Algorithm 1.
2c296504932757927a2c66cddcaf35cdfefd859ff6b5e293191c4512ba0b78f1  code/recover2.py
af21748f996645810fb73bdc6c47edfd8ac6b3011f538bdb15016df3d0e69962  code/py_methods.py
89e62b355fe09b4dc7c478698ae435181a1a6fba80309b2c9ec9f2dd12f8f7f7  code/simv4.py
Note: py_methods.py differs from the Plan 4 file only by the optional argument z_init of noise_mix (default None, unused by the plan).
