# Pre-registration — Gate 2: the group-recovery step (Algorithm 2)

Written 2026-09-27T13:13:25Z, before the recovery step has been applied to any data set of
the study. The step was developed on data generated with development seeds only
(SEED0 = 777777, 555555, 333333; checked once on 111111). The study data sets (parts 1, 1b,
1c, 1d, 1e of the paper, 6,400 data sets) have not been used in its development.

## The procedure under test

Algorithm 2 = the fit of Algorithm 1 exactly as stored in the study results (method adaptX),
followed by up to three recovery steps of `code/recover2.py`
(sha256 2c296504932757927a2c66cddcaf35cdfefd859ff6b5e293191c4512ba0b78f1), with its default
constants: c = 2.5, restriction factor 12, minimum support max(2(d+1), 3% of n), no recovery
when more than 1/3 is flagged, peak weight >= 1/2 within 3 thresholds, margin (d+1)/2 log n.
The random generator of the recovery step is seeded per data set by seed_for("v4rec", unit).
Helper modules: py_methods.py sha256 3a71036f..., eesp_variants.py 4f64fc1c..., eespcore.py fb3dba61....

## Measures

Accuracy on clean units as in the paper (nearest fitted surface, best relabelling). Paired
differences Algorithm 2 minus Algorithm 1 (or minus a competitor), per cell, with the paired
standard error SE.

## Criteria

(a) Recovery. In the small-group designs (groups 80/20 and 85/15) at eps in {0, 0.10, 0.20},
    the mean gain of Algorithm 2 over Algorithm 1 is at least 0.05 in every one of the six cells.
(b) No harm. In every cell of every part with eps <= 0.20, the mean difference Algorithm 2
    minus Algorithm 1 is at least -0.01.
(c) The pre-registered comparison of Gate 1, repeated with Algorithm 2: in each of the 28
    cells (designs D1, D2, D3, D4, D6, D7, D8; eps in {0, 0.05, 0.10, 0.20}), with the
    competitor of highest mean accuracy chosen after the fact among the Gate 1 competitors, the
    cell fails if the difference is below -0.03 or its upper 95% bound (diff + 1.96 SE) is below
    -0.02. (c) passes if no cell fails.

Reported, not gated: cells with eps >= 0.25; clustered outliers; the flagged fraction; the
added computing time; the comparison with the Laplace and noise-component mixtures.

## Decision

- (a), (b), (c) all pass: Algorithm 2 becomes the procedure of the paper; Algorithm 1 is
  reported as the first stage and as an ablation.
- (a) fails in some cell: the recovery is reported with that cell named; the claim is limited
  to the cells where (a) holds.
- (b) or (c) fails: the failing cells are reported with the reason; Algorithm 2 is not
  presented as uniformly better, and the paper states where it is worse.
No change to recover2.py after this point is claimed without a new pre-registered gate.
