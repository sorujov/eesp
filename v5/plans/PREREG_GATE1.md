# Gate 1: pre-registered analysis (written 27 Sept 2026, before any v4 run)

## Proposed method

**adaptX**: Algorithm 2 of v3, with covariate trimming added (W2). The v3
Algorithm 2 without covariate trimming is reported as **adapt**, as an
ablation.

## Competitors

Every method runs on the same 2,000 data sets.

- **Fixed trimming level**, with α ∈ {0.05, 0.10, 0.15, 0.25, 0.40}:
  - TCR(α): TCLUST-REG, FSDA `tclustreg`, restriction factor 12, nsamp 300,
    no second trimming;
  - TCR(α)+rw: TCR(α) followed by the reweighting step of Algorithm 2;
  - TCRx(α) and TCRx(α)+rw: TCLUST-REG with 5% second trimming, α ∈ {0.10,
    0.25}, run in D5 and D6 only;
  - tcwm(α): trimmed CWM (RobMixReg `trim.cwm`, 50 starts, 20
    concentration steps);
  - TLE(α): trimmed likelihood (RobMixReg `TLE`);
  - TA(α): home-made trimmed alternation + reweighting, at the time budget of
    adapt (supplement only).
- **No level**:
  - CTLE (RobMixReg `CTLERob`);
  - bisq (bisquare mixture, Bai, Yao & Boyer 2012; RobMixReg code with a
    scoping fix).
- **No protection**: HA50.

## Designs

- n = 300.
- ε ∈ {0, .05, .10, .20, .30}.
- 50 replicates per cell.
- Seeds fixed by the cell and the replicate.
- Designs D1–D8, as listed in `simv4.py`.

**Response-type designs:**

| design | description |
|---|---|
| D1 | K2 crossing |
| D2 | K3 parallel |
| D3 | unequal weights |
| D4 | unequal variances |
| D6 | uniform noise |
| D7 | p = 3 |
| D8 | well-separated parallel |

**Leverage design:** D5.

## Metric

The primary metric is **acc**: accuracy on the clean units of the
nearest-surface labels implied by each method's coefficients. Every method is
scored identically.

Secondary metrics:
- perr (the largest coefficient error);
- fail (acc < 0.7);
- ahat;
- time.

## Criteria

These are fixed now. Each difference is a paired mean over the 50
replicates, adaptX minus the competitor. SE is the paired standard error.

**(a) Not worse than the best competitor.**

In every response-type cell with ε ≤ 0.20, take the competitor with the
highest mean acc in that cell, chosen after the fact (this is conservative
against adaptX). The cell **fails** if either:
- the difference is below −0.03; or
- the upper 95% bound of the difference, diff + 1.96·SE, is below −0.02.

(a) passes if no cell fails.

**(b) No single fixed level works everywhere.**

For every fixed-level competitor, meaning every family and every α (TCR,
TCR+rw, tcwm, TLE), there must be at least one response-type cell with
ε ≤ 0.20 in which it trails adaptX by at least 0.05, with diff − 1.96·SE > 0.

**(c) Leverage (D5, ε ≤ 0.20).**

This is reported, not gated. adaptX is compared with TCRx+rw.

## Decision

- **Gate 1 passes** if (a) and (b) both hold.
- **If (a) fails** in some cell, the cell and the reason are reported to Sam
  before any decision.
- **If (b) fails**, the claim "no single fixed level works" is dropped, and
  so is the robust-methods framing.

Every figure is computed by `gate1_report.py` from `results_v4/metrics.csv`.
