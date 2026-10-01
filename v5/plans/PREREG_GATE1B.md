# Gate 1b: pre-registered confirmatory test (written 27 Sept 2026, after Gate 1, before any 1b run)

## Why there is a Gate 1b

Gate 1 **failed** its criterion (b), in one competitor only. TCLUST-REG at
the conservative level 0.25, followed by the reweighting step
(**TCR(0.25)+rw**), was never more than 0.033 below adaptX in any cell with
ε ≤ 0.20. Criterion (a) passed in all 28 cells.

A pattern appeared in the ε = 0.30 cells. Those cells were **not** part of
the pre-registered criteria, so the pattern is exploratory: TCR(0.25)+rw
collapsed to about 0.6 in five of seven response designs, and adaptX held at
about 0.9 or above in those five. The alternative conservative level, 0.40,
had already failed in Gate 1 when one group is small (D3, ε = 0: 0.67
against 0.90).

Gate 1b tests the exploratory pattern on fresh data, so that the paper can
state it as a confirmed result rather than a post-hoc one.

## Design

- Same eight designs, methods and code as Gate 1.
- New seed base (SEED0 = 20260928).
- ε ∈ {0.25, 0.30, 0.35}.
- 100 replicates per cell.
- 2,400 data sets in all.

## Criteria

Response-type designs are D1, D2, D3, D4, D6, D7 and D8. Differences are
paired, adaptX minus the competitor, with 95% intervals given by ±1.96 SE.

**(b′) Gated.** TCR(0.25)+rw must trail adaptX by at least 0.05, with
diff − 1.96·SE > 0, in at least one response-type cell of Gate 1b.

If (b′) holds, then together with Gate 1 (where TCR(0.40)+rw trails by
0.233 in D3 at ε = 0), neither conservative level works in every design. The
statement "no fixed level, with or without reweighting, matches adaptX in
every design" is then confirmed for all the fixed-level competitors run
(TCR, TCR+rw, tcwm and TLE, each at 0.05–0.40).

**(d) Reported, not gated: adaptX's own limits.**
- Every cell of Gate 1b in which adaptX's mean accuracy is below 0.85.
- Every cell in which some competitor beats adaptX by more than 0.03, with
  diff + 1.96·SE < 0.

Both are stated in the paper as limits of the method.

## Decision

- **(b′) holds:** Gate 1 is passed with the corrected claim above. Writing
  starts.
- **(b′) fails:** the robust-methods framing is dropped, and the options go
  to Sam.
