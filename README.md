# EESP: code and results

Code and raw results for

> S. Orujov, *Ensemble Exhaustive-Subsample Partitioning: What Subsample Ensembles Estimate in
> Clusterwise Least Squares, and When They Are Worth Using* (2026).

Repository: https://github.com/sorujov/eesp. Each release is archived on Zenodo.

Every table and figure in the paper and its Supplementary Material can be regenerated from the
files here, either from the stored results (a few seconds) or by rerunning the experiments.

## Contents

```
code/
  eespcore.py        exact micro-solve (branch and bound), alignment, EESP (Algorithm 1),
                     hard alternation, emEM, trimmed alternation, exact K=2 oracle
  eesp_variants.py   adaptively trimmed EESP (Algorithm 2) and the reweighting step
  simcommon.py       data-generating designs, seeding, output location
  exp_qprofile.py    Section 5.1-5.2 (replicate law against the exact minimiser)
  exp_optim.py       Section 5.3 (optimisers on clean data)
  exp_contam.py      Section 5.4 (contamination and the subsample-size window)
  exp_real.py        Section 6 (real data; "contam" and "flags" for the contaminated tone data)
  chunk_runner.py    runs the Section 5.3-5.4 units in chunks (used on a Slurm cluster)
  make_outputs.py    writes results_v3/tables/*.tex, results_v3/figures/*.pdf, summaries
  test_eespcore.py   unit tests (brute force, lstsq, exact oracle)
data/
  gnp.csv, usrec.csv US real GNP (GNPC96) and NBER recession indicator (USREC), FRED
results_v3/          raw results: one row per data set x method (CSV), E1 profiles (NPZ)
```

## Requirements

Python 3.11 with the packages in `requirements.txt`
(`pip install -r requirements.txt`).

The tone, NO and CO2 data are the `tonedata`, `NOdata` and `CO2data` sets of the R package
**mixtools** (GPL). They are not redistributed here. Download the package source from CRAN,
unpack it, and set `RDATA` to its `data/` directory:

```
export RDATA=/path/to/mixtools/data
```

## Reproducing

From `code/`:

```
python3 test_eespcore.py                 # unit tests
python3 make_outputs.py                  # tables and figures from the stored results
```

To rerun the experiments (outputs go to `$EESP_ROOT/results_v3`, where `EESP_ROOT` defaults to
the directory above `code/`):

```
python3 exp_qprofile.py 0 1              # E1   (JOB NJOBS: split cells across processes)
python3 exp_optim.py 0 1                 # E3
python3 exp_contam.py 0 1                # E4
python3 exp_real.py; python3 exp_real.py contam; python3 exp_real.py flags   # E6
```

E3 and E4 were run on a Slurm cluster with `chunk_runner.py`, which reads `UNIT_START`,
`UNIT_END`, `WORKERS`, `RESULTS_DIR` and `CHUNK_ID` from the environment and runs
`python3 chunk_runner.py e3|e4|all`. Every unit is seeded by a CRC32 hash of its design and
replication, so results do not depend on how the units are split. Timings depend on the
hardware; everything else regenerates exactly.

## Licence

MIT (see `LICENSE`).
