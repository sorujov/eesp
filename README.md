# Group recovery after trimming, and level-free flagging, in robust clusterwise regression

Samir Orujov, School of Business, ADA University, Baku

This repository holds the code, the raw simulation output and the pre-registered analysis plans behind the paper, which is under review at *Computational Statistics*. Every table and figure in the paper and its supplement is produced by a script in this repository from the stored results; no number is typed by hand.

**In one paragraph.** Clusterwise regression splits the data into groups, each with its own linear fit. Trimming methods such as TCLUST-REG make it robust to outliers by discarding a fixed fraction of the data: too low a fraction breaks the fit, too generous a fraction can trim away a small group. The paper makes two proposals. The first is a group-recovery step that can follow any trimming or flagging method: it searches the discarded units for a line and restores it as a group when the evidence for it is strong enough. The second is ESF (exact-subsample flagging), which flags outliers without a trimming level by solving the clusterwise least-squares problem exactly on small subsamples. Both are evaluated in a pre-registered simulation study on 7,750 data sets and on real data (fishery, tone perception, and taxi fares with known tariffs).

## Where to start

| If you want to | Look in |
|---|---|
| read the paper | `v5/springer/main.pdf` (manuscript) and `v5/paper/supplement.pdf` (supplement) |
| see what was submitted to the journal | `v5/submission/` |
| run the method on your own data | `v5/code/` (start with `v5/README.md`) |
| reproduce a table or figure | `v5/code/make_*.py`, which read the stored results in `v5/results_*/` |
| check the analysis plans | `v5/plans/`: the five plans (`PREREG_*.md`), their reports and their SHA-256 digests (`HASHES.txt`) |
| see the earlier version of the method (EESP) | `v3/` |

## Layout

```
v5/                 current version (ESF and group recovery)
  code/             methods, data generators, cluster job scripts, table and figure scripts
  plans/            pre-registered analysis plans, their reports and digests
  results_*/        raw per-data-set output of every simulation run
  matlab_check/     MATLAB check of TCLUST-REG (Supplement S1, Table S51)
  realdata/         fishery, tone and taxi analyses
  paper/            LaTeX of the supplement and the generated tables and figures
  springer/         LaTeX of the manuscript (Springer Nature template)
  submission/       the files uploaded to the journal, by version
v3/                 earlier version of the method (EESP): code, data, results
```

## Requirements

- Python 3.11 with the packages in `requirements.txt`:

  ```
  pip install -r requirements.txt
  ```

- TCLUST-REG runs use GNU Octave with the FSDA toolbox (see `v5/README.md`).
- The tone, NO and CO2 data are the `tonedata`, `NOdata` and `CO2data` sets of the R package **mixtools** (GPL) and are not redistributed here. Download the mixtools source from CRAN, unpack it, and set the environment variable `RDATA` to its `data/` directory (by default the scripts look in `data/mixtools/`).

## Citation

Orujov, S. (2026). Group recovery after trimming, and level-free flagging, in robust clusterwise regression. Working paper, ADA University. See `CITATION.cff` for the software citation; each release is archived on Zenodo.

## Licence

Code: MIT (see `LICENSE`). The text of the paper remains the author's; please cite it rather than reuse it.
