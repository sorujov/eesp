# EESP / ESF: group recovery after trimming, and level-free flagging, in robust clusterwise regression

Code, raw results and analysis plans for the manuscript submitted to *Computational Statistics*
(29 September 2026). "eesp" is the name of an earlier version of the method; the level-free
procedure is called ESF in the paper, and the group-recovery step is `code/recover2.py`.

- `code/`: ESF (`py_methods.eesp_adaptive_x`), the recovery step (`recover2.recover`), the data
  generators (`simv4.py`), the TCLUST-REG drivers for Octave + FSDA (`oct_long.m`, `oct_r9.m`,
  `oct_idx.m`, with the substitutes in `fsda_shim/`), the cluster job scripts, and the scripts
  that write every table and figure (`make_main_tables.py`, `make_r9_table.py`,
  `make_r10_tables.py`, `make_r12_extra.py`, `make_supplement.py`, `make_supp_r3.py`,
  `relabel.py`, `fig_illustr.py`).
- `matlab_check/`: the MATLAB driver and the 80 data sets of the MATLAB check of TCLUST-REG
  (Supplement, Section S1, Table S51); `results_matlab/`: its output and the comparison with the
  Octave runs (`code/mlcompare.py`, `code/mlobj.py`, `code/make_matlab_table.py`).
- `results_*`: raw per-data-set output of every run (7,750 simulated data sets and the
  additional runs of review rounds 8 to 13).
- `realdata/`: fishery, tone and taxi analyses; `code/data/taxi_jfk_2024-03.csv` is the taxi extract
  (NYC TLC public trip records).
- `plans/`: the five analysis plans (`PREREG_*.md`), the gate reports, and `HASHES.txt` with their SHA-256 digests (Plan 5 hashed
  without its last line, as stated in Supplement Table S34).
- `paper/`, `springer/`: LaTeX sources of the supplement and the manuscript.
- `submission/`: the files uploaded to the journal.

Paths in `plans/HASHES.txt` are relative to this folder (`v5/`).
