"""Supplement Table S40: the long-search TCLUST-REG rows cell by cell, with the paired difference
from the procedure and its standard error (all 123 cells)."""
import os
import pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = pd.read_csv(os.path.join(ROOT, "results_v4", "tcrlong_cells.csv"))
R = R[~((R.study.astype(str) == "1") & (R.eps.round(2) == 0.30))]
W = R.pivot_table(index=["study", "design", "eps"], columns="method", values=["rival", "diff", "se"])
P = R.groupby(["study", "design", "eps"]).proc.first()
PART = {"1": "1", "1b": "2", "1c": "3", "1d": "3", "1e": "3", "1f": "3", "1g": "P4", "1h": "P5"}
L = ["{\\scriptsize", "\\begin{longtable}{llcccccc}",
     "\\caption{Long-search TCLUST-REG ($1000$ starts, $50$ refinement steps) with reweighting, alone and followed by Algorithm~2, at levels $0.25$ and $0.40$, cell by cell: mean accuracy on clean units, and the paired difference of the procedure from the rows with Algorithm~2 (standard error in brackets). Part: 1, 2 and 3 of the study, P4 and P5: Plans~4 and~5.}\\label{tab:S40} \\\\",
     "\\toprule", "Part & Design & $\\varepsilon$ & Procedure & $0.25$, rw & $0.25$, rw, A2 (diff.) & $0.40$, rw & $0.40$, rw, A2 (diff.) \\\\", "\\midrule", "\\endfirsthead",
     "\\multicolumn{8}{l}{\\emph{Table \\thetable, continued}} \\\\", "\\toprule",
     "Part & Design & $\\varepsilon$ & Procedure & $0.25$, rw & $0.25$, rw, A2 (diff.) & $0.40$, rw & $0.40$, rw, A2 (diff.) \\\\", "\\midrule", "\\endhead"]
for (st, d, e), row in W.iterrows():
    g = lambda m: row[("rival", m)]
    df = lambda m: f"{row[('diff', m)]:+.3f} ({row[('se', m)]:.3f})"
    L.append(f"{PART[str(st)]} & {d} & {e:.2f} & {P.loc[(st, d, e)]:.3f} & {g('TCRlong(0.25)+rw'):.3f} & {g('TCRlong(0.25)+rw+A2'):.3f} {df('TCRlong(0.25)+rw+A2')} & {g('TCRlong(0.40)+rw'):.3f} & {g('TCRlong(0.40)+rw+A2'):.3f} {df('TCRlong(0.40)+rw+A2')} \\\\")
L += ["\\bottomrule", "\\end{longtable}", "}"]
open(os.path.join(ROOT, "paper", "tables", "supp_S40.tex"), "w").write("\n".join(L).replace("_", "\\_").replace("\\\\_", "\\_"))
print(len(W))
