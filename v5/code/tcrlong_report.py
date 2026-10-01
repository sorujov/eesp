"""Round 8: the rival pipeline with the long TCLUST-REG search against the procedure, all parts.
Usage: python tcrlong_report.py TCRLONG_CSV_DIR"""
import glob, os, sys
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
T = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(sys.argv[1], "tcrlong_*.csv"))])
FOLD = {"1": "results_v4", "1b": "results_v4b", "1c": "results_v4c", "1d": "results_v4d",
        "1e": "results_v4e", "1f": "results_v4f", "1g": "results_v4g", "1h": "results_v4h"}
A = []
for st, f in FOLD.items():
    a = pd.read_csv(os.path.join(ROOT, f, "alg2.csv"))[["design", "eps", "rep", "unit", "acc", "ahat"]]
    a["study"] = st; A.append(a)
A = pd.concat(A)
rows = []
for meth in sorted(T.method.unique()):
    t = T[T.method == meth][["study", "unit", "acc", "ahat"]]
    m = A.merge(t, on=["study", "unit"], suffixes=("", "_r"))
    for (st, des, e), g in m.groupby(["study", "design", "eps"]):
        dd = g.acc - g.acc_r
        rows.append(dict(method=meth, study=st, design=des, eps=e, n=len(g), proc=g.acc.mean(), rival=g.acc_r.mean(),
                         diff=dd.mean(), se=dd.std(ddof=1) / np.sqrt(len(g)), ahat_rival=g.ahat_r.mean()))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(ROOT, "results_v4", "tcrlong_cells.csv"), index=False)
pd.set_option("display.width", 200)
for meth in ["TCRlong(0.40)+rw+A2", "TCRlong(0.25)+rw+A2", "TCRlong(0.40)+rw", "TCRlong(0.40)"]:
    r = R[R.method == meth]
    print("\n==", meth, "cells", len(r), "units", r.n.sum())
    print(r[["study", "design", "eps", "proc", "rival", "diff", "se"]].round(3).to_string(index=False))
