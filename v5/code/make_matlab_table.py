"""Supplement S1: TCLUST-REG in MATLAB against the Octave runs of the paper on 80 data sets
(results_matlab/compare.csv from mlcompare.py, crit.csv from mlobj.py).  Descriptive."""
import os
import pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = pd.read_csv(os.path.join(ROOT, "results_matlab", "compare.csv"))
C = pd.read_csv(os.path.join(ROOT, "results_matlab", "crit.csv"))
k = ["study", "design", "eps", "rep", "method"]
for X in (D, C):
    X["eps"] = X.eps.round(2); X["study"] = X.study.astype(str)
D = D.merge(C[k + ["c_ml", "c_oct"]], on=k)
DN = {"D1-K2cross": "D1", "D2-K3par": "D2", "D5-leverage": "D5", "D6-uniform": "D6", "E4-K4par": "$K=4$"}
L = ["\\begin{tabular}{llcccccc}", "\\toprule",
     "Design & $\\varepsilon$ & $\\alpha$ & Same fit & MATLAB & Octave & MATLAB higher & Octave higher \\\\", "\\midrule"]
for (st, d, e, m), g in D.groupby(["study", "design", "eps", "method"], sort=False):
    dc = g.c_ml - g.c_oct
    L.append(f"{DN[d]} & {e:.2f} & {m[8:12]} & {int((g.coef_diff < 1e-3).sum())} & {g.accA2_matlab.mean():.3f} & "
             f"{g.accA2_octave.mean():.3f} & {int((dc > 0.5).sum())} & {int((dc < -0.5).sum())} \\\\")
L += ["\\midrule",
      f"All & & & {int((D.coef_diff < 1e-3).sum())} & {D.accA2_matlab.mean():.3f} & {D.accA2_octave.mean():.3f} & "
      f"{int(((D.c_ml - D.c_oct) > 0.5).sum())} & {int(((D.c_ml - D.c_oct) < -0.5).sum())} \\\\",
      "\\bottomrule", "\\end{tabular}"]
open(os.path.join(ROOT, "paper", "tables", "supp_matlab.tex"), "w").write("\n".join(L))
print("\n".join(L))
r = D.time_octave / D.time_matlab
print("time ratio median", round(r.median(), 1), "MATLAB median s", round(D.time_matlab.median(), 2), "Octave median s", round(D.time_octave.median(), 1))
