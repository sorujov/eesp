"""Gate 1c (descriptive): sample size, four groups, correlated covariates."""
import os, sys
import numpy as np, pandas as pd
_argv = list(sys.argv); sys.argv = [sys.argv[0], os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
import make_v4_outputs as O
sys.argv = _argv
M = pd.concat([pd.read_csv(f) for f in sys.argv[1].split(",")], ignore_index=True); M.loc[M.acc.isna(), "acc"] = 0
SH = {"E1-K2cross-n100": "D1, $n=100$", "E2-K2cross-n1000": "D1, $n=1000$",
      "E3-unequal-n1000": "D3, $n=1000$", "E4-K4par": "four parallel lines, $n=400$",
      "E5-p3corr": "D7, dependent covariates",
      "F1-K2cross-moderate": "D1, moderate outliers", "F2-unequal-moderate": "D3, moderate outliers",
      "G1-small20": "D1, groups of 80\\% and 20\\%", "G2-small15": "D1, groups of 85\\% and 15\\%",
      "G3-cluster": "D1, clustered outliers",
      "H1-K2cross-t3": "D1, $t_3$ errors", "H2-unequal-t3": "D3, $t_3$ errors"}
comp = [m for m in M.method.unique() if m not in ("Alg2", "adaptX", "adapt", "HA50")]
rows = []
for d in SH:
    for e in sorted(M.eps.unique()):
        g = M[(M.design == d) & (M.eps == e)]
        if g.empty: continue
        mm = g.groupby("method").acc.mean()
        b = mm[[c for c in comp if c in mm.index]].idxmax()
        A = g[g.method == "Alg2"].set_index("rep").acc; B = g[g.method == b].set_index("rep").acc
        dd = (A - B)
        # worst fixed level: best single level of TCLUST-REG+rw across this design
        rows.append(dict(design=d, eps=e, alg=mm["Alg2"], alg1=mm["adaptX"], best=b, bestacc=mm[b], diff=dd.mean(),
                         se=dd.std(ddof=1) / np.sqrt(len(dd)), tcr25=mm.get("TCR(0.25)+rw", np.nan),
                         tcr40=mm.get("TCR(0.40)+rw", np.nan), ahat=g[g.method == "Alg2"].ahat.mean(),
                         t_alg=g[g.method == "Alg2"].time.median(), t_tcr=g[g.method.str.startswith("TCR(")].time.median()))
R = pd.DataFrame(rows)
print(R.round(3).to_string())
L = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
     "\\caption{Further designs (third part of the study, descriptive, $50$ data sets per cell): mean accuracy on clean units of the proposed procedure (Algorithms~1 and~2), of Algorithm~1 alone, of the best competitor chosen afterwards in each cell, and of TCLUST-REG with reweighting at the two conservative levels ($300$ starts); $\\hat\\alpha$ is the mean flagged fraction of the proposed procedure.}",
     "\\label{tab:further}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lccclcccc}", "\\toprule",
     "Design & $\\varepsilon$ & Proposed & Alg.~1 alone & Best competitor & & TCR$_{0.25}$+rw & TCR$_{0.40}$+rw & $\\hat\\alpha$ \\\\", "\\midrule"]
for d in SH:
    first = True
    for _, r in R[R.design == d].iterrows():
        L.append(f"{SH[d] if first else ''} & {r.eps:.2f} & {r.alg:.3f} & {r.alg1:.3f} & {O.label(r.best)} & {r.bestacc:.3f} & {r.tcr25:.3f} & {r.tcr40:.3f} & {r.ahat:.3f} \\\\")
        first = False
    L.append("\\addlinespace")
L[-1] = "\\bottomrule"
L += ["\\end{tabular}}", "\\end{table}"]
open(os.path.join(sys.argv[2], "further.tex"), "w").write("\n".join(L))
