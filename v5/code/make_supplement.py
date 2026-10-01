"""Supplement tables: S1 (pre-registered comparison, Gate 1), S2 (lambda ablation, v3 study)."""
import os, sys
import numpy as np, pandas as pd
ROOT = sys.argv[1]; V3 = sys.argv[2]
TAB = os.path.join(ROOT, "paper", "tables")
M = pd.read_csv(os.path.join(ROOT, "results_v4", "metrics.csv")); M.loc[M.acc.isna(), "acc"] = 0
RESP = ["D1-K2cross", "D2-K3par", "D3-unequal", "D4-hetero", "D6-uniform", "D7-p3", "D8-K2par-sep"]
comp = [m for m in M.method.unique() if m not in ("adaptX", "adapt")]
import make_v4_outputs as O
L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize",
     "\\caption{Pre-registered comparison (first part of the study): in each response design and level, the competitor with the highest mean accuracy, chosen afterwards, and the paired difference Algorithm 1 minus that competitor with its standard error (50 data sets).}",
     "\\label{tab:S1}", "\\begin{tabular}{llcclcc}", "\\toprule",
     "Design & $\\varepsilon$ & Algorithm 1 & Best competitor & & Difference & SE \\\\", "\\midrule"]
for d in RESP:
    for e in [0, 0.05, 0.1, 0.2]:
        g = M[(M.design == d) & (M.eps == e)]
        mm = g.groupby("method").acc.mean()
        b = mm[[c for c in comp if c in mm.index]].idxmax()
        A = g[g.method == "adaptX"].set_index("rep").acc; B = g[g.method == b].set_index("rep").acc
        dd = A - B
        L.append(f"{O.SHORT[d]} & {e:.2f} & {mm['adaptX']:.3f} & {O.label(b)} & {mm[b]:.3f} & {dd.mean():+.3f} & {dd.std(ddof=1)/np.sqrt(len(dd)):.3f} \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(TAB, "supp_S1.tex"), "w").write("\n".join(L))
# lambda ablation from the v3 study
E = pd.read_csv(os.path.join(V3, "e4_rows_cluster.csv"))
E = E[E.part == "a"]
t = E[E.method.isin(["adapt", "adapt-sel", "adapt-lam0"])].groupby(["cell", "eps", "method"]).acc_clean.mean().unstack()
L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize",
     "\\caption{Preliminary study (earlier version of the procedure without the covariate screen, other data): mean accuracy on clean units with vote weight $\\lambda=3$, $\\lambda=\\infty$ (the best replicate alone) and $\\lambda=0$ (equal votes).}",
     "\\label{tab:S2}", "\\begin{tabular}{llccc}", "\\toprule",
     "Design & $\\varepsilon$ & $\\lambda=3$ & $\\lambda=\\infty$ & $\\lambda=0$ \\\\", "\\midrule"]
for (c, e), r in t.iterrows():
    L.append(f"{c} & {e:.2f} & {r['adapt']:.3f} & {r['adapt-sel']:.3f} & {r['adapt-lam0']:.3f} \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(TAB, "supp_S2.tex"), "w").write("\n".join(L))
print((t['adapt-sel']-t['adapt']).abs().max(), (t['adapt']-t['adapt-lam0']).max())
# S3: sensitivity to c and the covariate-screen quantile (sens.py output)
d = pd.read_csv(os.path.join(ROOT, "results_v4", "sens.csv"))
t = d.groupby(["design", "eps", "variant"]).acc.mean().unstack()
cols = ["c=2.0", "c=2.5", "c=3.0", "q=0.95", "q=0.99"]
S = {"D1-K2cross": "D1", "D3-unequal": "D3", "D5-leverage": "D5", "D6-uniform": "D6"}
L = ["\\begin{table}[htbp]", "\\centering", "\\small",
     "\\caption{Sensitivity of Algorithm~1 to the flagging cut-off $c$ (with the covariate screen at the $0.975$ quantile) and to the quantile $q$ of the covariate screen (with $c=2.5$): mean accuracy on clean units on the data of the first part ($50$ data sets per cell). These runs use a different random seed from those of Tables~S4--S11, so the column $c=2.5$ differs from them by the seed-to-seed variation of the procedure (up to $0.03$ in D3). The values used in the paper are $c=2.5$ and $q=0.975$.}",
     "\\label{tab:S3}", "\\begin{tabular}{llccccc}", "\\toprule",
     "Design & $\\varepsilon$ & $c=2$ & $c=2.5$ & $c=3$ & $q=0.95$ & $q=0.99$ \\\\", "\\midrule"]
for (dsg, e), r in t.iterrows():
    L.append(f"{S[dsg]} & {e:.2f} & " + " & ".join(f"{r[c]:.3f}" for c in cols) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(TAB, "supp_S3.tex"), "w").write("\n".join(L))
# S12: subsample size m (msens.py output)
d = pd.read_csv(os.path.join(ROOT, "results_v4", "msens.csv"))
t = d[d.kind == "m"].groupby(["design", "eps", "m"]).acc.mean().unstack()
S = {"D1-K2cross": "D1", "D2-K3par": "D2", "D3-unequal": "D3", "D6-uniform": "D6"}
ms = list(t.columns)
L = ["\\begin{table}[htbp]", "\\centering", "\\small",
     "\\caption{Sensitivity of Algorithm~1 to the subsample size $m$: mean accuracy on clean units on the data of the first part ($50$ data sets per cell; a different random seed from Tables~S4--S11). Dashes: size not run for that design. The sizes used in the paper are $m=8$ (D1, D6), $12$ (D2) and $10$ (D3).}",
     "\\label{tab:S12}", "\\begin{tabular}{ll" + "c" * len(ms) + "}", "\\toprule",
     "Design & $\\varepsilon$ & " + " & ".join(f"$m={int(x)}$" for x in ms) + " \\\\", "\\midrule"]
for (dsg, e), r in t.iterrows():
    L.append(f"{S[dsg]} & {e:.2f} & " + " & ".join("--" if pd.isna(r[c]) else f"{r[c]:.3f}" for c in ms) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(TAB, "supp_S12.tex"), "w").write("\n".join(L))
# S13: bound c_b of the scoring loss (cbsens.py output; parts 1, 2 and 3)
parts = [("results_v4", "1"), ("results_v4b", "2"), ("results_v4e", "3")]
S = {**O.SHORT, "G1-small20": "D1, groups 80/20", "G2-small15": "D1, groups 85/15", "G3-cluster": "D1, clustered outliers"}
L = ["\\begin{table}[htbp]", "\\centering", "\\small",
     "\\caption{Sensitivity of Algorithm~1 to the bound $c_b$ of the scoring loss: mean accuracy on clean units ($50$ data sets per cell, from the part of the study shown; a different random seed from the main tables). The value used in the paper is $c_b=3$. A larger bound recovers small groups but lowers the breakdown point in the other designs.}",
     "\\label{tab:S13}", "\\begin{tabular}{lllccc}", "\\toprule",
     "Part & Design & $\\varepsilon$ & $c_b=3$ & $c_b=4$ & $c_b=5$ \\\\", "\\midrule"]
for fold, part in parts:
    d = pd.read_csv(os.path.join(ROOT, fold, "cbsens.csv"))
    t = d.groupby(["design", "eps", "cb"]).acc.mean().unstack()
    for (dsg, e), r in t.iterrows():
        L.append(f"{part} & {S[dsg]} & {e:.2f} & " + " & ".join(f"{r[c]:.3f}" for c in (3.0, 4.0, 5.0)) + " \\\\")
    L.append("\\addlinespace")
L[-1] = "\\bottomrule"
L += ["\\end{tabular}", "\\end{table}"]
open(os.path.join(TAB, "supp_S13.tex"), "w").write("\n".join(L))
# S14: subsample solver ablation (solver_ablation.py output; parts 1, 2 and 3)
L = ["\\begin{table}[htbp]", "\\centering", "\\small",
     "\\caption{Algorithm~1 with the exact subsample fit replaced by one start of hard alternation on the subsample, by ten starts, or by elemental fits ($d$ units drawn for each group and fitted exactly): mean accuracy on clean units ($50$ data sets per cell) and median time per data set in seconds. Everything else is unchanged; the random seed differs from the main tables.}",
     "\\label{tab:S14}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lllcccc}", "\\toprule",
     "Part & Design & $\\varepsilon$ & Exact & Alternation, 1 start & Alternation, 10 starts & Elemental \\\\", "\\midrule"]
S = {**O.SHORT, "E4-K4par": "four parallel lines"}
cols = ["exact", "local", "multistart", "elemental"]
for fold, part in [("results_v4", "1"), ("results_v4b", "2"), ("results_v4c", "3")]:
    d = pd.read_csv(os.path.join(ROOT, fold, "solver.csv"))
    t = d.groupby(["design", "eps", "solver"]).acc.mean().unstack()
    for (dsg, e), r in t.iterrows():
        L.append(f"{part} & {S[dsg]} & {e:.2f} & " + " & ".join(f"{r[c]:.3f}" for c in cols) + " \\\\")
    tm = d.groupby("solver").time.median()
    L.append(f" & time (s) & & " + " & ".join(f"{tm[c]:.2f}" for c in cols) + " \\\\")
    L.append("\\addlinespace")
L[-1] = "\\bottomrule"
L += ["\\end{tabular}}", "\\end{table}"]
open(os.path.join(TAB, "supp_S14.tex"), "w").write("\n".join(L))
# S16: precision and recall of the flagged set of the procedure (flags_pr.py output)
S = {**O.SHORT}
L = ["\\begin{table}[htbp]", "\\centering", "\\small",
     "\\caption{The flagged set of the procedure: fraction of clean units flagged, fraction of outliers flagged (recall) and fraction of flagged units that are outliers (precision), means over data sets (first part for $\\varepsilon\\le0.20$, second part beyond).}",
     "\\label{tab:S16}", "\\begin{tabular}{llccc}", "\\toprule",
     "Design & $\\varepsilon$ & Clean flagged & Recall & Precision \\\\", "\\midrule"]
P1 = pd.read_csv(os.path.join(ROOT, "results_v4", "flags_pr.csv")); P1 = P1[P1.eps <= 0.2 + 1e-9]
P2 = pd.read_csv(os.path.join(ROOT, "results_v4b", "flags_pr.csv"))
PP = pd.concat([P1, P2]).groupby(["design", "eps"])[["clean_flagged", "recall", "precision"]].mean()
for dsg in ["D1-K2cross", "D2-K3par", "D3-unequal", "D4-hetero", "D6-uniform", "D7-p3", "D8-K2par-sep", "D5-leverage"]:
    for e, r in PP.loc[dsg].iterrows():
        f = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
        L.append(f"{S[dsg]} & {e:.2f} & {f(r.clean_flagged)} & {f(r.recall)} & {f(r.precision)} \\\\")
    L.append("\\addlinespace")
L[-1] = "\\bottomrule"
L += ["\\end{tabular}", "\\end{table}"]
open(os.path.join(TAB, "supp_S16.tex"), "w").write("\n".join(L))
# S17: the recovery step after TCLUST-REG with reweighting (rec_after_tcr.py output)
S2 = {**O.SHORT, "G1-small20": "groups 80/20", "G2-small15": "groups 85/15", "E4-K4par": "four groups", "E1-K2cross-n100": "D1, $n=100$", "E2-K2cross-n1000": "D1, $n=1000$", "E3-unequal-n1000": "D3, $n=1000$", "E5-p3corr": "D7, correlated", "F1-K2cross-moderate": "D1, moderate", "F2-unequal-moderate": "D3, moderate", "G3-cluster": "clustered outliers"}
L = ["\\begin{table}[htbp]", "\\centering", "\\small",
     "\\caption{Algorithm~2 applied after TCLUST-REG at levels $0.25$ and $0.40$ ($300$ starts, $10$ refinement steps; the long search is in Table~S40) with reweighting: mean accuracy on clean units without and with the recovery step, in the designs where it changes the mean by more than $0.01$.}",
     "\\label{tab:S17}", "\\begin{tabular}{llcccc}", "\\toprule",
     " & & \\multicolumn{2}{c}{$\\alpha=0.25$} & \\multicolumn{2}{c}{$\\alpha=0.40$} \\\\", "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
     "Design & $\\varepsilon$ & without & with & without & with \\\\", "\\midrule"]
_r1 = pd.read_csv(os.path.join(ROOT, "results_v4", "rec_after_tcr.csv")); _r1 = _r1[_r1.eps.round(2) != 0.30]
RT = pd.concat([_r1] + [pd.read_csv(os.path.join(ROOT, f, "rec_after_tcr.csv")) for f in ["results_v4b", "results_v4c", "results_v4d", "results_v4e"]])
G = RT.groupby(["design", "eps", "method"])[["acc", "acc_rec"]].mean().unstack("method")
for (dsg, e), r in G.iterrows():
    d25 = r[("acc_rec", "TCR(0.25)+rw")] - r[("acc", "TCR(0.25)+rw")]
    d40 = r[("acc_rec", "TCR(0.40)+rw")] - r[("acc", "TCR(0.40)+rw")]
    if max(abs(d25), abs(d40)) > 0.01:
        L.append(f"{S2.get(dsg, dsg)} & {e:.2f} & {r[('acc', 'TCR(0.25)+rw')]:.3f} & {r[('acc_rec', 'TCR(0.25)+rw')]:.3f} & {r[('acc', 'TCR(0.40)+rw')]:.3f} & {r[('acc_rec', 'TCR(0.40)+rw')]:.3f} \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(TAB, "supp_S17.tex"), "w").write("\n".join(L))
# S18: choice of K by BIC (ksel_sim.py output)
S3 = {**O.SHORT, "G1-small20": "groups 80/20", "G2-small15": "groups 85/15", "G3-cluster": "clustered outliers", "E4-K4par": "four groups"}
KS = pd.concat([pd.read_csv(os.path.join(ROOT, f, "ksel.csv")) for f in ["results_v4", "results_v4e", "results_v4c"]])
L = ["\\begin{table}[htbp]", "\\centering", "\\small",
     "\\caption{Choice of $K$ by the Bayesian information criterion of the Gaussian-lines-plus-noise model at the fit of the procedure, among $K_0-1$, $K_0$ and $K_0+1$ ($K_0$ the true number): fraction of data sets for which $K_0$ is chosen ($20$ data sets per cell; $10$ for four groups; descriptive).}",
     "\\label{tab:S18}", "\\begin{tabular}{lcccc}", "\\toprule", "Design & $K_0$ & $\\varepsilon=0$ & $0.10$ & $0.20$ \\\\", "\\midrule"]
T = KS.assign(ok=KS.Khat == KS.K).groupby(["design", "eps"]).ok.mean().unstack()
for dsg, r in T.iterrows():
    L.append(f"{S3.get(dsg, dsg)} & {int(KS[KS.design == dsg].K.iloc[0])} & " + " & ".join(f"{r[e]:.2f}" for e in [0.0, 0.1, 0.2]) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(TAB, "supp_S18.tex"), "w").write("\n".join(L))
