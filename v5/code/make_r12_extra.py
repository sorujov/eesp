"""Round 12 (descriptive): the eleven deciding cells with 200 data sets each (replicates 0-49 of the
study and 150 further ones from r12_extra.py): mean accuracy on clean units with its standard error,
and the paired difference ESF minus each TCLUST-REG pipeline with its standard error.
Writes paper/tables/supp_extra.tex and results_r12/extra_cells.csv."""
import glob, os
import numpy as np, pandas as pd
import r12_extra as RX
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FOLD = {"1": "results_v4", "1c": "results_v4c", "1e": "results_v4e", "1g": "results_v4g"}
METH = ["Alg2", "TCRlong(0.25)+rw+A2", "TCRlong(0.40)+rw+A2", "TCRlong(0.30)+rw+A2", "TCReqwL(0.25)+rw+A2"]
LAB = {"Alg2": "ESF", "TCRlong(0.25)+rw+A2": "$0.25$", "TCRlong(0.40)+rw+A2": "$0.40$",
       "TCRlong(0.30)+rw+A2": "$0.30$", "TCReqwL(0.25)+rw+A2": "$0.25$, eq.\\ w."}
NAME = {"D2-K3par": "D2", "E4-K4par": "$K=4$", "G1-small20": "80/20", "G3-cluster": "clustered",
        "G4-small20-p3": "80/20, $p=3$", "G5-small15-par": "85/15, parallel"}
parts = []
for st, f in FOLD.items():
    a = pd.read_csv(os.path.join(ROOT, f, "alg2.csv")); a["study"] = st
    parts.append(a[["study", "design", "eps", "rep", "method", "acc"]])
T = pd.read_csv(os.path.join(ROOT, "results_tcrlong", "tcrlong_0.csv")); T["study"] = T.study.astype(str)
parts.append(T[T.method.isin(METH)][["study", "design", "eps", "rep", "method", "acc"]])
for f in glob.glob(os.path.join(ROOT, "results_r10", "fitspost_*.csv")):
    R = pd.read_csv(f); R["study"] = R.study.astype(str)
    parts.append(R[R.method.isin(METH)][["study", "design", "eps", "rep", "method", "acc"]])
for f in glob.glob(os.path.join(ROOT, "results_r12", "extra", "*.csv")):
    R = pd.read_csv(f); R["study"] = R.study.astype(str)
    parts.append(R[R.method.isin(METH)][["study", "design", "eps", "rep", "method", "acc"]])
A = pd.concat(parts, ignore_index=True); A["eps"] = A.eps.round(2); A.loc[A.acc.isna(), "acc"] = 0.0
keep = pd.Series(False, index=A.index)
for st, cells in RX.CELLS.items():
    for d, e in cells:
        keep |= (A.study == st) & (A.design == d) & (A.eps == round(e, 2))
A = A[keep].drop_duplicates(["study", "design", "eps", "rep", "method"])
W = A.pivot_table(index=["study", "design", "eps", "rep"], columns="method", values="acc")
rows, L = [], ["\\begin{tabular}{llcccccccc}", "\\toprule",
               "Design & $\\varepsilon$ & $n_{\\mathrm{rep}}$ & ESF & \\multicolumn{2}{c}{$0.25$} & \\multicolumn{2}{c}{$0.30$} & \\multicolumn{2}{c}{$0.25$, equal weights} \\\\",
               "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
               " & & & mean (s.e.) & mean & ESF $-$ (s.e.) & mean & ESF $-$ (s.e.) & mean & ESF $-$ (s.e.) \\\\", "\\midrule"]
for (st, d, e), g in W.groupby(level=[0, 1, 2]):
    g = g.dropna(subset=METH)
    n = len(g)
    r = dict(study=st, design=d, eps=e, n=n)
    for m in METH:
        r[m] = g[m].mean(); r[m + "_se"] = g[m].std(ddof=1) / np.sqrt(n)
        if m != "Alg2":
            dd = g["Alg2"] - g[m]; r["d_" + m] = dd.mean(); r["dse_" + m] = dd.std(ddof=1) / np.sqrt(n)
    rows.append(r)
    cell = lambda m: f"{r[m]:.3f} & {r['d_' + m]:+.3f} ({r['dse_' + m]:.3f})"
    L.append(f"{NAME[d]} & {e:.2f} & {n} & {r['Alg2']:.3f} ({r['Alg2_se']:.3f}) & {cell('TCRlong(0.25)+rw+A2')} & {cell('TCRlong(0.30)+rw+A2')} & {cell('TCReqwL(0.25)+rw+A2')} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
os.makedirs(os.path.join(ROOT, "results_r12"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results_r12", "extra_cells.csv"), index=False)
open(os.path.join(ROOT, "paper", "tables", "supp_extra.tex"), "w").write("\n".join(L))
print("\n".join(L))
print(pd.DataFrame(rows)[["design", "eps", "n", "Alg2", "TCRlong(0.40)+rw+A2", "d_TCRlong(0.40)+rw+A2", "dse_TCRlong(0.40)+rw+A2"]].round(3).to_string())
