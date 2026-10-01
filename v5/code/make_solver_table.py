"""Table of the solver benchmark (solver_bench2.py, results_v4/solver_bench2.csv)."""
import os
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
d = pd.read_csv(os.path.join(ROOT, "results_v4", "solver_bench2.csv")).drop_duplicates(subset=["design", "m", "solver", "rep"])
NAME = {"D1-K2cross": "D1", "D2-K3par": "D2", "D7-p3": "D7", "E4-K4par": "four lines"}
order = ["D1-K2cross", "D2-K3par", "D7-p3", "E4-K4par"]
def ms(v):
    v = 1000 * v
    s = f"{v:.2f}" if v < 10 else (f"{v:.1f}" if v < 100 else f"{v:,.0f}")
    return s.replace(",", "\\,")
L = ["\\begin{tabular}{lccrrrrrr}", "\\toprule",
     " & & & \\multicolumn{3}{c}{Branch and bound} & \\multicolumn{2}{c}{MIQP} & \\\\",
     "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}",
     "Design & $K$ & $m$ & ours & with incumbent & repetitive & Gurobi & SCIP & Nodes, ours \\\\", "\\midrule"]
for des in order:
    for m in sorted(d[d.design == des].m.unique()):
        g = d[(d.design == des) & (d.m == m)]
        med = g.groupby("solver").time.median()
        row = [NAME[des], str(int(g.K.iloc[0])), str(m)]
        for s in ["bb", "bb_inc", "rbba", "miqp_g", "miqp_s"]:
            tl = int((~g[g.solver == s].optimal.astype(bool)).sum())
            row.append(ms(med[s]) + (f"$^{{{tl}}}$" if tl else ""))
        row.append(f"{g[g.solver == 'bb'].nodes.median():,.0f}".replace(",", "\\,"))
        L.append(" & ".join(row) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(ROOT, "paper", "tables", "solvers.tex"), "w").write("\n".join(L))
sm = d.groupby(["design", "m", "solver"]).time.median().unstack()
for c in ["miqp_g", "miqp_s", "rbba", "bb_inc"]:
    sm[c + "/bb"] = sm[c] / sm.bb
print(sm[[c for c in sm.columns if "/" in c]].round(1))
ok = d.groupby(["design", "m", "rep"]).apply(lambda g: g[g.solver.isin(["bb", "miqp_g"]) & g.optimal.astype(bool)].value.agg(lambda v: v.max() - v.min()))
print("max |bb - gurobi| when both optimal:", ok.max())
sc = d[(d.solver == "miqp_s") & d.optimal.astype(bool) & (d.gap > 1e-6)]
print("SCIP optimal but above the others (coefficient bound):", len(sc), "of", ((d.solver == "miqp_s") & d.optimal.astype(bool)).sum())
print("timeouts:", d[~d.optimal.astype(bool)].groupby(["design", "m", "solver"]).size())
