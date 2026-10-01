"""Gate 2 evaluation (criteria in PREREG_GATE2.md)."""
import os
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
parts = {"1": "results_v4", "1b": "results_v4b", "1c": "results_v4c", "1d": "results_v4d", "1e": "results_v4e"}
rows = []
M = {}
for p, f in parts.items():
    m = pd.read_csv(R + f + "/metrics.csv"); a2 = pd.read_csv(R + f + "/alg2.csv")
    m.loc[m.acc.isna(), "acc"] = 0; a2.loc[a2.acc.isna(), "acc"] = 0
    M[p] = (m, a2)
    a1 = m[m.method == "adaptX"].set_index("unit").acc
    b = a2.set_index("unit")
    b["a1"] = a1
    b["d"] = b.acc - b.a1
    for (des, e), g in b.groupby(["design", "eps"]):
        rows.append(dict(part=p, design=des, eps=e, n=len(g), alg1=g.a1.mean(), alg2=g.acc.mean(),
                         diff=g.d.mean(), se=g.d.std(ddof=1) / np.sqrt(len(g)), changed=(g.steps > 0).mean()))
T = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(T.round(3).to_string())
# (a)
A = T[T.design.isin(["G1-small20", "G2-small15"]) & (T.eps <= 0.2 + 1e-9)]
print("\n(a) recovery cells:\n", A[["design", "eps", "alg1", "alg2", "diff", "se"]].round(3).to_string())
pa = bool((A["diff"] >= 0.05).all())
# (b)
B = T[T.eps <= 0.2 + 1e-9]
worst = B.sort_values("diff").head(5)
pb = bool((B["diff"] >= -0.01).all())
print("\n(b) worst cells eps<=0.2:\n", worst[["part", "design", "eps", "diff", "se"]].round(4).to_string())
# (c) Gate 1 comparison with Alg2 in place of adaptX, competitors of Gate 1 (all methods in metrics.csv except adapt, adaptX)
m, a2 = M["1"]
comp = [x for x in m.method.unique() if x not in ("adapt", "adaptX")]
res = []
for des in ["D1-K2cross", "D2-K3par", "D3-unequal", "D4-hetero", "D6-uniform", "D7-p3", "D8-K2par-sep"]:
    for e in [0.0, 0.05, 0.1, 0.2]:
        g = m[(m.design == des) & (m.eps == e)]
        mm = g.groupby("method").acc.mean(); mm = mm[[c for c in comp if c in mm.index]]
        best = mm.idxmax()
        A2 = a2[(a2.design == des) & (a2.eps == e)].set_index("rep").acc
        Bc = g[g.method == best].set_index("rep").acc
        dd = (A2 - Bc).dropna()
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        res.append(dict(design=des, eps=e, best=best, alg2=A2.mean(), bestacc=mm.max(), diff=dd.mean(), se=se,
                        fail=bool(dd.mean() < -0.03 or dd.mean() + 1.96 * se < -0.02)))
C = pd.DataFrame(res)
print("\n(c) Gate 1 comparison with Algorithm 2:\n", C.sort_values("diff").head(8).round(3).to_string())
pc = not C.fail.any()
print("\nRESULT: (a)", "PASS" if pa else "FAIL", " (b)", "PASS" if pb else "FAIL", " (c)", "PASS" if pc else "FAIL")
print("cells < 0.005 shortfall in (c):", int((C["diff"] > -0.005).sum()), "of", len(C))
T.to_csv(R + "results_v4/gate2_cells.csv", index=False); C.to_csv(R + "results_v4/gate2_c.csv", index=False)
