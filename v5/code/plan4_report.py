"""Plan 4 report (fresh-data confirmation of the recovery step): criteria (a)-(c) of
PREREG_PLAN4.md and the table tables/plan4.tex."""
import os
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
F = os.path.join(ROOT, "results_v4g")
M = pd.concat([pd.read_csv(os.path.join(F, f)) for f in ["metrics.csv", "alg2.csv", "laplace.csv", "cnmix.csv", "noisemix.csv"]
               if os.path.exists(os.path.join(F, f))])
M.loc[M.acc.isna(), "acc"] = 0.0
NAME = {"G1f-small20": "groups 80/20", "G2f-small15": "groups 85/15", "G4-small20-p3": "80/20, three covariates",
        "G5-small15-par": "85/15, parallel lines"}
res = []
for (des, e), g in M.groupby(["design", "eps"]):
    P = g.pivot_table(index="rep", columns="method", values="acc")
    comp = [c for c in P.columns if c not in ("Alg2", "adaptX", "adapt")]
    best = P[comp].mean().idxmax()
    d1 = P["Alg2"] - P["adaptX"]; d2 = P["Alg2"] - P[best]
    n = len(d1)
    res.append(dict(design=des, eps=e, alg1=P["adaptX"].mean(), proc=P["Alg2"].mean(), gain=d1.mean(),
                    gain_se=d1.std(ddof=1) / np.sqrt(n), best=best, bestacc=P[best].mean(), diff=d2.mean(),
                    se=d2.std(ddof=1) / np.sqrt(n)))
R = pd.DataFrame(res)
R["a"] = np.where(R.design.isin(["G1f-small20", "G2f-small15"]), R.gain >= 0.05, True)
R["b"] = R.gain >= -0.01
R["c"] = (R["diff"] >= -0.03) & (R["diff"] + 1.96 * R.se >= -0.02)
print(R.round(3).to_string())
print("(a)", "PASS" if R.a.all() else "FAIL", " (b)", "PASS" if R.b.all() else "FAIL", " (c)", "PASS" if R.c.all() else "FAIL")
R.to_csv(os.path.join(F, "plan4_cells.csv"), index=False)
L = ["\\begin{tabular}{llccccc}", "\\toprule",
     "Design & $\\varepsilon$ & Algorithm~1 & Procedure & Gain (s.e.) & Best competitor & Difference (s.e.) \\\\", "\\midrule"]
for r in R.itertuples():
    lab = r.best.replace("TCR", "TCLUST-REG ").replace("+rw", ", rw").replace("tcwm", "trimmed CWM ").replace("bisq", "bisquare").replace("CNmix", "contam. normal").replace("NoiseMix", "noise comp.")
    L.append(f"{NAME[r.design]} & {r.eps:.2f} & {r.alg1:.3f} & {r.proc:.3f} & {r.gain:+.3f} ({r.gain_se:.3f}) & {lab}: {r.bestacc:.3f} & {r.diff:+.3f} ({r.se:.3f}) \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(ROOT, "paper", "tables", "plan4.tex"), "w").write("\n".join(L))
