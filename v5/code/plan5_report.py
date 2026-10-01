"""Plan 5 report: criterion (c) of PREREG_PLAN5.md per cell, the flagged fraction split into
residual and covariate flags, and tables/plan5.tex."""
import os
import numpy as np, pandas as pd
os.environ["V4_STUDY"] = "1h"
import simv4, py_methods as P
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
F = os.path.join(ROOT, "results_v4h")
M = pd.concat([pd.read_csv(os.path.join(F, f)) for f in ["metrics.csv", "alg2.csv", "laplace.csv", "cnmix.csv", "noisemix.csv"]])
M.loc[M.acc.isna(), "acc"] = 0.0
A2 = pd.read_csv(os.path.join(F, "alg2.csv")).set_index("unit")
fl = []
for u in simv4.units():
    X, y, z0, out = simv4.generate(u)
    K, d = 2, X.shape[1]
    th = A2.loc[simv4.unit_name(u), [f"t{j+1}" for j in range(K * d)]].values.astype(float).reshape(K, d)
    f, _, _ = P._flag_all(X, y, th, 2.5, True, 0.975); fr, _, _ = P._flag_all(X, y, th, 2.5, False, 0.975)
    fl.append(dict(design=u[0], eps=u[1], rep=u[2], resid=fr.mean(), cov=(f & ~fr).mean(), recall=f[out].mean() if out.any() else np.nan))
FL = pd.DataFrame(fl).groupby(["design", "eps"]).mean()
NAME = {"S1-skew": "skewed errors", "S2-p3-unequal": "70/30, three covariates", "S3-xhet": "scale growing with $|x|$"}
res = []
for (des, e), g in M.groupby(["design", "eps"]):
    Pv = g.pivot_table(index="rep", columns="method", values="acc")
    comp = [c for c in Pv.columns if c not in ("Alg2", "adaptX", "adapt")]
    best = Pv[comp].mean().idxmax()
    d2 = Pv["Alg2"] - Pv[best]; n = len(d2)
    res.append(dict(design=des, eps=e, alg1=Pv["adaptX"].mean(), proc=Pv["Alg2"].mean(), best=best, bestacc=Pv[best].mean(),
                    diff=d2.mean(), se=d2.std(ddof=1) / np.sqrt(n), resid=FL.loc[(des, e), "resid"], cov=FL.loc[(des, e), "cov"]))
R = pd.DataFrame(res)
R["c"] = (R["diff"] >= -0.03) & (R["diff"] + 1.96 * R.se >= -0.02)
print(R.round(3).to_string()); print("(c)", "PASS" if R.c.all() else "FAIL", int((~R.c).sum()), "failing cells")
R.to_csv(os.path.join(F, "plan5_cells.csv"), index=False)
L = ["\\begin{tabular}{llcccccc}", "\\toprule",
     "Design & $\\varepsilon$ & Algorithm~1 & Procedure & Best competitor & Difference (s.e.) & Flagged, residual & Flagged, covariates \\\\", "\\midrule"]
for r in R.itertuples():
    lab = r.best.replace("TCR", "TCLUST-REG ").replace("+rw", ", rw").replace("tcwm", "trimmed CWM ").replace("bisq", "bisquare").replace("CNmix", "contam. normal").replace("NoiseMix", "noise comp.")
    L.append(f"{NAME[r.design]} & {r.eps:.2f} & {r.alg1:.3f} & {r.proc:.3f} & {lab}: {r.bestacc:.3f} & {r.diff:+.3f} ({r.se:.3f}) & {r.resid:.3f} & {r.cov:.3f} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(ROOT, "paper", "tables", "plan5.tex"), "w").write("\n".join(L))
