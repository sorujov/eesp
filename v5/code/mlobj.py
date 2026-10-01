"""Common criterion for the MATLAB and Octave TCLUST-REG fits: from the coefficients, concentration
steps (assign by weighted density, trim the fraction alpha with the lowest
density, re-estimate group scales and weights, variance ratio capped at 12) and the trimmed
classification log-likelihood.  Descriptive."""
import itertools, os, importlib
import numpy as np, pandas as pd
from scipy.stats import norm
M = pd.read_csv("../results_matlab/results_01_80.csv")
import tcrlong_post as TP
O = TP.O
def crit(X, y, B, al, c=12.0):
    n = len(y); K = B.shape[0]; R = y[:, None] - X @ B.T
    s = np.full(K, np.median(np.abs(R).min(1)) / 0.6745 + 1e-9); p = np.full(K, 1 / K)
    for _ in range(20):
        L = np.log(p) + norm.logpdf(R / s) - np.log(s)
        z = L.argmax(1); lm = L.max(1); keep = lm >= np.quantile(lm, al)
        for k in range(K):
            m = keep & (z == k)
            p[k] = max(m.mean() / keep.mean(), 1e-6)
            if m.sum() > 1: s[k] = np.sqrt(np.mean(R[m, k] ** 2))
        v = s ** 2; v = np.clip(v, v.max() / c, None); s = np.sqrt(v)
    L = np.log(p) + norm.logpdf(R / s) - np.log(s)
    lm = L.max(1); keep = lm >= np.quantile(lm, al)
    return lm[keep].sum()
rows = []
for st in ["1", "1b", "1c"]:
    os.environ["V4_STUDY"] = st
    import simv4; importlib.reload(simv4)
    U = {simv4.unit_name(u): u for u in simv4.units()}
    for _, m in M[M.unit.str.startswith("s" + st + "/")].iterrows():
        nm = m.unit.split("/")[1]; u = U[nm]; X, y, z0, out = simv4.generate(u)
        K, d = simv4.DESIGNS[u[0]]["K"], X.shape[1]; al = float(m.method[8:12])
        o = O[(O.unit == nm) & (O.study == st) & (O.method == m.method)].iloc[0]
        bm = np.array([m[f"b{j+1}"] for j in range(K*d)]).reshape(K, d)
        bo = np.array([o[f"b{j+1}"] for j in range(K*d)], float).reshape(K, d)
        rows.append(dict(study=st, design=u[0], eps=u[1], rep=u[2], method=m.method, obj_ml=m.obj,
                         c_ml=crit(X, y, bm, al), c_oct=crit(X, y, bo, al)))
D = pd.DataFrame(rows); D.to_csv("../results_matlab/crit.csv", index=False)
d = D.c_ml - D.c_oct
print("corr(obj_ml, c_ml)", np.corrcoef(D.obj_ml, D.c_ml)[0, 1].round(4))
print("matlab higher >0.5:", int((d > 0.5).sum()), " octave higher >0.5:", int((d < -0.5).sum()), " within 0.5:", int((d.abs() <= 0.5).sum()))
print(D.assign(d=d).groupby(["study", "design", "eps", "method"]).d.agg(["mean", lambda x: (x > .5).sum(), lambda x: (x < -.5).sum()]).round(2).to_string())
r = D.obj_ml - D.c_ml
print("score vs MATLAB objective: within 0.5:", int((r.abs() <= 0.5).sum()), "of", len(r), " max |diff|", round(r.abs().max(), 1))
for mg in (2, 5):
    print("margin", mg, "matlab higher", int((d > mg).sum()), "octave higher", int((d < -mg).sum()))
