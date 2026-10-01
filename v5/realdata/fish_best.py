"""Fishery: run the procedure with several seeds and keep the fit of highest likelihood (the model of
Algorithm 2); repeated 20 times with disjoint seed sets of size 5."""
import sys; sys.path.insert(0, "../code")
import numpy as np, pandas as pd, recover2 as R, py_methods as P
D = pd.read_csv("fishery_log.csv"); X = np.column_stack([np.ones(len(D)), D.x1.values]); y = D.y.values
n = len(y); xf = P._xflag(X, np.zeros(n, dtype=int), 1); Rg = 1.1 * np.ptp(y)
rows = []
for K, m in [(2, 8), (3, 16)]:
    fits = []
    for s in range(100):
        th2, a2, info, th, a = R.algorithm1_r(X, y, K, m, 100, np.random.default_rng(50000 + s))
        ll, fl, sc = R.loglik(X, y, th2, Rg, excl=xf)
        o = np.argsort(th2[:, 0]); fits.append((ll, tuple(np.round(th2[o, 0], 2)), a2))
    for g in range(20):
        best = max(fits[5 * g:5 * g + 5], key=lambda t: t[0])
        rows.append(dict(K=K, group=g, ll=best[0], b0=best[1], flagged=best[2]))
    bic = [-2 * f[0] + (K * 3 + K) * np.log(n) for f in fits]
    print(K, pd.Series([f[1] for f in fits]).value_counts().head(4).to_dict(), "best-of-5:",
          pd.Series([r["b0"] for r in rows if r["K"] == K]).value_counts().to_dict(), "min BIC", round(min(bic), 1), flush=True)
pd.DataFrame(rows).to_csv("fish_best.csv", index=False)
