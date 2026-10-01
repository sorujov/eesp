"""Choice of K by BIC of the Gaussian-lines-plus-uniform model at the Algorithm 2 fit (dev seeds)."""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, recover2 as R, py_methods as P
if os.environ.get("DEV_SEED0", "study") != "study":
    simv4.SEED0 = int(os.environ["DEV_SEED0"])
MFOR = {1: 4, 2: 8, 3: 12, 4: 16, 5: 20}
def run(u):
    X, y, z0, out = simv4.generate(u); D = simv4.DESIGNS[u[0]]; n, d = X.shape
    xf = P._xflag(X, np.zeros(n, dtype=int), 1)
    res = {}
    for K in range(max(1, D["K"] - 1), D["K"] + 2):
        m = max(D["m"], MFOR[K]) if K >= D["K"] else MFOR[K]
        m = max(m, (d + 1) * K + 2)
        th2, a2, info, th, a = R.algorithm1_r(X, y, K, m, 100, np.random.default_rng(simv4.seed_for("ksel", K, *u)))
        ll, fl, s = R.loglik(X, y, th2, 1.1 * np.ptp(y), excl=xf)
        res[K] = -2 * ll + (K * (d + 1) + K) * np.log(n)
    return dict(design=u[0], eps=u[1], rep=u[2], K=D["K"], Khat=min(res, key=res.get))
if __name__ == "__main__":
    des = sys.argv[2].split(","); eps = [float(e) for e in sys.argv[3].split(",")]; reps = int(sys.argv[4])
    U = [(dd, e, r) for dd in des for e in eps for r in range(reps)]
    with Pool(2) as p: rows = p.map(run, U)
    df = pd.DataFrame(rows); df.to_csv(sys.argv[1], index=False)
    print(df.groupby(["design", "eps"]).apply(lambda g: (g.Khat == g.K).mean()).round(2))
    print(df.groupby(["design", "eps"]).Khat.value_counts().unstack(fill_value=0))
