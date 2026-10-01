"""Sensitivity of Algorithm 1 to the cut-off c and the covariate-screen quantile (Gate 1 data)."""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, py_methods as P, evaluate as EV

VARIANTS = [("c=2.0", dict(c=2.0)), ("c=2.5", dict(c=2.5)), ("c=3.0", dict(c=3.0)),
            ("q=0.95", dict(xq=0.95)), ("q=0.99", dict(xq=0.99))]
DES = ["D1-K2cross", "D3-unequal", "D5-leverage", "D6-uniform"]
EPSS = [0.10, 0.20, 0.30]


def run(u):
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    th0 = EV.true_theta(D)
    rows = []
    for name, kw in VARIANTS:
        rng = np.random.default_rng(simv4.seed_for("v4sens", *u))
        th, ah, t = P.eesp_adaptive_x(X, y, D["K"], D["m"], 100, rng, xtrim=True, **kw)
        rows.append(dict(design=u[0], eps=u[1], rep=u[2], variant=name,
                         acc=EV.acc_clean(X, y, th, z0, ~out), ahat=ah))
    return rows


if __name__ == "__main__":
    U = [u for u in simv4.units() if u[0] in DES and u[1] in EPSS]
    with Pool(int(os.environ.get("WORKERS", "2"))) as p:
        R = p.map(run, U, chunksize=4)
    df = pd.DataFrame([r for rr in R for r in rr])
    df.to_csv(sys.argv[1], index=False)
    print(df.groupby(["design", "eps", "variant"]).acc.mean().unstack().round(3))
