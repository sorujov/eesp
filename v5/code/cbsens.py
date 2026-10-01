"""Sensitivity of Algorithm 1 to the bound c_b of the scoring loss (descriptive, after Gate 1e).
Usage: V4_STUDY=... python cbsens.py OUT.csv DESIGN1,DESIGN2 EPS1,EPS2"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, py_methods as P, evaluate as EV

CB = [3.0, 4.0, 5.0]


def run(u):
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    rows = []
    for cb in CB:
        rng = np.random.default_rng(simv4.seed_for("v4cbsens", *u))
        th, ah, t = P.eesp_adaptive_x(X, y, D["K"], D["m"], 100, rng, xtrim=True, c_bound=cb)
        rows.append(dict(design=u[0], eps=u[1], rep=u[2], cb=cb,
                         acc=EV.acc_clean(X, y, th, z0, ~out), ahat=ah))
    return rows


if __name__ == "__main__":
    des = sys.argv[2].split(","); eps = [float(e) for e in sys.argv[3].split(",")]
    U = [u for u in simv4.units() if u[0] in des and any(abs(u[1] - e) < 1e-9 for e in eps) and u[2] < 50]
    with Pool(int(os.environ.get("WORKERS", "2"))) as p:
        R = p.map(run, U, chunksize=4)
    df = pd.DataFrame([r for rr in R for r in rr])
    df.to_csv(sys.argv[1], index=False)
    print(df.groupby(["design", "eps", "cb"]).acc.mean().unstack().round(3))
