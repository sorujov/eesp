"""Sensitivity of Algorithm 1 to the subsample size m, and its seed-to-seed variability
(part-1 data, Python only)."""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, py_methods as P, evaluate as EV

GRID = {"D1-K2cross": [6, 8, 10, 12, 16], "D2-K3par": [9, 12, 16, 20],
        "D3-unequal": [6, 8, 10, 12, 16], "D6-uniform": [6, 8, 10, 12, 16]}
EPSS = [0.0, 0.10, 0.20]


def run(u):
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    rows = []
    for m in GRID[u[0]]:
        rng = np.random.default_rng(simv4.seed_for("v4msens", m, *u))
        th, ah, t = P.eesp_adaptive_x(X, y, D["K"], m, 100, rng, xtrim=True)
        rows.append(dict(design=u[0], eps=u[1], rep=u[2], kind="m", m=m, seed=0,
                         acc=EV.acc_clean(X, y, th, z0, ~out), ahat=ah, time=t))
    if u[0] in ("D1-K2cross", "D3-unequal") and u[1] == 0.20:
        for s in range(1, 6):
            rng = np.random.default_rng(simv4.seed_for("v4seed", s, *u))
            th, ah, t = P.eesp_adaptive_x(X, y, D["K"], D["m"], 100, rng, xtrim=True)
            rows.append(dict(design=u[0], eps=u[1], rep=u[2], kind="seed", m=D["m"], seed=s,
                             acc=EV.acc_clean(X, y, th, z0, ~out), ahat=ah, time=t))
    return rows


if __name__ == "__main__":
    U = [u for u in simv4.units() if u[0] in GRID and u[1] in EPSS]
    with Pool(2) as p:
        R = p.map(run, U, chunksize=2)
    df = pd.DataFrame([r for rr in R for r in rr])
    df.to_csv(sys.argv[1], index=False)
    print(df[df.kind == "m"].groupby(["design", "eps", "m"]).acc.mean().unstack().round(3))
    s = df[df.kind == "seed"]
    print(s.groupby(["design", "seed"]).acc.mean().round(3))
    print("within-data sd of acc across seeds:", s.groupby(["design", "rep"]).acc.std().groupby("design").mean().round(4))
