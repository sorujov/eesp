"""The procedure with B = 500 replicates instead of 100 (descriptive): Algorithm 1 in 10 stages
of 50, then Algorithm 2.  Usage: V4_STUDY=... python bvar.py OUT  (WORKERS processes;
UNIT_START/UNIT_END select a range)"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, evaluate as EV, recover2 as R


def run(u):
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    th2, a2, info, th, a = R.algorithm1_r(X, y, D["K"], D["m"], 500, np.random.default_rng(simv4.seed_for("v4B500", *u)))
    return dict(design=u[0], eps=u[1], rep=u[2], unit=simv4.unit_name(u), acc_alg1=EV.acc_clean(X, y, th, z0, ~out),
                acc=EV.acc_clean(X, y, th2, z0, ~out), ahat=a2)


if __name__ == "__main__":
    U = simv4.units()
    a = int(os.environ.get("UNIT_START", 0)); b = int(os.environ.get("UNIT_END", len(U)))
    with Pool(int(os.environ.get("WORKERS", "2"))) as p:
        rows = p.map(run, U[a:b], chunksize=1)
    pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
    print("UNITS_DONE=%d" % len(rows))
