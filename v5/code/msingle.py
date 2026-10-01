"""Algorithm 1 with one subsample size for every design (m = 12), after the pre-registered study.
Usage: V4_STUDY=... python msingle.py OUT.csv [EPS_MAX]"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, py_methods as P, evaluate as EV


def run(u):
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    rng = np.random.default_rng(simv4.seed_for("v4msingle", *u))
    th, ah, t = P.eesp_adaptive_x(X, y, D["K"], 12, 100, rng, xtrim=True)
    return dict(design=u[0], eps=u[1], rep=u[2], m=12, acc=EV.acc_clean(X, y, th, z0, ~out), ahat=ah, time=t)


if __name__ == "__main__":
    emax = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    U = [u for u in simv4.units() if u[1] <= emax + 1e-9]
    with Pool(int(os.environ.get("WORKERS", "2"))) as p:
        R = p.map(run, U, chunksize=4)
    df = pd.DataFrame(R); df.to_csv(sys.argv[1], index=False)
    print(df.groupby(["design", "eps"]).acc.mean().unstack().round(3))
