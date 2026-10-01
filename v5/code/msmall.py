"""The procedure with one subsample size m=12 in the small-group designs (descriptive):
Algorithm 1 (seed as in the study) and Algorithm 2.  Usage: V4_STUDY=... python msmall.py OUT"""
import sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, evaluate as EV, recover2 as R

KEEP = ("G1", "G2", "G4", "G5")


def run(u):
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    th2, a2, info, th, a = R.algorithm1_r(X, y, D["K"], 12, 100, np.random.default_rng(simv4.seed_for("v4m12", *u)))
    return dict(design=u[0], eps=u[1], rep=u[2], unit=simv4.unit_name(u), acc_alg1=EV.acc_clean(X, y, th, z0, ~out),
                acc=EV.acc_clean(X, y, th2, z0, ~out), ahat=a2)


if __name__ == "__main__":
    us = [u for u in simv4.units() if u[0].startswith(KEEP)]
    with Pool(2) as p:
        rows = p.map(run, us, chunksize=4)
    pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
    print(len(rows))
