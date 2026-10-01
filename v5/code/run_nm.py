"""Noise-component mixture of regressions (py_methods.noise_mix) on every data set of a study.
Usage: V4_STUDY=... python run_nm.py OUT.csv   -> rows in the metrics format of evaluate.py"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, py_methods as P, evaluate as EV
STARTS = int(os.environ.get("NM_STARTS", "10")); INIT = os.environ.get("NM_INIT", "partition")
NAME = "NoiseMix" if STARTS == 10 else f"NoiseMix{STARTS}"


def run(u):
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    rng = np.random.default_rng(simv4.seed_for("v4nm" if STARTS == 10 else "v4nm100", *u))
    th, ah, t = P.noise_mix(X, y, D["K"], rng, starts=STARTS, init=INIT)
    th0 = EV.true_theta(D)
    ok = np.isfinite(th).all()
    a = EV.acc_clean(X, y, th, z0, ~out) if ok else np.nan
    return dict(design=u[0], eps=u[1], rep=u[2], unit=simv4.unit_name(u), method=NAME,
                acc=a, perr=EV.align_err(th, th0) if ok else np.nan,
                fail=bool(not np.isfinite(a) or a < 0.7), ahat=ah, time=t)


if __name__ == "__main__":
    with Pool(2) as p:
        R = p.map(run, simv4.units(), chunksize=20)
    pd.DataFrame(R).to_csv(sys.argv[1], index=False)
    print(len(R))
