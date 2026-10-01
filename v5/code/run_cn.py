"""Noise-component mixture of regressions (py_methods.noise_mix) on every data set of a study.
Usage: V4_STUDY=... python run_nm.py OUT.csv   -> rows in the metrics format of evaluate.py"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, cn_mix as C, evaluate as EV
STARTS = int(os.environ.get("CN_STARTS", "50")); INIT = os.environ.get("CN_INIT", "elemental")
ONLY = os.environ.get("CN_DESIGNS", "")
NAME = "CNmix" if STARTS == 50 else f"CNmix{STARTS}"


def run(u):
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    rng = np.random.default_rng(simv4.seed_for("v4cn" if STARTS == 50 else "v4cn100", *u))
    th, ah, t = C.cn_mix(X, y, D["K"], rng, starts=STARTS, init=INIT)
    th0 = EV.true_theta(D)
    ok = np.isfinite(th).all()
    a = EV.acc_clean(X, y, th, z0, ~out) if ok else np.nan
    return dict(design=u[0], eps=u[1], rep=u[2], unit=simv4.unit_name(u), method=NAME,
                acc=a, perr=EV.align_err(th, th0) if ok else np.nan,
                fail=bool(not np.isfinite(a) or a < 0.7), ahat=ah, time=t)


if __name__ == "__main__":
    with Pool(2) as p:
        R = p.map(run, [u for u in simv4.units() if not ONLY or u[0].startswith(tuple(ONLY.split(",")))], chunksize=10)
    pd.DataFrame(R).to_csv(sys.argv[1], index=False)
    print(len(R))
