"""Best-of-five rule in simulation (descriptive, added after the review of 27 Sep 2026).
The procedure (Algorithms 1 and 2, recover2.algorithm1_r) is run with five fresh seeds on every
data set of a study; the fit of highest likelihood under the Gaussian-lines-plus-uniform model of
Algorithm 2 is kept.  Reported: accuracy of each run and of the selected fit.
Usage: V4_STUDY=... python best5.py OUT.csv   (WORKERS processes)"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, evaluate as EV, recover2 as R, py_methods as P


def run(u):
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    K, n = D["K"], len(y)
    xf = P._xflag(X, np.zeros(n, dtype=int), 1)
    Rr = 1.1 * float(np.ptp(y))
    accs, lls = [], []
    for j in range(5):
        th2, a2, info, th, a = R.algorithm1_r(X, y, K, D["m"], 100,
                                              np.random.default_rng(simv4.seed_for("v4best5", j, *u)))
        ok = np.isfinite(th2).all()
        accs.append(EV.acc_clean(X, y, th2, z0, ~out) if ok else 0.0)
        lls.append(R.loglik(X, y, th2, Rr, excl=xf)[0] if ok else -np.inf)
    b = int(np.argmax(lls))
    return dict(design=u[0], eps=u[1], rep=u[2], unit=simv4.unit_name(u),
                **{f"acc{j}": accs[j] for j in range(5)}, acc_mean=float(np.mean(accs)),
                acc_best5=accs[b], acc_min=min(accs), acc_max=max(accs))


if __name__ == "__main__":
    W = int(os.environ.get("WORKERS", "2"))
    with Pool(W) as p:
        rows = p.map(run, simv4.units(), chunksize=4)
    pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
    print("UNITS_DONE=%d" % len(rows))
