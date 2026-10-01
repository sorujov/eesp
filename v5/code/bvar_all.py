"""B = 500 variant over the units of several studies in one list (interleaved so that any range
mixes cheap and costly units).  Env: BV_STUDIES (comma list), UNIT_START, UNIT_END, WORKERS,
RESULTS_DIR, CHUNK_ID.  Writes one row per unit as it finishes."""
import os, sys, importlib
from multiprocessing import Pool
import numpy as np, pandas as pd

STUDIES = os.environ.get("BV_STUDIES", "1,1b,1e,1g").split(",")


def unit_list():
    L = []
    for st in STUDIES:
        os.environ["V4_STUDY"] = st
        import simv4
        importlib.reload(simv4)
        L.append([(st, u) for u in simv4.units()])
    out = []
    for i in range(max(len(l) for l in L)):
        for l in L:
            if i < len(l):
                out.append(l[i])
    return out


def run(item):
    st, u = item
    os.environ["V4_STUDY"] = st
    import simv4, evaluate as EV, recover2 as R
    importlib.reload(simv4)
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    th2, a2, info, th, a = R.algorithm1_r(X, y, D["K"], D["m"], 500, np.random.default_rng(simv4.seed_for("v4B500", *u)))
    return dict(study=st, design=u[0], eps=u[1], rep=u[2], unit=simv4.unit_name(u),
                acc_alg1=EV.acc_clean(X, y, th, z0, ~out), acc=EV.acc_clean(X, y, th2, z0, ~out), ahat=a2)


if __name__ == "__main__":
    U = unit_list()
    a = int(os.environ.get("UNIT_START", 0)); b = int(os.environ.get("UNIT_END", len(U)))
    d = os.environ.get("RESULTS_DIR", "."); os.makedirs(d, exist_ok=True)
    f = os.path.join(d, "bvar_%s.csv" % os.environ.get("CHUNK_ID", "0"))
    if os.path.exists(f):
        os.remove(f)
    n = 0
    with Pool(int(os.environ.get("WORKERS", "2"))) as p:
        for row in p.imap_unordered(run, U[a:b], chunksize=1):
            pd.DataFrame([row]).to_csv(f, mode="a", header=(n == 0), index=False)
            n += 1
            if n % 20 == 0:
                print("UNITS_DONE=%d" % n, flush=True)
    print("UNITS_DONE=%d" % n)
