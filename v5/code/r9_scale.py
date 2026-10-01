"""Round 9 sensitivity (descriptive): the scale of the scoring loss of Algorithm 1 taken as the
minimum of the replicates' median squared residuals (the procedure) or as their 10% quantile,
with B = 100 and B = 500, each followed by Algorithm 2; parts one (eps <= 0.20) and three
(small groups and clustered outliers).  Usage: UNIT_START/UNIT_END/WORKERS python r9_scale.py OUT"""
import importlib, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

LIST = [("1", n) for n in pd.read_csv("units_v4.csv").name] + [("1e", n) for n in pd.read_csv("units_v4e.csv").name]


def work(st_names):
    st, names = st_names
    os.environ["V4_STUDY"] = st
    import simv4, evaluate as EV, py_methods as P, recover2 as R
    importlib.reload(simv4)
    U = {simv4.unit_name(u): u for u in simv4.units()}
    rows = []
    for name in names:
        u = U[name]
        X, y, z0, out = simv4.generate(u)
        D = simv4.DESIGNS[u[0]]; K, m = D["K"], D["m"]
        for B in (100, 500):
            for q in (None, 0.1):
                rng = np.random.default_rng(simv4.seed_for("v4scale", B, *u))
                th, ah, t = P.eesp_adaptive_x(X, y, K, m, B, rng, xtrim=True, scale_q=q)
                th2 = th
                for _ in range(3):
                    th3, a3, info = R.recover(X, y, th2, rng)
                    if not info["accepted"]:
                        break
                    th2 = th3
                rows.append(dict(study=st, design=u[0], eps=u[1], rep=u[2], unit=name, B=B,
                                 scale="min" if q is None else "q10",
                                 acc1=EV.acc_clean(X, y, th, z0, ~out), acc=EV.acc_clean(X, y, th2, z0, ~out)))
    return rows


if __name__ == "__main__":
    a = int(os.environ.get("UNIT_START", 0)); b = int(os.environ.get("UNIT_END", len(LIST)))
    W = int(os.environ.get("WORKERS", 1))
    jobs = []
    for k in range(a, b, 4):
        grp = LIST[k:min(k + 4, b)]
        for st in dict.fromkeys(s for s, _ in grp):
            jobs.append((st, [n for s, n in grp if s == st]))
    with Pool(W) as p:
        res = p.map(work, jobs, chunksize=1)
    outd = sys.argv[1]; os.makedirs(outd, exist_ok=True)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(outd, "r9scale_%s.csv" % os.environ.get("CHUNK_ID", "0")), index=False)
    print("UNITS_DONE=%d" % (b - a))
