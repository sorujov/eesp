"""Round 12 (descriptive): sensitivity of the recovery step to the noise range R = 1.1 range(y).
R is multiplied by f in {2, 5}, as if the most extreme outlier lay two or five times farther out,
for ESF (stored fits of ESF alone, then the recovery step) and for TCLUST-REG at level 0.40 with
the long search and reweighting.  Seeds as in the study.
Usage: OCT_DIR=... UNIT_START=a UNIT_END=b WORKERS=w python r12_range.py OUTDIR"""
import importlib, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

import tcrlong_post as TP
A1 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "alg1_fits.csv"))
F = (2.0, 5.0)


def rec(X, y, th, rng, **kw):
    import recover2 as R
    th2 = th
    for _ in range(3):
        th3, a3, info = R.recover(X, y, th2, rng, **kw)
        if not info["accepted"]:
            break
        th2 = th3
    return th2


def work(st_names):
    st, names = st_names
    os.environ["V4_STUDY"] = st
    import simv4, evaluate as EV, py_methods as P
    importlib.reload(simv4)
    U = {simv4.unit_name(u): u for u in simv4.units()}
    rows = []
    for name in names:
        u = U[name]
        X, y, z0, out = simv4.generate(u)
        D = simv4.DESIGNS[u[0]]; K, d = D["K"], X.shape[1]
        acc = lambda th: EV.acc_clean(X, y, th, z0, ~out)
        base = dict(study=st, design=u[0], eps=u[1], rep=u[2], unit=name)
        a = A1[(A1.unit == name) & (A1.study == st)].iloc[0]
        th1 = np.array([a[f"b{j+1}"] for j in range(K * d)], float).reshape(K, d)
        for f in F:
            rng = np.random.default_rng(simv4.seed_for("v4rec", *u))
            rows.append(dict(base, method=f"Alg2_R{int(f)}", acc=acc(rec(X, y, th1, rng, noise_range=f))))
        r = TP.O[(TP.O.unit == name) & (TP.O.study == st) & (TP.O.method == "TCRlong(0.40)")]
        b = np.array([r.iloc[0][f"b{j+1}"] for j in range(K * d)], float)
        if np.isfinite(b).all():
            thr, ar = P.reweight(X, y, b.reshape(K, d), xtrim=False)
            for f in F:
                rng = np.random.default_rng(simv4.seed_for("v4rec_tcrlong", "TCRlong(0.40)", *u))
                rows.append(dict(base, method=f"TCRlong(0.40)+rw+A2_R{int(f)}", acc=acc(rec(X, y, thr, rng, noise_range=f))))
    return rows


if __name__ == "__main__":
    a = int(os.environ.get("UNIT_START", 0)); b = int(os.environ.get("UNIT_END", len(TP.LIST)))
    W = int(os.environ.get("WORKERS", 1))
    jobs = []
    for k in range(a, b, 5):
        grp = TP.LIST[k:min(k + 5, b)]
        for st in dict.fromkeys(s for s, _ in grp):
            jobs.append((st, [n for s, n in grp if s == st]))
    with Pool(W) as p:
        res = p.map(work, jobs, chunksize=1)
    outd = sys.argv[1]; os.makedirs(outd, exist_ok=True)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(outd, "r12range_%s.csv" % os.environ.get("CHUNK_ID", "0")), index=False)
    print("UNITS_DONE=%d" % (b - a))
