"""Round 13 (descriptive), per study unit (global list as in oct_long.sh):
  ESF12, ESF12alone   ESF with one subsample size m = 12 in every design (the level-free default),
                      with and without the recovery step; seeded by the unit
  Alg2_Rs20, Alg2_Ru10, TCRlong(0.40)+rw+A2_Rs20, ..._Ru10   the recovery step with a noise range
                      tied to the fitted scales: 20 median scales, or the range of the unflagged
                      units plus 10 median scales
Usage: OCT_DIR=... UNIT_START=a UNIT_END=b WORKERS=w python r13.py OUTDIR"""
import importlib, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

import tcrlong_post as TP
A1 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "alg1_fits.csv"))
NR = (("Rs20", "scale20"), ("Ru10", "unflagged10"))


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
        rng = np.random.default_rng(simv4.seed_for("v4py", *u))
        th12, ah12, t12 = P.eesp_adaptive_x(X, y, K, 12, 100, rng, xtrim=True)
        rows.append(dict(base, method="ESF12alone", acc=acc(th12), ahat=ah12, time=t12))
        rng = np.random.default_rng(simv4.seed_for("v4rec", *u))
        rows.append(dict(base, method="ESF12", acc=acc(rec(X, y, th12, rng))))
        a = A1[(A1.unit == name) & (A1.study == st)].iloc[0]
        th1 = np.array([a[f"b{j+1}"] for j in range(K * d)], float).reshape(K, d)
        for tag, opt in NR:
            rng = np.random.default_rng(simv4.seed_for("v4rec", *u))
            rows.append(dict(base, method=f"Alg2_{tag}", acc=acc(rec(X, y, th1, rng, noise_range=opt))))
        r = TP.O[(TP.O.unit == name) & (TP.O.study == st) & (TP.O.method == "TCRlong(0.40)")]
        b = np.array([r.iloc[0][f"b{j+1}"] for j in range(K * d)], float)
        if np.isfinite(b).all():
            thr, ar = P.reweight(X, y, b.reshape(K, d), xtrim=False)
            for tag, opt in NR:
                rng = np.random.default_rng(simv4.seed_for("v4rec_tcrlong", "TCRlong(0.40)", *u))
                rows.append(dict(base, method=f"TCRlong(0.40)+rw+A2_{tag}", acc=acc(rec(X, y, thr, rng, noise_range=opt))))
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
    pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(outd, "r13_%s.csv" % os.environ.get("CHUNK_ID", "0")), index=False)
    print("UNITS_DONE=%d" % (b - a))
