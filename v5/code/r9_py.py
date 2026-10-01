"""Round 9 (descriptive), per study unit (global list as in oct_long.sh):
  TMIX            mixture of t regressions (tmix.py), 20 starts
  TCRlong(0.40)+irw, +irw+A2   incremental reweighting in the manner of Dotto et al. (2018) after
                  TCLUST-REG at 0.40 with the long search: the trimming level is lowered from 0.40
                  to 0 in steps of 0.02, keeping at each step the units closest to their nearest
                  line in robust scales and refitting, then the reweighting step (c = 2.5)
  TCRlong(0.40)+rw+A2R, Alg1+A2R   Algorithm 2 with the noise range taken over the unflagged units
Usage: OCT_DIR=... UNIT_START=a UNIT_END=b WORKERS=w python r9_py.py OUTDIR"""
import importlib, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

import tcrlong_post as TP          # builds TP.O (long-search fits with study) and TP.LIST
A1 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "alg1_fits.csv"))


def robust_scales(X, y, th, keep):
    R = np.abs(y[:, None] - X @ th.T)
    z = R.argmin(axis=1)
    s = np.empty(th.shape[0])
    for k in range(th.shape[0]):
        r = R[keep & (z == k), k]
        s[k] = np.median(r) / 0.6745 if len(r) > 0 else np.median(R.min(axis=1)) / 0.6745
    return np.maximum(s, 1e-12)


def incr_reweight(X, y, th0, alpha0=0.40, step=0.02):
    import py_methods as P
    n, d = X.shape; K = th0.shape[0]
    th = th0.copy(); keep = np.ones(n, dtype=bool)
    for a in np.arange(alpha0, -1e-9, -step):
        s = robust_scales(X, y, th, keep)
        Z = np.abs(y[:, None] - X @ th.T) / s[None, :]
        z = Z.argmin(axis=1); dist = Z.min(axis=1)
        h = n - int(round(a * n))
        keep = np.zeros(n, dtype=bool); keep[np.argsort(dist)[:h]] = True
        for k in range(K):
            S = keep & (z == k)
            if S.sum() >= d + 1:
                th[k] = np.linalg.lstsq(X[S], y[S], rcond=None)[0]
    return P.reweight(X, y, th, xtrim=False)


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
    import simv4, evaluate as EV, py_methods as P, tmix
    importlib.reload(simv4)
    U = {simv4.unit_name(u): u for u in simv4.units()}
    rows = []
    for name in names:
        u = U[name]
        X, y, z0, out = simv4.generate(u)
        D = simv4.DESIGNS[u[0]]; K, d = D["K"], X.shape[1]
        acc = lambda th: EV.acc_clean(X, y, th, z0, ~out)
        base = dict(study=st, design=u[0], eps=u[1], rep=u[2], unit=name)
        th, ll, nu = tmix.tmix(X, y, K, np.random.default_rng(simv4.seed_for("v4tmix", *u)))
        rows.append(dict(base, method="TMIX", acc=0.0 if th is None else acc(th), nu=nu))
        r = TP.O[(TP.O.unit == name) & (TP.O.study == st) & (TP.O.method == "TCRlong(0.40)")]
        b = np.array([r.iloc[0][f"b{j+1}"] for j in range(K * d)], float)
        if np.isfinite(b).all():
            th0 = b.reshape(K, d)
            thi, ai = incr_reweight(X, y, th0)
            rows.append(dict(base, method="TCRlong(0.40)+irw", acc=acc(thi), ahat=ai))
            rng = np.random.default_rng(simv4.seed_for("v4rec_tcrlong", "TCRlong(0.40)", *u))
            rows.append(dict(base, method="TCRlong(0.40)+irw+A2", acc=acc(rec(X, y, thi, rng))))
            thr, ar = P.reweight(X, y, th0, xtrim=False)
            rng = np.random.default_rng(simv4.seed_for("v4rec_tcrlong", "TCRlong(0.40)", *u))
            rows.append(dict(base, method="TCRlong(0.40)+rw+A2R", acc=acc(rec(X, y, thr, rng, noise_range="unflagged"))))
        a = A1[(A1.unit == name) & (A1.study == st)].iloc[0]
        th1 = np.array([a[f"b{j+1}"] for j in range(K * d)], float).reshape(K, d)
        rng = np.random.default_rng(simv4.seed_for("v4rec", *u))
        rows.append(dict(base, method="Alg1+A2R", acc=acc(rec(X, y, th1, rng, noise_range="unflagged"))))
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
    pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(outd, "r9py_%s.csv" % os.environ.get("CHUNK_ID", "0")), index=False)
    print("UNITS_DONE=%d" % (b - a))
