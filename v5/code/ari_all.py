"""Round 10 (descriptive): criteria that count flagged or trimmed clean units, for the procedure
and the long-search TCLUST-REG pipelines.  For each final fit every unit gets the label of its
nearest line, or 0 if the fit flags it (the flagging rule of the reweighting step, c = 2.5, with
the covariate screen for the procedure and without it for TCLUST-REG); the truth labels the
outliers 0.  Reported: the adjusted Rand index on all units (ARI), and the misclassification rate
of the clean units counting a flagged clean unit as an error (MIS).
Usage: OCT_DIR=... UNIT_START/UNIT_END/WORKERS python ari_all.py OUTDIR"""
import importlib, os, sys, itertools
from multiprocessing import Pool
import numpy as np, pandas as pd
import tcrlong_post as TP

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FOLD = {"1": "results_v4", "1b": "results_v4b", "1c": "results_v4c", "1d": "results_v4d",
        "1e": "results_v4e", "1f": "results_v4f", "1g": "results_v4g", "1h": "results_v4h"}
A2 = {st: pd.read_csv(os.path.join(os.environ.get("ALG2_DIR", ROOT), f, "alg2.csv")).set_index("unit") for st, f in FOLD.items()}


def ari(a, b):
    from scipy.special import comb
    _, ia = np.unique(a, return_inverse=True); _, ib = np.unique(b, return_inverse=True)
    C = np.zeros((ia.max() + 1, ib.max() + 1)); np.add.at(C, (ia, ib), 1)
    s = comb(C, 2).sum(); x = comb(C.sum(1), 2).sum(); y = comb(C.sum(0), 2).sum(); n = comb(len(a), 2)
    e = x * y / n; m = (x + y) / 2
    return 1.0 if m == e else (s - e) / (m - e)


def scores(X, y, th, z0, out, xtrim, P):
    fl, _, z = P._flag_all(X, y, th, 2.5, xtrim, 0.975)
    K = th.shape[0]
    lab = np.where(fl, 0, z + 1)
    truth = np.where(out, 0, z0 + 1)
    best = 1.0
    for perm in itertools.permutations(range(K)):
        m = np.array(perm)[z] + 1
        e = np.mean((fl | (m != truth))[~out])
        best = min(best, e)
    return ari(lab, truth), best


def work(st_names):
    st, names = st_names
    os.environ["V4_STUDY"] = st
    import simv4, py_methods as P, recover2 as R
    importlib.reload(simv4)
    U = {simv4.unit_name(u): u for u in simv4.units()}
    rows = []
    for name in names:
        u = U[name]
        X, y, z0, out = simv4.generate(u)
        D = simv4.DESIGNS[u[0]]; K, d = D["K"], X.shape[1]
        base = dict(study=st, design=u[0], eps=u[1], rep=u[2], unit=name)
        r = A2[st].loc[name]
        th = np.array([r[f"t{j+1}"] for j in range(K * d)], float).reshape(K, d)
        a, m = scores(X, y, th, z0, out, True, P)
        rows.append(dict(base, method="Alg2", ari=a, mis=m))
        for meth in ["TCRlong(0.25)", "TCRlong(0.40)"]:
            q = TP.O[(TP.O.unit == name) & (TP.O.study == st) & (TP.O.method == meth)]
            b = np.array([q.iloc[0][f"b{j+1}"] for j in range(K * d)], float)
            if not np.isfinite(b).all():
                continue
            thr, _ = P.reweight(X, y, b.reshape(K, d), xtrim=False)
            a, m = scores(X, y, thr, z0, out, False, P)
            rows.append(dict(base, method=meth + "+rw", ari=a, mis=m))
            th2 = thr
            rng = np.random.default_rng(simv4.seed_for("v4rec_tcrlong", meth, *u))
            for _ in range(3):
                th3, a3, info = R.recover(X, y, th2, rng)
                if not info["accepted"]:
                    break
                th2 = th3
            a, m = scores(X, y, th2, z0, out, False, P)
            rows.append(dict(base, method=meth + "+rw+A2", ari=a, mis=m))
    return rows


if __name__ == "__main__":
    a = int(os.environ.get("UNIT_START", 0)); b = int(os.environ.get("UNIT_END", len(TP.LIST)))
    W = int(os.environ.get("WORKERS", 1))
    jobs = []
    for k in range(a, b, 10):
        grp = TP.LIST[k:min(k + 10, b)]
        for st in dict.fromkeys(s for s, _ in grp):
            jobs.append((st, [n for s, n in grp if s == st]))
    with Pool(W) as p:
        res = p.map(work, jobs, chunksize=1)
    outd = sys.argv[1]; os.makedirs(outd, exist_ok=True)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(outd, "ari_%s.csv" % os.environ.get("CHUNK_ID", "0")), index=False)
    print("UNITS_DONE=%d" % (b - a))
