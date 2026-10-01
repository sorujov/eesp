"""Round 9 post-processing (descriptive): accuracy of stored fits, alone, after the reweighting
step, and after Algorithm 2.  Fit files have unit names "s<study>/<unit>" (oct_r9.sh, r9_rlga.sh).
Methods: RLGA(a), TCRx99(a), TCReqw(0.40), TCRr1(0.40), and MONsel / MONsel2, the level chosen
on the monitoring path MON(0), ..., MON(0.40):
  MONsel : the level after the first pair of consecutive levels whose classifications agree
           with adjusted Rand index at least 0.9 (0.40 if there is none);
  MONsel2: the same with two consecutive pairs at least 0.9;
  MONrest9, MONrest8: the smallest level from which every later pair of consecutive levels agrees
           with adjusted Rand index at least 0.9 (0.8).
Usage: FITS="a.csv,b.csv,..." UNIT_START/UNIT_END/WORKERS python fits_post.py OUTDIR"""
import glob, importlib, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

STUDIES = ["1", "1b", "1c", "1d", "1e", "1f", "1g", "1h"]
LIST = []
for st in STUDIES:
    for nm in pd.read_csv("units_v4%s.csv" % st[1:]).name:
        LIST.append((st, nm))
F = pd.concat([pd.read_csv(f) for pat in os.environ["FITS"].split(",") for f in glob.glob(pat)], ignore_index=True)
F["study"] = F.unit.str.split("/").str[0].str[1:]
F["unit"] = F.unit.str.split("/").str[1]
F = F.drop_duplicates(["study", "unit", "method"])
LEV = [round(0.05 * j, 2) for j in range(9)]


def monsel(g, need, thr=0.9):
    ari = g[g.method == "MONARI"]
    if ari.empty:
        return None
    a = [ari.iloc[0][f"b{j+1}"] for j in range(8)]
    if need == "rest":
        # the smallest level from which every later pair of consecutive levels agrees
        for j in range(8):
            if all(np.isfinite(v) and v >= thr for v in a[j:]):
                return LEV[j]
        return 0.40
    for j in range(8 - need + 1):
        if all(np.isfinite(a[j + t]) and a[j + t] >= 0.9 for t in range(need)):
            return LEV[j + 1]
    return 0.40


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
        D = simv4.DESIGNS[u[0]]; K, d = D["K"], X.shape[1]
        g = F[(F.study == st) & (F.unit == name)]
        fits = {m: g[g.method == m].iloc[0] for m in g.method.unique() if not m.startswith("MON")}
        for tag, need, thr in (("MONsel", 1, 0.9), ("MONsel2", 2, 0.9), ("MONrest9", "rest", 0.9), ("MONrest8", "rest", 0.8)):
            lv = monsel(g, need, thr)
            if lv is not None:
                r = g[g.method == "MON(%.2f)" % lv]
                if not r.empty:
                    fits[tag] = r.iloc[0]
                    rows.append(dict(study=st, design=u[0], eps=u[1], rep=u[2], unit=name, method=tag + "_level", acc=lv))
        for m, r in fits.items():
            base = dict(study=st, design=u[0], eps=u[1], rep=u[2], unit=name)
            b = np.array([r[f"b{j+1}"] for j in range(K * d)], float)
            if not np.isfinite(b).all():
                for s in ("", "+rw", "+rw+A2"):
                    rows.append(dict(base, method=m + s, acc=0.0, fail=True))
                continue
            th0 = b.reshape(K, d)
            rows.append(dict(base, method=m, acc=EV.acc_clean(X, y, th0, z0, ~out), ahat=r.alpha_hat, time=r.time))
            th, ah = P.reweight(X, y, th0, xtrim=False)
            rows.append(dict(base, method=m + "+rw", acc=EV.acc_clean(X, y, th, z0, ~out), ahat=ah))
            th2, a2 = th, ah
            rng = np.random.default_rng(simv4.seed_for("v4rec_r9", m, *u))
            for _ in range(3):
                th3, a3, info = R.recover(X, y, th2, rng)
                if not info["accepted"]:
                    break
                th2, a2 = th3, a3
            rows.append(dict(base, method=m + "+rw+A2", acc=EV.acc_clean(X, y, th2, z0, ~out), ahat=a2))
    return rows


if __name__ == "__main__":
    a = int(os.environ.get("UNIT_START", 0)); b = int(os.environ.get("UNIT_END", len(LIST)))
    W = int(os.environ.get("WORKERS", 1))
    jobs = []
    for k in range(a, b, 10):
        grp = LIST[k:min(k + 10, b)]
        for st in dict.fromkeys(s for s, _ in grp):
            jobs.append((st, [n for s, n in grp if s == st]))
    with Pool(W) as p:
        res = p.map(work, jobs, chunksize=1)
    outd = sys.argv[1]; os.makedirs(outd, exist_ok=True)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(outd, "fitspost_%s.csv" % os.environ.get("CHUNK_ID", "0")), index=False)
    print("UNITS_DONE=%d" % (b - a))
