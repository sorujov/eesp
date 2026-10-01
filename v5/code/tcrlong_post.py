"""TCLUST-REG long search (oct_long.m output) followed by the reweighting step and Algorithm 2
(round 8, descriptive).  Global unit list as in oct_long.sh.  Usage:
OCT_DIR=<folder with octlong_*.csv> UNIT_START=a UNIT_END=b WORKERS=w python tcrlong_post.py OUTDIR"""
import glob, importlib, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

STUDIES = ["1", "1b", "1c", "1d", "1e", "1f", "1g", "1h"]
FILES = ["units_v4%s.csv" % s[1:] for s in STUDIES]
LIST = []
for st, f in zip(STUDIES, FILES):
    for nm in pd.read_csv(f).name:
        LIST.append((st, nm))
# Unit names repeat across studies (parts 1 and 1b share the e30 names), so every row is mapped
# back to its global index from the order in which oct_long.m wrote it: worker k of a chunk
# (start S, end E, W workers) takes indices S+k, S+k+W, ... (0-based), or, in reverse mode,
# E-1-k, E-1-k-W, ...; each index writes two rows.
CHUNKS = {"0": (0, 1333, 64, False), "1": (1333, 2666, 64, False), "2": (2666, 3958, 62, False),
          "3": (3958, 5083, 54, False), "4": (5083, 6083, 48, False), "5": (6083, 6917, 40, False),
          "6": (6917, 7750, 40, False), "h5": (6083, 6917, 60, True)}
import re
parts = []
for f in glob.glob(os.path.join(os.environ["OCT_DIR"], "octlong_*.csv")):
    c, k = re.match(r"octlong_(\w+?)_(\d+)\.csv", os.path.basename(f)).groups()
    S, E, W, rev = CHUNKS[c]; k = int(k)
    idx = list(range(E - 1 - k, S - 1, -W)) if rev else list(range(S + k, E, W))
    df = pd.read_csv(f)
    df = df[df.method.isin(["TCRlong(0.25)", "TCRlong(0.40)"])]
    n = len(df) // 2
    df = df.iloc[:2 * n]                     # a cancelled worker may leave half a unit
    assert (df.method.values[0::2] == "TCRlong(0.25)").all() and (df.method.values[1::2] == "TCRlong(0.40)").all()
    df["gidx"] = np.repeat(idx[:n], 2)
    parts.append(df)
O = pd.concat(parts)
names = [nm for _, nm in LIST]
assert (O.unit.values == np.array(names)[O.gidx.values]).all(), "index reconstruction failed"
O["study"] = [LIST[i][0] for i in O.gidx]
O = O.drop_duplicates(["gidx", "method"])
assert O.gidx.nunique() == len(LIST), O.gidx.nunique()


def work(st_names):
    st, names = st_names
    os.environ["V4_STUDY"] = st
    import simv4, evaluate as EV, recover2 as R, py_methods as P
    importlib.reload(simv4)
    U = {simv4.unit_name(u): u for u in simv4.units()}
    rows = []
    for name in names:
        u = U[name]
        X, y, z0, out = simv4.generate(u)
        D = simv4.DESIGNS[u[0]]; K, d = D["K"], X.shape[1]
        for meth in ["TCRlong(0.25)", "TCRlong(0.40)"]:
            r = O[(O.unit == name) & (O.study == st) & (O.method == meth)]
            if r.empty:
                continue
            b = np.array([r.iloc[0][f"b{j+1}"] for j in range(K * d)], float)
            base = dict(study=st, design=u[0], eps=u[1], rep=u[2], unit=name, time=r.iloc[0].time)
            if not np.isfinite(b).all():
                for s in ("", "+rw", "+rw+A2"):
                    rows.append(dict(base, method=meth + s, acc=0.0, ahat=np.nan, fail=True))
                continue
            th0 = b.reshape(K, d)
            rows.append(dict(base, method=meth, acc=EV.acc_clean(X, y, th0, z0, ~out), ahat=r.iloc[0].alpha_hat, fail=False))
            th, ah = P.reweight(X, y, th0, xtrim=False)
            rows.append(dict(base, method=meth + "+rw", acc=EV.acc_clean(X, y, th, z0, ~out), ahat=ah, fail=False))
            th2, a2 = th, ah
            rng = np.random.default_rng(simv4.seed_for("v4rec_tcrlong", meth, *u))
            for _ in range(3):
                th3, a3, info = R.recover(X, y, th2, rng)
                if not info["accepted"]:
                    break
                th2, a2 = th3, a3
            rows.append(dict(base, method=meth + "+rw+A2", acc=EV.acc_clean(X, y, th2, z0, ~out), ahat=a2, fail=False))
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
    pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(outd, "tcrlong_%s.csv" % os.environ.get("CHUNK_ID", "0")), index=False)
    print("UNITS_DONE=%d" % (b - a))
