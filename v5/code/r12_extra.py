"""Round 12 (descriptive): 150 further data sets (replicates 50 to 199, same generators and seeds
rule as the study) in the eleven cells that decide the comparison up to 20% of outliers:
D2 at 0, 0.05, 0.10, 0.20 (part 1); four groups at 0, 0.10, 0.20 (part 3, 1c); groups 80/20 and
clustered outliers at 0.20 (1e); groups 80/20 with three covariates and 85/15 parallel at 0.20 (1g).
Modes (argv[1]):
  units DIR   write the unit list (name with study folder, K, rseed) and the data files under DIR
  esf OUT     ESF alone (stored for the post step) and ESF, units UNIT_START..UNIT_END of the list
  post OUT    accuracy of TCLUST-REG fits alone, reweighted, and followed by the recovery step
Seeds: data and ESF exactly as in the study (seed_for("v4data"/"v4py"/"v4rec", unit)); Octave
rseed = seed_for("v4oct_extra", unit) mod 10^6 + 1."""
import glob, importlib, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

CELLS = {"1": [("D2-K3par", e) for e in (0.0, 0.05, 0.10, 0.20)],
         "1c": [("E4-K4par", e) for e in (0.0, 0.10, 0.20)],
         "1e": [("G1-small20", 0.20), ("G3-cluster", 0.20)],
         "1g": [("G4-small20-p3", 0.20), ("G5-small15-par", 0.20)]}
REPS = range(50, 200)


def load(st):
    os.environ["V4_STUDY"] = st; os.environ["V4_REPS"] = "200"
    import simv4
    importlib.reload(simv4)
    return simv4


def unit_list():
    L = []
    for st, cells in CELLS.items():
        for d, e in cells:
            for r in REPS:
                L.append((st, (d, e, r)))
    return L


def esf(item):
    st, u = item
    S = load(st)
    import py_methods as P, recover2 as R, evaluate as EV
    X, y, z0, out = S.generate(u)
    D = S.DESIGNS[u[0]]; K = D["K"]
    rng = np.random.default_rng(S.seed_for("v4py", *u))
    P.eesp_adaptive_x(X, y, K, D["m"], 100, rng, xtrim=False)     # as run_all: same stream
    th, ah, t = P.eesp_adaptive_x(X, y, K, D["m"], 100, rng, xtrim=True)
    rows = [dict(study=st, design=u[0], eps=u[1], rep=u[2], method="adaptX", acc=EV.acc_clean(X, y, th, z0, ~out), ahat=ah)]
    rng = np.random.default_rng(S.seed_for("v4rec", *u))
    th2, ah2 = th, ah
    for _ in range(3):
        th3, ah3, info = R.recover(X, y, th2, rng)
        if not info["accepted"]:
            break
        th2, ah2 = th3, ah3
    rows.append(dict(study=st, design=u[0], eps=u[1], rep=u[2], method="Alg2", acc=EV.acc_clean(X, y, th2, z0, ~out), ahat=ah2))
    return rows


def post(args):
    (st, u), F = args
    S = load(st)
    import py_methods as P, recover2 as R, evaluate as EV
    X, y, z0, out = S.generate(u)
    D = S.DESIGNS[u[0]]; K, d = D["K"], X.shape[1]
    base = dict(study=st, design=u[0], eps=u[1], rep=u[2])
    rows = []
    for _, r in F.iterrows():
        m = r.method
        b = np.array([r[f"b{j+1}"] for j in range(K * d)], float)
        if not np.isfinite(b).all():
            for s in ("", "+rw", "+rw+A2"):
                rows.append(dict(base, method=m + s, acc=0.0)); continue
        th0 = b.reshape(K, d)
        rows.append(dict(base, method=m, acc=EV.acc_clean(X, y, th0, z0, ~out)))
        th, ah = P.reweight(X, y, th0, xtrim=False)
        rows.append(dict(base, method=m + "+rw", acc=EV.acc_clean(X, y, th, z0, ~out)))
        tag = ("v4rec_tcrlong", m) if m in ("TCRlong(0.25)", "TCRlong(0.40)") else ("v4rec_r9", m)
        rng = np.random.default_rng(S.seed_for(*tag, *u))
        th2 = th
        for _ in range(3):
            th3, a3, info = R.recover(X, y, th2, rng)
            if not info["accepted"]:
                break
            th2 = th3
        rows.append(dict(base, method=m + "+rw+A2", acc=EV.acc_clean(X, y, th2, z0, ~out)))
    return rows


if __name__ == "__main__":
    mode, outd = sys.argv[1], sys.argv[2]
    os.makedirs(outd, exist_ok=True)
    L = unit_list()
    a = int(os.environ.get("UNIT_START", 0)); b = min(int(os.environ.get("UNIT_END", len(L))), len(L))
    W = int(os.environ.get("WORKERS", 1)); C = os.environ.get("CHUNK_ID", "0")
    if mode == "units":
        with open(os.path.join(outd, "units.csv"), "w") as f:
            f.write("name,K,rseed\n")
            for st, u in L:
                S = load(st)
                os.makedirs(os.path.join(outd, "s" + st), exist_ok=True)
                S.write_csv(u, os.path.join(outd, "s" + st))
                f.write(f"s{st}/{S.unit_name(u)},{S.DESIGNS[u[0]]['K']},{S.seed_for('v4oct_extra', *u) % 1000000 + 1}\n")
        print("units", len(L))
    elif mode == "esf":
        with Pool(W) as p:
            res = p.map(esf, L[a:b], chunksize=1)
        pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(outd, f"esf_{C}.csv"), index=False)
        print(f"UNITS_DONE={b - a}")
    elif mode == "post":
        F = pd.concat([pd.read_csv(f) for pat in os.environ["FITS"].split(",") for f in glob.glob(pat)], ignore_index=True)
        F = F.drop_duplicates(["unit", "method"])
        jobs = []
        for st, u in L[a:b]:
            load(st)
            import simv4
            nm = f"s{st}/{simv4.unit_name(u)}"
            jobs.append(((st, u), F[F.unit == nm]))
        with Pool(W) as p:
            res = p.map(post, jobs, chunksize=1)
        pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(outd, f"post_{C}.csv"), index=False)
        print(f"UNITS_DONE={b - a}")
