"""Algorithm 2 applied after the trimmed alternation at level 0.40 with reweighting (descriptive):
the natural home-made rival 'generous trimming + reweighting + recovery'.
Usage: V4_STUDY=... python ta_rec.py RESULTS_DIR   (writes ta_rec.csv)"""
import glob, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, evaluate as EV, recover2 as R

FOLD = sys.argv[1]
A = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(FOLD, "py_*.csv"))])
A = A[A.method == "TA(0.40)"].set_index("unit")


def run(u):
    name = simv4.unit_name(u)
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]; K, d = D["K"], X.shape[1]
    rec = A.loc[name]
    th = np.array([rec[f"b{j+1}"] for j in range(K * d)], float).reshape(K, d)
    if not np.isfinite(th).all():
        return dict(design=u[0], eps=u[1], rep=u[2], unit=name, acc=0.0, acc_rec=0.0)
    rng = np.random.default_rng(simv4.seed_for("v4tarec", *u))
    th2 = th
    for _ in range(3):
        th3, ah, info = R.recover(X, y, th2, rng)
        if not info["accepted"]:
            break
        th2 = th3
    return dict(design=u[0], eps=u[1], rep=u[2], unit=name, acc=EV.acc_clean(X, y, th, z0, ~out),
                acc_rec=EV.acc_clean(X, y, th2, z0, ~out))


if __name__ == "__main__":
    with Pool(2) as p:
        rows = p.map(run, simv4.units(), chunksize=8)
    pd.DataFrame(rows).to_csv(os.path.join(FOLD, "ta_rec.csv"), index=False)
    print(len(rows))
