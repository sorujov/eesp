"""Gate 2: Algorithm 2 = stored Algorithm 1 fit (adaptX) + recovery (recover2.py) on every data
set of a study.  Usage: V4_STUDY=... python run_rec.py RESULTS_DIR"""
import glob, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, evaluate as EV, recover2 as R

FOLD = sys.argv[1]
A = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(FOLD, "py_*.csv"))])
A = A[A.method == "adaptX"].set_index("unit")


def run(u):
    name = simv4.unit_name(u)
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    K, d = D["K"], X.shape[1]
    rec = A.loc[name]
    th = np.array([rec[f"b{j+1}"] for j in range(K * d)], float).reshape(K, d)
    rng = np.random.default_rng(simv4.seed_for("v4rec", *u))
    t0 = time.perf_counter()
    th2, ah2 = th, float(rec.alpha_hat)
    steps = 0
    for _ in range(3):
        th3, ah3, info = R.recover(X, y, th2, rng)
        if not info["accepted"]:
            break
        th2, ah2, steps = th3, ah3, steps + 1
    t = time.perf_counter() - t0
    a = EV.acc_clean(X, y, th2, z0, ~out)
    return dict(design=u[0], eps=u[1], rep=u[2], unit=name, method="Alg2", acc=a,
                perr=EV.align_err(th2, EV.true_theta(D)), fail=bool(a < 0.7), ahat=ah2,
                time=float(rec.time) + t, rec_time=t, steps=steps,
                **{f"t{j+1}": v for j, v in enumerate(th2.reshape(-1))})


if __name__ == "__main__":
    with Pool(2) as p:
        rows = p.map(run, simv4.units(), chunksize=8)
    pd.DataFrame(rows).to_csv(os.path.join(FOLD, "alg2.csv"), index=False)
    print(len(rows))
