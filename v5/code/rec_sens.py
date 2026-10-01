"""Sensitivity of Algorithm 2 to its constants (descriptive, added after the review of 27 Sep 2026).
Each constant is moved one step down and one step up, the others kept, and the recovery step is
applied to the stored fit of Algorithm 1 as in Gate 2 (run_rec.py).
Usage: V4_STUDY=... python rec_sens.py RESULTS_DIR"""
import glob, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, evaluate as EV, recover2 as R

FOLD = sys.argv[1]
A = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(FOLD, "py_*.csv"))])
A = A[A.method == "adaptX"].set_index("unit")
CONFIGS = {
    "base": {},
    "min_w=0.4": dict(min_w=0.4), "min_w=0.6": dict(min_w=0.6),
    "wide=2": dict(wide=2.0), "wide=4": dict(wide=4.0),
    "min_frac=0.02": dict(min_frac=0.02), "min_frac=0.05": dict(min_frac=0.05),
    "restr=8": dict(restr=8.0), "restr=20": dict(restr=20.0),
    "margin/2": "half", "margin*2": "double",
    "max_alpha=0.25": dict(max_alpha=0.25), "max_alpha=0.40": dict(max_alpha=0.40),
}


def run(u):
    name = simv4.unit_name(u)
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    K, d = D["K"], X.shape[1]
    n = len(y)
    rec = A.loc[name]
    th0 = np.array([rec[f"b{j+1}"] for j in range(K * d)], float).reshape(K, d)
    row = dict(design=u[0], eps=u[1], rep=u[2], unit=name,
               alg1=EV.acc_clean(X, y, th0, z0, ~out))
    for cname, kw in CONFIGS.items():
        if isinstance(kw, str):
            g = 0.5 * (d + 1) * np.log(n) * (0.5 if kw == "half" else 2.0)
            kw = dict(min_ll_gain=g)
        rng = np.random.default_rng(simv4.seed_for("v4rec", *u))
        th2 = th0
        for _ in range(3):
            th3, ah3, info = R.recover(X, y, th2, rng, **kw)
            if not info["accepted"]:
                break
            th2 = th3
        row[cname] = EV.acc_clean(X, y, th2, z0, ~out)
    return row


if __name__ == "__main__":
    with Pool(int(os.environ.get("WORKERS", "2"))) as p:
        rows = p.map(run, simv4.units(), chunksize=8)
    pd.DataFrame(rows).to_csv(os.path.join(FOLD, "rec_sens.csv"), index=False)
    print(len(rows))
