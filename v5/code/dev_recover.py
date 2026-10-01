"""Development harness: fresh data (SEED0 overridden), Algorithm 1 vs Algorithm 1 + recovery.
Usage: V4_STUDY=... python dev_recover.py OUT.csv DESIGNS EPS REPS"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import simv4, evaluate as EV, recover as R
simv4.SEED0 = int(os.environ.get("DEV_SEED0", "777777"))


def run(u):
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]
    rng = np.random.default_rng(simv4.seed_for("dev", *u))
    th2, a2, info, th, a = R.algorithm1_r(X, y, D["K"], D["m"], 100, rng)
    return dict(design=u[0], eps=u[1], rep=u[2], acc=EV.acc_clean(X, y, th, z0, ~out), ahat=a,
                acc_r=EV.acc_clean(X, y, th2, z0, ~out), ahat_r=a2, tried=info["tried"],
                acc_changed=info["accepted"], reason=info.get("reason", ""))


if __name__ == "__main__":
    des = sys.argv[2].split(","); eps = [float(e) for e in sys.argv[3].split(",")]; reps = int(sys.argv[4])
    U = [(d, e, r) for d in des for e in eps for r in range(reps)]
    with Pool(2) as p:
        rows = p.map(run, U, chunksize=2)
    df = pd.DataFrame(rows); df.to_csv(sys.argv[1], index=False)
    g = df.groupby(["design", "eps"])
    print(pd.concat([g.acc.mean(), g.acc_r.mean(), g.acc_changed.mean(), g.ahat.mean(), g.ahat_r.mean()], axis=1).round(3).to_string())
