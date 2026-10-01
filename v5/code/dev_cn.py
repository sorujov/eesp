import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
import simv4, evaluate as EV, cn_mix as C, json
KW = json.loads(os.environ.get("KW", "{}"))
simv4.SEED0 = 777777
def run(u):
    X, y, z0, out = simv4.generate(u); D = simv4.DESIGNS[u[0]]
    th, ah, t = C.cn_mix(X, y, D["K"], np.random.default_rng(1), **KW)
    return dict(design=u[0], eps=u[1], acc=EV.acc_clean(X, y, th, z0, ~out), ahat=ah, t=t)
U = [(d, e, r) for d in sys.argv[1].split(",") for e in [0, 0.1, 0.2, 0.3] for r in range(10)]
with Pool(2) as p: R = pd.DataFrame(p.map(run, U))
print(R.groupby(["design", "eps"])[["acc", "ahat", "t"]].mean().round(3))
