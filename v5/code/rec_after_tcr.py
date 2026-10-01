"""The recovery step applied after TCLUST-REG + reweighting (descriptive ablation)."""
import glob, os, sys
import numpy as np, pandas as pd
import simv4, evaluate as EV, recover2 as R, py_methods as P
FOLD = sys.argv[1]
O = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(FOLD, "oct_*.csv"))])
rows = []
for u in simv4.units():
    name = simv4.unit_name(u)
    X, y, z0, out = simv4.generate(u); D = simv4.DESIGNS[u[0]]; K, d = D["K"], X.shape[1]
    for meth in ["TCR(0.25)", "TCR(0.40)"]:
        r = O[(O.unit == name) & (O.method == meth)]
        if r.empty:
            continue
        b = np.array([r.iloc[0][f"b{j+1}"] for j in range(K * d)], float)
        if not np.isfinite(b).all():
            continue
        th, ah = P.reweight(X, y, b.reshape(K, d), xtrim=False)
        th2 = th
        rng = np.random.default_rng(simv4.seed_for("v4rec_tcr", meth, *u))
        for _ in range(3):
            th3, a3, info = R.recover(X, y, th2, rng)
            if not info["accepted"]:
                break
            th2 = th3
        rows.append(dict(design=u[0], eps=u[1], rep=u[2], method=meth + "+rw",
                         acc=EV.acc_clean(X, y, th, z0, ~out), acc_rec=EV.acc_clean(X, y, th2, z0, ~out)))
df = pd.DataFrame(rows); df.to_csv(os.path.join(FOLD, "rec_after_tcr.csv"), index=False)
print(df.groupby(["design", "eps", "method"])[["acc", "acc_rec"]].mean().round(3))
