"""Round 8: taxi samples, TCLUST-REG long search at 0.25/0.40 + reweighting + Algorithm 2."""
import os, sys
os.environ["V4_STUDY"] = "taxi"
import numpy as np, pandas as pd
import simv4, evaluate as EV, recover2 as R, py_methods as P
O = pd.read_csv(sys.argv[1])
rows = []
for u in simv4.units():
    name = simv4.unit_name(u)
    X, y, z0, out = simv4.generate(u)
    D = simv4.DESIGNS[u[0]]; K, d = D["K"], X.shape[1]
    for meth in ["TCRlong(0.25)", "TCRlong(0.40)"]:
        r = O[(O.unit == name) & (O.method == meth)].iloc[0]
        th0 = np.array([r[f"b{j+1}"] for j in range(K * d)], float).reshape(K, d)
        rows.append(dict(unit=name, method=meth, acc=EV.acc_clean(X, y, th0, z0, ~out), ahat=r.alpha_hat, time=r.time))
        th, ah = P.reweight(X, y, th0, xtrim=False)
        rows.append(dict(unit=name, method=meth + "+rw", acc=EV.acc_clean(X, y, th, z0, ~out), perr=EV.align_err(th, EV.true_theta(D)), ahat=ah, time=r.time))
        th2, a2 = th, ah
        rng = np.random.default_rng(simv4.seed_for("v4rec_tcrlong", meth, *u))
        for _ in range(3):
            th3, a3, info = R.recover(X, y, th2, rng)
            if not info["accepted"]:
                break
            th2, a2 = th3, a3
        rows.append(dict(unit=name, method=meth + "+rw+A2", acc=EV.acc_clean(X, y, th2, z0, ~out), perr=EV.align_err(th2, EV.true_theta(D)), ahat=a2, time=r.time))
df = pd.DataFrame(rows); df.to_csv(os.path.join(os.path.dirname(sys.argv[1]), "tcrlong_taxi.csv"), index=False)
print(df.groupby("method").agg(acc=("acc", "mean"), accmin=("acc", "min"), ahat=("ahat", "mean"), t=("time", "median")).round(3))
