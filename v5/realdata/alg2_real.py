"""Algorithm 2 (recovery) on the fishery and tone data, after Algorithm 1, 20 seeds."""
import sys; sys.path.insert(0, "../code")
import numpy as np, pandas as pd
import py_methods as P, recover2 as R
rows = []
D = pd.read_csv("fishery_log.csv"); X = np.column_stack([np.ones(len(D)), D.x1.values]); y = D.y.values
sets = {"fishery": (X, y, 8)}
for f in ["tone_clean", "tone_out"]:
    T = pd.read_csv(f + ".csv"); sets[f] = (np.column_stack([np.ones(len(T)), T.stretchratio.values]), T.tuned.values, 8)
for name, (X, y, m) in sets.items():
    for s in range(20):
        th2, a2, info, th, a = R.algorithm1_r(X, y, 2, m, 100, np.random.default_rng(s))
        o = np.argsort(th2[:, 1] if name != "fishery" else th2[:, 0]); th2 = th2[o]
        rows.append(dict(data=name, seed=s, changed=info["accepted"], a1=a, a2=a2, b0_1=th2[0, 0], b1_1=th2[0, 1], b0_2=th2[1, 0], b1_2=th2[1, 1]))
R_ = pd.DataFrame(rows); R_.to_csv("alg2_real.csv", index=False)
print(R_.groupby("data")[["changed", "a1", "a2", "b0_1", "b1_1", "b0_2", "b1_2"]].agg(["mean", "min", "max"]).round(3).T.to_string())
