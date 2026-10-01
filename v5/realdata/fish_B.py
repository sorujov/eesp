import sys; sys.path.insert(0, "../code")
import numpy as np, pandas as pd, recover2 as R
D = pd.read_csv("fishery_log.csv"); X = np.column_stack([np.ones(len(D)), D.x1.values]); y = D.y.values
rows = []
for K, m in [(2, 8), (3, 16)]:
    for B in [100, 500, 1000]:
        for s in range(20):
            th2, a2, info, th, a = R.algorithm1_r(X, y, K, m, B, np.random.default_rng(1000 * B + s))
            o = np.argsort(th2[:, 0]); th2 = th2[o]
            rows.append(dict(K=K, B=B, seed=s, **{f"b0_{k+1}": th2[k, 0] for k in range(K)}, flagged=a2))
        d = pd.DataFrame([r for r in rows if r["K"] == K and r["B"] == B])
        print(K, B, d[[c for c in d.columns if c.startswith("b0")]].round(2).value_counts().head(3).to_dict(), flush=True)
pd.DataFrame(rows).to_csv("fish_B.csv", index=False)
